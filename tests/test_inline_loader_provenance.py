"""Stock-loader call attribution, including equal strengths and branch holes."""
from unittest import mock

import pytest
import torch

from tests.test_lora_optimizer import lora_optimizer as m
from tests.test_inline_optimizer import _AttachModel, _AttachClip, _adapter, _entry


def loader_class(sources):
    class Loader:
        def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
            if strength_model == strength_clip == 0:
                return model, clip
            model = model.clone() if model is not None else None
            clip = clip.clone() if clip is not None else None
            for host, strength, payloads in (
                    (model, strength_model, sources[lora_name][0]),
                    (clip.patcher if clip is not None else None,
                     strength_clip, sources[lora_name][1])):
                if host is not None:
                    for key, payload in payloads.items():
                        host.patches.setdefault(key, []).append(_entry(strength, payload))
            return model, clip
    m._install_lora_name_stamp(Loader)
    return Loader


def captured(model, clip=None, options=None):
    node = m.LoRAOptimizerInline()
    seen = {}
    def merge(base, stack, output_strength, **kwargs):
        seen.update(model=base, clip=kwargs.get("clip"), stack=stack)
        return base, kwargs.get("clip"), "report", None, None
    with mock.patch.object(node, "optimize_merge", side_effect=merge):
        result = node.execute_inline(model, 1., clip=clip, chain_options=options)
    return seen, result


@pytest.mark.parametrize("overlap", [True, False])
def test_same_strength_calls_keep_each_files_unique_targets(overlap):
    a = {"diffusion_model.a.weight": _adapter()}
    b = {"diffusion_model.b.weight": _adapter()}
    if overlap:
        a["diffusion_model.shared.weight"] = _adapter()
        b["diffusion_model.shared.weight"] = _adapter()
    Loader = loader_class({"a.safetensors": (a, {}), "b.safetensors": (b, {})})
    upstream = _AttachModel()
    first, _ = Loader().load_lora(upstream, None, "a.safetensors", 1, 0)
    second, _ = Loader().load_lora(first, None, "b.safetensors", 1, 0)
    seen, _ = captured(second)
    assert len(seen["stack"]) == 2
    for item, name, expected in zip(seen["stack"], ("a.safetensors", "b.safetensors"), (a, b)):
        assert item["_resolved_file_name"] == name
        assert set(item["lora"]) == set(expected)
        assert all(item["lora"][key] is payload for key, payload in expected.items())
    assert not upstream.patches and set(first.patches) == set(a)
    assert not seen["model"].patches


def test_model_only_hole_at_equal_strength_keeps_clip_on_its_file():
    sources = {name: ({f"diffusion_model.{name}.weight": _adapter()},
                      {f"clip_l.{name}.weight": _adapter()}) for name in ("a", "b", "c")}
    Loader = loader_class(sources)
    model, clip = Loader().load_lora(_AttachModel(), _AttachClip(_AttachModel()), "a", 1, 1)
    model, _ = Loader().load_lora(model, None, "b", 1, 0)
    model, clip = Loader().load_lora(model, clip, "c", 1, 1)
    seen, _ = captured(model, clip, dict(visibility="simple", slots=[{}, {"enabled": False}, {}]))
    assert [item["_resolved_file_name"] for item in seen["stack"]] == ["a", "c"]
    for item, name in zip(seen["stack"], ("a", "c")):
        assert set(item["lora"]) == {f"diffusion_model.{name}.weight", f"clip_l.{name}.weight"}
    assert not seen["model"].patches and not seen["clip"].patcher.patches


def test_repeated_cached_file_is_two_calls_and_slot_strengths_are_independent():
    a = {"diffusion_model.a.weight": _adapter()}
    Loader = loader_class({"a": (a, {})})
    loader = Loader()
    first, _ = loader.load_lora(_AttachModel(), None, "a", 1, 0)
    second, _ = loader.load_lora(first, None, "a", 1, 0)
    seen, result = captured(second, options=dict(visibility="simple", slots=[
        {"strength": .25}, {"strength": -.5}]))
    assert [item["strength"] for item in seen["stack"]] == [.25, -.5]
    assert [item["_resolved_file_name"] for item in seen["stack"]] == ["a", "a"]
    assert "exact stock-loader" in result[2]
    records = second.get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH)
    assert len(records) == 2 and records[0]["call_id"] != records[1]["call_id"]
    assert len(first.get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH)) == 1
    assert seen["model"].get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH) == []
    assert seen["model"].get_attachment(m.LORAOPT_CHAIN_NAMES_ATTACH) == []


def test_te_only_call_and_zero_model_strength_keep_exact_slots():
    sources = {"te": ({}, {"clip_l.te.weight": _adapter()}),
               "both": ({"diffusion_model.a.weight": _adapter()}, {"clip_l.a.weight": _adapter()})}
    Loader = loader_class(sources)
    model, clip = Loader().load_lora(_AttachModel(), _AttachClip(_AttachModel()), "te", 1, 1)
    model, clip = Loader().load_lora(model, clip, "both", 0, -.5)
    seen, _ = captured(model, clip)
    assert [item["_resolved_file_name"] for item in seen["stack"]] == ["te", "both"]
    assert [set(item["lora"]) for item in seen["stack"]] == [{"clip_l.te.weight"}, {"clip_l.a.weight"}]
    assert [item["clip_strength"] for item in seen["stack"]] == [1., -.5]


def test_partial_record_cannot_claim_whole_file_even_if_clip_is_complete():
    sources = {"a": ({"diffusion_model.a.weight": _adapter(), "diffusion_model.b.weight": _adapter()},
                      {"clip_l.a.weight": _adapter()})}
    Loader = loader_class(sources)
    model, clip = Loader().load_lora(_AttachModel(), _AttachClip(_AttachModel()), "a", 1, 1)
    altered = model.clone()
    altered.patches.pop("diffusion_model.b.weight")
    seen, _ = captured(altered, clip)
    assert len(seen["stack"]) == 1
    assert "_resolved_file_name" not in seen["stack"][0]
    assert "diffusion_model.b.weight" in model.patches


def test_same_value_replacement_tuple_is_not_treated_as_the_recorded_patch():
    key = "diffusion_model.a.weight"
    original = _entry(1, _adapter())
    replacement = tuple(list(original))
    record = dict(call_id="call", name="a", entries=((key, original),), complete=True)
    assert m._loraopt_entry_provenance({key: [replacement]}, [record]) == {}
    assert m._loraopt_entry_provenance({key: [original]}, [record])[(key, id(original))]["complete"]


@pytest.mark.parametrize("blocked", ["dora", "set"])
def test_unsupported_patch_keeps_its_whole_loader_call_in_place(blocked):
    special = (_adapter(dora_scale=torch.ones(8)) if blocked == "dora"
               else ("set", (torch.ones(8, 8),)))
    sources = {"a": ({"diffusion_model.a.weight": _adapter(), "diffusion_model.special.weight": special}, {}),
               "b": ({"diffusion_model.b.weight": _adapter()}, {})}
    Loader = loader_class(sources)
    model, _ = Loader().load_lora(_AttachModel(), None, "a", 1, 0)
    model, _ = Loader().load_lora(model, None, "b", 1, 0)
    seen, _ = captured(model)
    assert [item["_resolved_file_name"] for item in seen["stack"]] == ["b"]
    assert set(seen["model"].patches) == set(sources["a"][0])
    records = seen["model"].get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH)
    assert len(records) == 1 and records[0]["name"] == "a"
    for key, entry in records[0]["entries"]:
        assert entry is model.patches[key][0]


def test_captured_clip_bias_resolves_via_its_real_weight_target():
    target = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj"
    groups = m.LoRAOptimizer()._build_target_groups(
        [target + ".bias"], {}, {"text_encoder.q": target + ".weight"})
    assert len(groups) == 1
    group = next(iter(groups.values()))
    assert group["is_clip"] and group["target_key"] == target + ".bias"
