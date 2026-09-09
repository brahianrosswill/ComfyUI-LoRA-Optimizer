"""Research-only character merge configuration tests; no server or GPU."""
import pytest

from scripts.h3_merge_study import PAIRS, merge_options
from scripts.h3_native_export_check import component_label, component_rows, effective_mode, factor_pair, replay_policy, require_coverage
from scripts.h3_dense_export_check import export_groups


def test_dense_checker_consumes_exact_union_of_native_dense_and_unique_factors():
    tensors = {"__metadata__": {}, "layer.diff": {}, "token.lora_up.weight": {},
               "token.lora_down.weight": {}, "token.alpha": {}}
    groups = export_groups(tensors, {"layer.weight", "token.weight"})
    assert groups == {"layer.weight": ["layer.diff"],
                      "token.weight": ["token.alpha", "token.lora_down.weight", "token.lora_up.weight"]}
    for changed in ({**tensors, "layer.alpha": {}},
                    {**tensors, "token.diff": {}},
                    {k: v for k, v in tensors.items() if k != "token.alpha"}):
        with pytest.raises(ValueError):
            export_groups(changed, {"layer.weight", "token.weight"})


def test_character_roles_are_first_and_crossed_with_both_effects():
    for character in ("sully", "series30"):
        for effect in ("cinema", "combat"):
            subject, second = PAIRS[f"{character}_{effect}"]
            assert subject.startswith("character/h3-identity-study-20260908/")
            assert second == PAIRS["combat_cinema"][effect == "cinema"]


@pytest.mark.parametrize("mode", ["slerp", "ties"])
def test_explicit_stable_arm_is_global_not_an_automatic_per_prefix_winner(mode):
    options = merge_options(mode, "smart")
    assert options == dict(optimization_mode="global", merge_strategy_override=mode,
                          _experimental_config=None, patch_compression="smart")


def test_existing_additive_and_experimental_configuration_is_unchanged():
    assert merge_options("additive", "aggressive") == dict(
        optimization_mode="additive", merge_strategy_override="",
        _experimental_config=None, patch_compression="aggressive")
    np = merge_options("np_lora", "smart")
    assert np["_experimental_config"] == dict(version=1, subject_slot=1, style_slot=2,
                                              strength=.5, rank=0, energy=1.)
    assert merge_options("ct_merge", "smart")["_experimental_config"] == dict(
        version=1, common_rank=4, residual_rank=16, scale=1.)
    with pytest.raises(ValueError):
        merge_options("unknown", "smart")


def test_native_checker_demands_exact_coverage_including_unique_targets():
    require_coverage([{"q": 1}, {"q": 2, "token": 3}], {"q": 4, "token": 5})
    for export in ({"q": 4}, {"q": 4, "token": 5, "extra": 6}, {}):
        with pytest.raises(ValueError, match="coverage"):
            require_coverage([{"q": 1}, {"q": 2, "token": 3}], export)


def test_explicit_additive_overrides_autodetected_export_metadata_mode():
    assert effective_mode({"merge_mode": "weighted_average", "merge_optimization_mode": "additive"}) == "weighted_sum"
    assert effective_mode({"merge_mode": "weighted_average", "merge_optimization_mode": "global"}) == "weighted_average"
    assert effective_mode({"merge_mode": "weighted_average", "merge_optimization_mode": "per_prefix"}) == "per_prefix"
    with pytest.raises(ValueError, match="experimental"):
        effective_mode({"merge_mode": "np_lora", "merge_optimization_mode": "additive", "merge_experimental": "{}"})


def test_native_replay_policy_keeps_qkv_modes_and_exact_auto_strength():
    import copy
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    decisions = {component_label(target, c): mode for c, mode in enumerate(("weighted_sum", "slerp", "weighted_average"))}
    assert list(decisions) == ["diffusion_model.blocks.0.attn.to_q", "diffusion_model.blocks.0.attn.to_k", "diffusion_model.blocks.0.attn.to_v"]
    assert component_label("diffusion_model.blocks.0.mlp.fc1.weight", 0) == "diffusion_model.blocks.0.mlp.fc1"
    config = dict(optimization_mode="per_prefix", sparsification="disabled", merge_refinement="none", auto_strength="enabled")
    manifest = dict(action="replay", strengths=[1., -.8], replay_per_prefix_decisions=decisions,
        selected=dict(config=config, per_prefix_decisions=dict(decisions)),
        replay_auto_strength=dict(model_scale=.5, original_model_strengths=[1., -.8], model_strengths=[.5, -.4]))
    metadata = {"merge_" + k: v for k, v in config.items()}
    assert replay_policy(manifest, metadata) == (decisions, .5)
    bad = copy.deepcopy(manifest)
    bad["replay_auto_strength"]["model_strengths"][1] = .4
    with pytest.raises(ValueError, match="strength"):
        replay_policy(bad, metadata)
    bad = copy.deepcopy(manifest)
    bad["replay_per_prefix_decisions"][next(iter(decisions))] = "weighted_average"
    with pytest.raises(ValueError, match="different numerical"):
        replay_policy(bad, metadata)
    with pytest.raises(ValueError, match="metadata"):
        replay_policy(manifest, {**metadata, "merge_auto_strength": "disabled"})


def test_native_qkv_component_slices_preserve_all_rows():
    rows = component_rows("diffusion_model.blocks.0.attn.qkv_proj.weight", (12, 8))
    assert [(r.start, r.stop) for r in rows] == [(0, 4), (4, 8), (8, 12)]
    assert len(component_rows("diffusion_model.blocks.0.mlp.fc1.weight", (12, 8))) == 1
    with pytest.raises(ValueError):
        component_rows("x.qkv_proj.weight", (13, 8))


def test_native_factor_reference_preserves_alpha_sign_and_fp32():
    import torch
    from types import SimpleNamespace
    up = torch.arange(24, dtype=torch.float32).reshape(12, 2).bfloat16()
    down = torch.ones(2, 8, dtype=torch.bfloat16)
    patch = SimpleNamespace(weights=(up, down, -3., None, None, None))
    b, a = factor_pair(patch, slice(4, 8), "cpu")
    assert b.dtype == a.dtype == torch.float32
    torch.testing.assert_close(b @ a, up[4:8].float() @ down.float() * -1.5)
    patch.weights = (up, down, 2., None, torch.ones(12), None)
    with pytest.raises(ValueError, match="plain linear"):
        factor_pair(patch, slice(0, 4), "cpu")


def test_character_protocol_keeps_all_controls_and_separate_unused_holdout():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/research/data/2026-09-08-h3-character-merge-protocol.json").read_text())
    assert set(protocol["variants"]) == {"base", "character_only", "effect_only", "additive",
                                         "slerp", "ties", "stable_tuner_winner", "np_lora", "ct_merge"}
    calibration = set(protocol["splits"]["calibration"]["seeds"])
    heldout = set(protocol["splits"]["heldout"]["seeds"])
    assert len(calibration) == len(heldout) == 2 and calibration.isdisjoint(heldout)
    previous = {2026090803, 2026090804, 2026090811, 2026090812, 2026090821, 2026090822}
    assert (calibration | heldout).isdisjoint(previous)
    assert len(protocol["characters"]) * len(protocol["effects"]) * 9 * 4 == 144
    assert protocol["profile"]["frames"] % 17 == 5
    assert protocol["profile"]["turbo"] is False
    expanded = []
    for name, character in protocol["characters"].items():
        for key, template in protocol["templates"].items():
            prompt = template.format(**character)
            assert "{" not in prompt and "}" not in prompt
            assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
            assert prompt.count("[Shot") == 1
            assert prompt.index("overall_soundscape:") < prompt.index("non_diegetic_music:")
            assert character["subject"] in prompt
            assert ("ASTROCINEMAV01K2T" in prompt) == key.startswith("cinema")
            expanded.append(prompt)
    assert len(set(expanded)) == 8
