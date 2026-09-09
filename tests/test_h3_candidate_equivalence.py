"""Offline scope/integrity tests; real GPU replays remain separately recorded."""
import copy
import json
import sys
import types

import pytest
import torch

from scripts import h3_candidate_equivalence as m


def tuner_for(pair):
    base = dict(merge_mode="per_prefix_auto", sparsification="disabled",
                sparsification_density=.7, dare_dampening=0., merge_refinement="none",
                auto_strength="enabled", optimization_mode="per_prefix")
    return dict(top_n=[dict(rank=i, config=dict(base, strategy_set=strategy),
                           score_final=.5,
                           per_prefix_decisions={f"layer{j}": "weighted_average" for j in range(312)})
                       for i, strategy in enumerate(("full", "no_slerp", "basic"), 1)],
                source_loras=[dict(name=name + ".safetensors", strength=.8) for name in m.PAIRS[pair]],
                decision_smoothing=.25, auto_strength_floor=-1., normalize_keys="enabled")


@pytest.fixture
def prepared(tmp_path):
    study = tmp_path / "study"
    sweeps = []
    for pair in sorted(m.PAIRS_REQUIRED):
        path = study / ("idm-" + pair.replace("_", "-") + "-tune-01") / "tuner_data.json"
        path.parent.mkdir(parents=True)
        m.save_new(path, tuner_for(pair))
        sweeps.append(dict(pair=pair, tuner_data_sha256=m.digest(path)))
    evidence = tmp_path / "evidence.json"
    m.save_new(evidence, dict(recorded_sweeps=sweeps))
    out = tmp_path / "validation"
    m.prepare(evidence, study, out)
    return out / "plan.json", evidence, study


def test_selection_only_reorders_without_mutating_any_original_setting():
    tuner = tuner_for("sully_combat")
    before = copy.deepcopy(tuner)
    for rank in (2, 3):
        selected = m.select_original_rank(tuner, rank)
        assert selected["top_n"][0] == tuner["top_n"][rank - 1]
        assert selected["top_n"][0]["rank"] == rank
        assert {k: v for k, v in selected.items() if k != "top_n"} == {
            k: v for k, v in tuner.items() if k != "top_n"}
        selected["top_n"][0]["config"]["auto_strength"] = "disabled"
    assert tuner == before


@pytest.mark.parametrize("rank", [1, 0, 4, True, 2.0])
def test_rejects_unobserved_or_ambiguous_rank(rank):
    with pytest.raises(ValueError):
        m.select_original_rank(tuner_for("sully_combat"), rank)


def test_all_eight_inputs_pinned_and_no_overwrite(prepared):
    path, evidence, study = prepared
    plan = m.validate_plan(path)
    assert len(plan["jobs"]) == 8
    for job in plan["jobs"]:
        assert m.digest(job["original"]) == job["original_sha256"]
        assert not m.Path(job["run"]).exists()
    with pytest.raises(FileExistsError):
        m.prepare(evidence, study, path.parent)


@pytest.mark.parametrize("field,value", [("auto_strength", "disabled"),
    ("sparsification", "dare"), ("merge_refinement", "refine"),
    ("strategy_set", "full"), ("sparsification_density", .9)])
def test_active_config_difference_is_not_claimed_duplicate(field, value):
    tuner = tuner_for("series30_cinema")
    tuner["top_n"][2]["config"][field] = value
    with pytest.raises(ValueError, match="plain-linear duplicate"):
        m.duplicate_evidence(tuner)


@pytest.mark.parametrize("defect", ["partial", "nonlinear", "changed_decision"])
def test_all_linear_target_decisions_required(defect):
    tuner = tuner_for("sully_cinema")
    if defect == "partial":
        for entry in tuner["top_n"]:
            del entry["per_prefix_decisions"]["layer0"]
    elif defect == "nonlinear":
        for entry in tuner["top_n"]:
            entry["per_prefix_decisions"]["layer0"] = "ties"
    else:
        tuner["top_n"][2]["per_prefix_decisions"]["layer0"] = "weighted_sum"
    with pytest.raises(ValueError):
        m.duplicate_evidence(tuner)


@pytest.mark.parametrize("defect", ["missing_job", "duplicate_job", "escape", "strengths", "source_hash", "selection_hash"])
def test_tampered_scope_fails_before_execution(prepared, defect):
    path, _, _ = prepared
    plan = json.loads(path.read_text())
    if defect == "missing_job":
        plan["jobs"].pop()
    elif defect == "duplicate_job":
        plan["jobs"][0] = plan["jobs"][1]
    elif defect == "escape":
        plan["jobs"][0]["run"] = str(path.parent.parent / "outside")
    elif defect == "strengths":
        plan["jobs"][0]["strengths"] = [.9, .8]
    elif defect == "source_hash":
        plan["source_sha256"]["lora_optimizer.py"] = "0" * 64
    else:
        plan["jobs"][0]["selection_sha256"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        m.validate_plan(path)


@pytest.mark.parametrize("queue", [{}, {"queue_running": []},
    {"queue_running": [[1]], "queue_pending": []},
    {"queue_running": [], "queue_pending": [[1]]}])
def test_unknown_or_occupied_queue_never_allows_gpu_work(monkeypatch, queue):
    monkeypatch.setattr(m, "request", lambda *a: queue)
    with pytest.raises(RuntimeError):
        m.require_idle("http://127.0.0.1:8189")


@pytest.fixture
def payload_reader(tmp_path, monkeypatch):
    paths = [tmp_path / "left", tmp_path / "right"]
    payloads = {str(path): {"up": torch.arange(35).reshape(7, 5).bfloat16(),
                           "alpha": torch.tensor(-2.), "down": torch.ones(5, 3)} for path in paths}
    for path in paths:
        path.write_bytes(b"fixture file identity, not a real safetensors encoding")
    class Reader:
        def __init__(self, path, **kwargs):
            self.values = payloads[path]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def keys(self):
            return list(self.values)
        def get_tensor(self, key):
            return self.values[key]
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace(safe_open=Reader))
    return paths, payloads


def test_chunked_payloads_cover_every_value_including_alpha_and_bf16(payload_reader):
    paths, _ = payload_reader
    report = m.compare_payloads(*paths, chunk_elements=4)
    assert report["all_tensor_bytes_equal"] and report["tensor_count"] == 3
    assert sum(item["numel"] for item in report["tensors"]) == 51
    assert next(item for item in report["tensors"] if item["key"] == "alpha")["shape"] == []
    assert report["metadata_compared"] is False


@pytest.mark.parametrize("defect", ["last_element", "alpha", "dtype", "shape", "missing", "extra", "nan", "signed_zero"])
def test_payload_mismatch_or_nonfinite_never_passes(payload_reader, defect):
    paths, payloads = payload_reader
    left, right = [payloads[str(p)] for p in paths]
    if defect == "last_element":
        right["up"][-1, -1] = 5
    elif defect == "alpha":
        right["alpha"] *= -1
    elif defect == "dtype":
        right["up"] = right["up"].float()
    elif defect == "shape":
        right["up"] = right["up"].T
    elif defect == "missing":
        del right["alpha"]
    elif defect == "extra":
        right["extra"] = torch.ones(1)
    elif defect == "nan":
        left["alpha"] = right["alpha"] = torch.tensor(float("nan"))
    else:
        left["alpha"], right["alpha"] = torch.tensor(0.), torch.tensor(-0.)
    with pytest.raises(ValueError):
        m.compare_payloads(*paths, chunk_elements=4)
