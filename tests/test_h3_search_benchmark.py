"""Protocol and measurement integrity, without Comfy/GPU execution."""
import copy
import hashlib
import json
import struct
import types

import pytest
import torch

from scripts import h3_search_benchmark as m


def test_schedule_balances_order_and_keeps_control_warmup_measured_separate():
    jobs = m.schedule()
    assert len(jobs) == len({j["id"] for j in jobs}) == 40
    for pair in m.PAIR_ORDER:
        for version in ("baseline", "current"):
            local = [j for j in jobs if j["pair"] == pair and j["version"] == version]
            assert [j["phase"] for j in local] == ["disconnected", "warmup", "measured", "measured", "measured"]
    first = [jobs[i]["version"] for i in range(0, len(jobs), 2) if jobs[i]["phase"] == "measured"]
    assert first.count("baseline") == first.count("current") == 6


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    baseline = b"isolated synthetic baseline for protocol tests"
    monkeypatch.setattr(m, "BASELINE_SHA", hashlib.sha256(baseline).hexdigest())
    def git(command, **kwargs):
        name = command[-1].split(":", 1)[1]
        return baseline if name == "lora_optimizer.py" else (m.REPO / name).read_bytes()
    monkeypatch.setattr(m.subprocess, "check_output", git)
    comfy = tmp_path / "Comfy with spaces"
    checkpoint = comfy / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    checkpoint.parent.mkdir(parents=True)
    data = json.dumps({"x.weight": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]}}).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(data)) + data + b"1234")
    loras = tmp_path / "loras"
    for pair in m.PAIR_ORDER:
        for name in m.PAIRS[pair]:
            path = loras / (name + ".safetensors")
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            m.save_new(path.with_suffix(".metadata.json"), dict(sha256=m.digest(path)))
    out = tmp_path / "benchmark"
    m.prepare(out, comfy, loras)
    return out / "plan.json"


def test_isolated_source_pins_and_no_overwrite(prepared):
    plan = m.validate_plan(prepared)
    assert plan["settings"]["output_mode"] == "merge"
    assert plan["snapshots"]["baseline"]["files"]["lora_optimizer.py"] == m.BASELINE_SHA
    assert plan["snapshots"]["current"]["files"]["lora_optimizer.py"] == m.digest(m.REPO / "lora_optimizer.py")
    with pytest.raises(FileExistsError):
        m.prepare(prepared.parent, m.Path(plan["comfy"]), m.Path(plan["loras"]))


@pytest.mark.parametrize("defect", ["missing_job", "reorder", "settings", "env", "root", "roles", "strength", "snapshot", "helper"])
def test_scope_or_provenance_tampering_stops_before_compute(prepared, defect):
    plan = json.loads(prepared.read_text())
    if defect == "missing_job":
        plan["jobs"].pop()
    elif defect == "reorder":
        plan["jobs"].reverse()
    elif defect == "settings":
        plan["settings"]["scoring_speed"] = "turbo"
    elif defect == "env":
        plan["environment"]["OMP_NUM_THREADS"] = "8"
    elif defect == "root":
        plan["root"] = str(prepared.parent.parent)
    elif defect == "roles":
        plan["adapters"]["sully_combat"].reverse()
    elif defect == "strength":
        plan["adapters"]["sully_combat"][0]["strength"] = .8
    elif defect == "snapshot":
        plan["snapshots"]["current"]["files"]["lora_optimizer.py"] = "0" * 64
    else:
        plan["helper_sha256"]["scripts/h3_search_benchmark.py"] = "0" * 64
    prepared.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        m.validate_plan(prepared)


def test_payload_hash_covers_nested_patches_scalar_alpha_offsets_bf16_and_tail():
    a = torch.arange(35).reshape(7, 5).bfloat16()
    patch = types.SimpleNamespace(weights=(a, torch.ones(5, 3), -2., None, None, None))
    value = {("layer.weight", (0, 5, 7)): patch}
    before = m.hash_payload(value, chunk_elements=4)
    assert before == m.hash_payload(copy.deepcopy(value), chunk_elements=11)
    a[-1, -1] += 2
    assert before != m.hash_payload(value, chunk_elements=4)
    assert m.hash_payload(torch.tensor(0.)) != m.hash_payload(torch.tensor(-0.))
    assert m.hash_payload(("diff", (torch.ones(1),))) != m.hash_payload(["diff", [torch.ones(1)]])


@pytest.mark.parametrize("value", [torch.tensor(float("nan")), torch.tensor(float("inf")), float("nan"), object()])
def test_unknown_or_nonfinite_payload_rejected(value):
    with pytest.raises(ValueError):
        m.hash_payload(value)


def records_fixture():
    return [dict(job=job, elapsed_seconds=(10000. if job["phase"] != "measured" else
                 (10. if job["version"] == "baseline" else 8.) + job["repeat"]),
                 peak_cuda_allocated=1, winner_payload_sha256="same") for job in m.schedule()]


def test_measurement_uses_all_three_repeats_and_excludes_warmups_controls():
    summary = m.measured_summary(records_fixture())
    for pair in summary.values():
        assert pair["baseline"]["seconds"] == [10., 11., 12.]
        assert pair["current"]["seconds"] == [8., 9., 10.]
        assert pair["current_over_baseline_median_time"] == 9/11


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "zero", "negative"])
def test_partial_or_invalid_timings_cannot_be_reported_as_complete(defect):
    records = records_fixture()
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records[-1] = records[-2]
    else:
        records[-1]["elapsed_seconds"] = {"nan": float("nan"), "zero": 0., "negative": -1.}[defect]
    with pytest.raises(ValueError):
        m.measured_summary(records)


def test_search_report_distinguishes_default_regression_from_new_admission(tmp_path):
    records = records_fixture()
    for record in records:
        job = record["job"]
        path = tmp_path / "runs" / job["id"] / "tuner_data.json"
        path.parent.mkdir(parents=True)
        rows = [dict(config=dict(strategy_set="full"), score_final=.8, per_prefix_decisions={"a":"slerp"}),
                dict(config=dict(strategy_set="no_slerp"), score_final=.7, per_prefix_decisions={"a":"weighted_average"})]
        if job["phase"] != "disconnected":
            rows.append(dict(config=dict(strategy_set="new" if job["version"] == "current" else "basic"), score_final=.6))
        m.save_new(path, dict(top_n=rows))
        record["tuner_sha256"] = m.digest(path)
    result = m.compare_searches(records, tmp_path)
    for pair in result.values():
        assert pair["disconnected_configs_scores_and_payload_equal"]
        assert len(pair["measured"]) == 3
        for run in pair["measured"]:
            assert run["admitted"] == [dict(strategy_set="new")]
            assert run["removed"] == [dict(strategy_set="basic")]
            assert run["winner_payload_equal"]
            assert all(c["current_minus_baseline"] == 0 for c in run["common_candidate_score_changes"])
    # A changed control payload must not be hidden by matching configuration.
    next(r for r in records if r["job"]["phase"] == "disconnected")["winner_payload_sha256"] = "changed"
    result = m.compare_searches(records, tmp_path)
    assert not result[m.PAIR_ORDER[0]]["disconnected_configs_scores_and_payload_equal"]


def test_run_refuses_partial_tree_without_submitting_or_restarting(prepared, monkeypatch):
    (prepared.parent / "runs").mkdir()
    monkeypatch.setattr(m, "require_idle", lambda *a: pytest.fail("No API access before partial-tree refusal"))
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("No process may restart"))
    with pytest.raises(FileExistsError):
        m.run(prepared)


def actual_budget():
    # Use the real experimental candidate constructor, which also inserts its
    # separate additive control. Do not assume that two methods mean two rows.
    spec = m.importlib.util.spec_from_file_location("h3_budget_fixture", m.REPO / "experimental_merge.py")
    module = m.importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extra, skipped = module.candidates(dict(np_lora=True, ct_merge=True), [{}, {}])
    assert not skipped
    stable = [dict(strategy_set=s) for s in ("full", "no_slerp", "basic")]
    return dict(analysis_summary=dict(prefix_count=312),
                top_n=[dict(config=c, score_final=.5) for c in stable + extra])


def test_real_experimental_constructor_requires_six_trials_not_five():
    tuner = actual_budget()
    assert m.validate_candidate_budget(tuner, {}) == dict(stable=3, additive_controls=1, experimental_methods=2, total=6)
    tuner["top_n"] = tuner["top_n"][:3]
    assert m.validate_candidate_budget(tuner, None) == dict(stable=3, additive_controls=0, experimental_methods=0, total=3)


@pytest.mark.parametrize("defect", ["missing_additive", "missing_np", "missing_ct", "extra_stable", "nonfinite", "partial_targets", "wrong_additive", "enabled_in_disconnected"])
def test_actual_budget_guards_all_controls_and_methods(defect):
    tuner = actual_budget()
    experimental = {}
    if defect.startswith("missing_"):
        tuner["top_n"].pop({"missing_additive": 3, "missing_np": 4, "missing_ct": 5}[defect])
    elif defect == "extra_stable":
        tuner["top_n"].append(copy.deepcopy(tuner["top_n"][0]))
    elif defect == "nonfinite":
        tuner["top_n"][0]["score_final"] = float("nan")
    elif defect == "partial_targets":
        tuner["analysis_summary"]["prefix_count"] = 2
    elif defect == "wrong_additive":
        tuner["top_n"][3]["config"]["merge_mode"] = "weighted_average"
    else:
        experimental = None
    with pytest.raises(ValueError):
        m.validate_candidate_budget(tuner, experimental)
