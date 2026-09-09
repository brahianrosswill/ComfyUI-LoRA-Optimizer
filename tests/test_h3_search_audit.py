"""Complete-plan audit failures are separate from negative benchmark findings."""
import copy
import json

import pytest

from scripts import h3_search_audit as audit
from scripts import h3_search_benchmark as benchmark
from tests.test_h3_search_benchmark import prepared  # Shared isolated-source fixture.


def rewrite(path, value):
    path.write_text(json.dumps(value))


def summarize(plan_path):
    root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    records = []
    for job in plan["jobs"]:
        directory = root / "runs" / job["id"]
        result = json.loads((directory / "result.json").read_text())
        result["telemetry_sha256"] = benchmark.digest(directory / "telemetry.json")
        records.append(result)
    value = dict(plan_sha256=benchmark.digest(plan_path), records=records,
                 pairs=benchmark.measured_summary(records),
                 comparisons=benchmark.compare_searches(records, root),
                 quality_claim=False, note="Synthetic isolated audit fixture, not H3 measurements")
    rewrite(root / "summary.json", value)


@pytest.fixture
def complete(prepared):
    plan = json.loads(prepared.read_text())
    for job in plan["jobs"]:
        directory = prepared.parent / "runs" / job["id"]
        directory.mkdir(parents=True)
        configs = [dict(merge_mode="per_prefix_auto", strategy_set=strategy)
                   for strategy in ("full", "no_slerp", "basic")]
        options = None
        if job["phase"] != "disconnected":
            options = dict(np_lora=True, ct_merge=True)
            configs.extend([dict(merge_mode="weighted_sum", optimization_mode="global", experimental_baseline=True),
                            dict(merge_mode="np_lora", experimental={}),
                            dict(merge_mode="ct_merge", experimental={})])
        tuner = dict(analysis_summary=dict(prefix_count=312), experimental_options=options,
                     top_n=[dict(config=config, score_final=.8-index*.01,
                                 per_prefix_decisions={str(i): "weighted_average" for i in range(312)})
                            for index, config in enumerate(configs)])
        benchmark.save_new(directory / "tuner_data.json", tuner)
        (directory / "merge.log").write_text("Synthetic complete merge log\n")
        (directory.parent / (job["id"] + ".log")).write_text("Synthetic worker log\n")
        benchmark.save_new(directory / "telemetry.json", {"synthetic": True})
        result = dict(job=job, plan_sha256=benchmark.digest(prepared), winner_native_targets=208,
                      snapshots=plan["snapshots"][job["version"]]["files"], winner_payload_sha256="a"*64,
                      candidate_counts=benchmark.validate_candidate_budget(tuner, options),
                      actual_experimental_options=options, tuner_sha256=benchmark.digest(directory / "tuner_data.json"),
                      log_sha256=benchmark.digest(directory / "merge.log"),
                      elapsed_seconds=(20. if job["phase"] != "measured" else
                                       10. + job["repeat"] + (2. if job["version"] == "current" else 0.)),
                      peak_cuda_allocated=1234, torch_version="synthetic", cuda_device="synthetic",
                      cuda_matmul_allow_tf32=False, matmul_precision="highest", svd_kernel_loaded=False,
                      triton_available=False, triton_disabled=True)
        benchmark.save_new(directory / "result.json", result)
    summarize(prepared)
    return prepared


def test_complete_audit_preserves_all_trials_and_negative_performance(complete):
    result = audit.audit(complete)
    assert result["trials"] == len(result["records"]) == 40
    assert len(result["artifact_sha256"]) == 40
    assert all(len(pins) == 5 for pins in result["artifact_sha256"].values())
    assert result["all_fidelity_gates_pass"]
    assert not result["quality_claim"] and not result["shipping_qualification_complete"]
    for pair in result["pairs"].values():
        assert pair["current_over_baseline_median_time"] == 13/11


@pytest.mark.parametrize("defect", ["no_summary", "stopped", "partial", "duplicate_record", "extra_run",
                                  "log", "tuner", "telemetry", "receipt", "adapter", "saved_ratio", "saved_comparison"])
def test_incomplete_or_changed_evidence_is_not_certified(complete, defect):
    root = complete.parent
    summary = json.loads((root / "summary.json").read_text())
    directory = root / "runs" / summary["records"][0]["job"]["id"]
    if defect == "no_summary":
        (root / "summary.json").unlink()
    elif defect == "stopped":
        rewrite(root / "stopped.json", {})
    elif defect == "partial":
        summary["records"].pop()
    elif defect == "duplicate_record":
        summary["records"][-1] = copy.deepcopy(summary["records"][0])
    elif defect == "extra_run":
        (root / "runs" / "unscheduled").mkdir()
    elif defect in {"log", "tuner", "telemetry", "receipt"}:
        name = dict(log="merge.log", tuner="tuner_data.json", telemetry="telemetry.json", receipt="result.json")[defect]
        with (directory / name).open("a") as stream:
            stream.write("corruption")
    elif defect == "adapter":
        plan = json.loads(complete.read_text())
        audit.Path(next(iter(plan["adapters"].values()))[0]["path"]).write_bytes(b"changed")
    elif defect == "saved_ratio":
        summary["pairs"][benchmark.PAIR_ORDER[0]]["current_over_baseline_median_time"] = .5
    else:
        summary["comparisons"][benchmark.PAIR_ORDER[0]]["measured"][0]["winner_payload_equal"] = False
    if defect in {"partial", "duplicate_record", "saved_ratio", "saved_comparison"}:
        rewrite(root / "summary.json", summary)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit(complete)


@pytest.mark.parametrize("defect", ["default_output", "repeat_output", "runtime"])
def test_reproducibility_or_default_regressions_remain_reportable_failures(complete, defect):
    job = next(job for job in benchmark.schedule() if job["version"] == "current"
               and job["phase"] == ("disconnected" if defect == "default_output" else "measured"))
    path = complete.parent / "runs" / job["id"] / "result.json"
    result = json.loads(path.read_text())
    result["torch_version" if defect == "runtime" else "winner_payload_sha256"] = (
        "changed runtime" if defect == "runtime" else "b"*64)
    rewrite(path, result)
    summarize(complete)
    report = audit.audit(complete)
    assert report["complete_integrity_audit"]
    assert not report["all_fidelity_gates_pass"]
    assert not report["shipping_qualification_complete"]


def test_equal_length_but_different_target_coverage_is_rejected(complete):
    job = benchmark.schedule()[0]
    directory = complete.parent / "runs" / job["id"]
    path = directory / "tuner_data.json"
    tuner = json.loads(path.read_text())
    del tuner["top_n"][1]["per_prefix_decisions"]["0"]
    tuner["top_n"][1]["per_prefix_decisions"]["unknown"] = "weighted_average"
    rewrite(path, tuner)
    receipt_path = directory / "result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["tuner_sha256"] = benchmark.digest(path)
    rewrite(receipt_path, receipt)
    summarize(complete)
    with pytest.raises(ValueError, match="per-target"):
        audit.audit(complete)


def test_same_winner_configuration_with_consistently_changed_payload_is_a_failure(complete):
    for job in benchmark.schedule():
        if job["version"] != "current" or job["phase"] != "measured":
            continue
        path = complete.parent / "runs" / job["id"] / "result.json"
        result = json.loads(path.read_text())
        result["winner_payload_sha256"] = "b"*64
        rewrite(path, result)
    summarize(complete)
    report = audit.audit(complete)
    assert report["fidelity_gates"]["measured_winners_repeatable"]
    assert not report["fidelity_gates"]["unchanged_winner_configs_preserve_payloads"]
    assert not report["all_fidelity_gates_pass"]


def test_nonfinite_json_is_rejected_without_a_quality_interpretation(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"score": NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        audit.read_json(path)
