"""Offline audit of a COMPLETE paired search run; no GPU, queue or reruns.

Run only after the timed parent exits: adapter rehashing is intentionally outside
the benchmark. Preserve negative findings; integrity success is not a shipping,
visual-quality or performance-improvement claim.
"""
import argparse
import json
from pathlib import Path
import re

try:
    from . import h3_search_benchmark as benchmark
except ImportError:
    import h3_search_benchmark as benchmark


def read_json(path):
    def nonfinite(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(path.read_text(), parse_constant=nonfinite)


def audit(plan_path):
    root = plan_path.resolve().parent
    summary_path = root / "summary.json"
    if (root / "stopped.json").exists() or not summary_path.is_file():
        raise ValueError("A complete, non-stopped benchmark summary is required")
    plan = benchmark.validate_plan(plan_path)
    summary = read_json(summary_path)
    plan_sha = benchmark.digest(plan_path)
    if summary.get("plan_sha256") != plan_sha or summary.get("quality_claim") is not False:
        raise ValueError("Summary provenance or claim scope changed")
    records = summary["records"]
    if [record["job"] for record in records] != plan["jobs"]:
        raise ValueError("All forty scheduled records, in order, are required")
    expected_dirs = {job["id"] for job in plan["jobs"]}
    if {path.name for path in (root / "runs").iterdir() if path.is_dir()} != expected_dirs:
        raise ValueError("Missing or extra trial directories")

    artifacts, sources = {}, {}
    # Rehash each unique actual adapter, not just its metadata or a sampled
    # tensor. This audit never loads model weights into torch or ComfyUI.
    for entries in plan["adapters"].values():
        for entry in entries:
            path = entry["path"]
            if path not in sources:
                sources[path] = benchmark.digest(path)
            if sources[path] != entry["sha256"]:
                raise ValueError("Adapter payload changed after the benchmark")

    runtime_fields = ("torch_version", "cuda_device", "cuda_matmul_allow_tf32",
                      "matmul_precision", "svd_kernel_loaded", "triton_available", "triton_disabled")
    runtime_profiles = []
    for record in records:
        job = record["job"]
        directory = root / "runs" / job["id"]
        if directory.resolve() != directory:
            raise ValueError("Trial directory redirected outside the frozen run")
        paths = {name: directory / name for name in
                 ("result.json", "tuner_data.json", "merge.log", "telemetry.json")}
        paths["worker.log"] = root / "runs" / (job["id"] + ".log")
        pins = {name: benchmark.digest(path) for name, path in paths.items()}
        receipt = read_json(paths["result.json"])
        if receipt != {key: value for key, value in record.items() if key != "telemetry_sha256"}:
            raise ValueError("Summary record differs from the actual worker receipt")
        for artifact, field in (("tuner_data.json", "tuner_sha256"),
                                ("merge.log", "log_sha256"),
                                ("telemetry.json", "telemetry_sha256")):
            if pins[artifact] != record[field]:
                raise ValueError(f"Changed {artifact} for {job['id']}")
        if (record["plan_sha256"] != plan_sha or record["winner_native_targets"] != 208
                or record["snapshots"] != plan["snapshots"][job["version"]]["files"]
                or not re.fullmatch(r"[0-9a-f]{64}", record["winner_payload_sha256"])):
            raise ValueError("Worker scope, implementation or payload identity changed")
        tuner = read_json(paths["tuner_data.json"])
        options = None if job["phase"] == "disconnected" else {"np_lora": True, "ct_merge": True}
        if benchmark.validate_candidate_budget(tuner, options) != record["candidate_counts"]:
            raise ValueError("Candidate count receipt differs from the actual tuner data")
        if record["actual_experimental_options"] != tuner.get("experimental_options"):
            raise ValueError("Experimental options receipt differs from the actual tuner data")
        decisions = [row.get("per_prefix_decisions", {}) for row in tuner["top_n"]]
        if any(len(row) != 312 or set(row) != set(decisions[0]) for row in decisions):
            raise ValueError("Incomplete or inconsistent per-target candidate decisions")
        identities = [benchmark.candidate_identity(row) for row in tuner["top_n"]]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate configuration identities would hide comparisons")
        profile = {key: record[key] for key in runtime_fields}
        if profile not in runtime_profiles:
            runtime_profiles.append(profile)
        artifacts[job["id"]] = pins

    pairs = benchmark.measured_summary(records)
    comparisons = benchmark.compare_searches(records, root)
    if summary["pairs"] != pairs or summary["comparisons"] != comparisons:
        raise ValueError("Saved summary disagrees with recomputed raw-record comparisons")
    gates = dict(
        disconnected_outputs_preserved=all(
            row["disconnected_configs_scores_and_payload_equal"] for row in comparisons.values()),
        common_candidate_scores_and_decisions_preserved=all(
            change["current_minus_baseline"] == 0 and change["per_prefix_decisions_equal"]
            for row in comparisons.values() for trial in row["measured"]
            for change in trial["common_candidate_score_changes"]),
        unchanged_winner_configs_preserve_payloads=all(
            trial["baseline_winner"] != trial["current_winner"] or trial["winner_payload_equal"]
            for row in comparisons.values() for trial in row["measured"]),
        measured_winners_repeatable=all(
            len(set(stats[version]["winner_payload_sha256"])) == 1
            for stats in pairs.values() for version in ("baseline", "current")),
        runtime_profile_consistent=len(runtime_profiles) == 1)
    # These gates report adverse evidence, never discard it or rewrite a
    # supposedly passing result. Changed winners are allowed and listed below.
    return dict(version=1, complete_integrity_audit=True, trials=len(records),
                scope="Four predeclared pairs; controls/warm-ups retained; three measured repeats per version/pair",
                plan_path=str(plan_path.resolve()), plan_sha256=plan_sha,
                summary_sha256=benchmark.digest(summary_path),
                audit_helper_sha256=benchmark.digest(Path(__file__)),
                adapter_sha256=sources, artifact_sha256=artifacts,
                fidelity_gates=gates, all_fidelity_gates_pass=all(gates.values()),
                runtime_profiles=runtime_profiles, pairs=pairs, comparisons=comparisons,
                records=records, quality_claim=False, shipping_qualification_complete=False,
                limitations=summary["note"] + " Winner hashes were computed by the pinned workers over all raw patches; "
                    "patch payloads were not retained for an independent second hashing. "
                    "This report verifies receipts and source identities, not audiovisual quality.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.plan)
    benchmark.save_new(args.out, result)
    print(json.dumps(dict(trials=result["trials"], fidelity_gates=result["fidelity_gates"],
                          report=str(args.out), sha256=benchmark.digest(args.out))))


if __name__ == "__main__":
    main()
