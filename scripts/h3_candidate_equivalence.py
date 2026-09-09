"""Full-size replay of the four observed F08 rank-2/rank-3 duplicates.

This does not alter the frozen video benchmark or replace its winners. The
existing selector and independent native-loader checker run in fresh serial
processes. Only a derived copy's ordering changes to select the original rank.
Exact stored tensor equality is stronger than equal scores/decision labels,
but is not a perceptual-quality or controlled-throughput measurement.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    from .h3_benchmark import digest, save_new
    from .h3_merge_study import PAIRS
    from .h3_render_study import request
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_merge_study import PAIRS
    from h3_render_study import request

REPO = Path(__file__).resolve().parents[1]
PAIRS_REQUIRED = {f"{character}_{effect}" for character in ("sully", "series30")
                  for effect in ("cinema", "combat")}
DEPENDENCIES = ("lora_optimizer.py", "experimental_merge.py", "kernel.py",
                "scripts/h3_merge_study.py", "scripts/h3_native_export_check.py",
                "scripts/h3_benchmark.py", "scripts/h3_render_study.py",
                "scripts/h3_candidate_equivalence.py")


def select_original_rank(tuner, rank):
    """Leave the selected record, settings and source stack entirely intact."""
    if type(rank) is not int or rank not in (2, 3):
        raise ValueError("Only the observed ranks 2 and 3 are in scope")
    entries = tuner["top_n"]
    if len(entries) != 3 or [entry["rank"] for entry in entries] != [1, 2, 3]:
        raise ValueError("Expected the original ordered top-three evidence")
    result = copy.deepcopy(tuner)
    result["top_n"] = [copy.deepcopy(entries[rank - 1])] + [
        copy.deepcopy(entry) for entry in entries if entry["rank"] != rank]
    return result


def duplicate_evidence(tuner):
    second, third = tuner["top_n"][1:3]
    configs = [dict(entry["config"]) for entry in (second, third)]
    strategies = [config.pop("strategy_set") for config in configs]
    decisions = second.get("per_prefix_decisions")
    if (strategies != ["no_slerp", "basic"] or configs[0] != configs[1]
            or not decisions or decisions != third.get("per_prefix_decisions")
            or len(decisions) != 312
            or set(decisions.values()) - {"weighted_sum", "weighted_average"}
            or configs[0]["optimization_mode"] != "per_prefix"
            or configs[0]["sparsification"] != "disabled"
            or configs[0]["merge_refinement"] != "none"
            or configs[0].get("experimental") is not None):
        raise ValueError("Evidence does not describe the claimed plain-linear duplicate")


def prepare(evidence_path, study_root, out):
    if out.exists():
        raise FileExistsError(out)
    evidence = json.loads(evidence_path.read_text())
    sweeps = evidence["recorded_sweeps"]
    if len(sweeps) != 4 or {s["pair"] for s in sweeps} != PAIRS_REQUIRED:
        raise ValueError("Require all four observed pairs, without duplicates")
    prepared = []
    # Validate the entire scope before creating any derived records.
    for sweep in sweeps:
        pair = sweep["pair"]
        original = study_root / ("idm-" + pair.replace("_", "-") + "-tune-01") / "tuner_data.json"
        if digest(original) != sweep["tuner_data_sha256"]:
            raise ValueError("Original tuner evidence changed")
        tuner = json.loads(original.read_text())
        duplicate_evidence(tuner)
        expected_names = [name + ".safetensors" for name in PAIRS[pair]]
        if [item["name"] for item in tuner["source_loras"]] != expected_names:
            raise ValueError("Tuner source stack differs from pair")
        for rank in (2, 3):
            prepared.append((pair, rank, original, select_original_rank(tuner, rank)))
    out.mkdir(parents=True)
    jobs = []
    for pair, rank, original, derived in prepared:
        path = out / f"{pair}-rank{rank}-selection.json"
        save_new(path, derived)
        jobs.append(dict(pair=pair, original_rank=rank, original=str(original.resolve()),
                         original_sha256=digest(original), selection=str(path.resolve()),
                         selection_sha256=digest(path),
                         strengths=[item["strength"] for item in derived["source_loras"]],
                         run=str((out / f"{pair}-rank{rank}").resolve())))
    plan = dict(version=1, created_at=time.time(), evidence=str(evidence_path.resolve()),
                evidence_sha256=digest(evidence_path), jobs=jobs,
                source_sha256={name: digest(REPO / name) for name in DEPENDENCIES},
                merge_seed=2026090800, queue_base="http://127.0.0.1:8189",
                quality_claim=False, timing_controlled=False,
                note="Derived selector inputs retain original records; only top_n ordering changes. "
                     "No baseline overwrite, model installation, server start/free/cancel, or generation.")
    save_new(out / "plan.json", plan)
    return plan


def validate_plan(path):
    plan = json.loads(path.read_text())
    if (plan.get("version") != 1 or plan.get("queue_base") != "http://127.0.0.1:8189"
            or plan.get("merge_seed") != 2026090800
            or set(plan["source_sha256"]) != set(DEPENDENCIES)):
        raise ValueError("Unexpected equivalence plan")
    for name, sha in plan["source_sha256"].items():
        if digest(REPO / name) != sha:
            raise ValueError("Implementation changed since plan preparation")
    if digest(plan["evidence"]) != plan["evidence_sha256"]:
        raise ValueError("Duplicate evidence changed")
    identities = [(j["pair"], j["original_rank"]) for j in plan["jobs"]]
    if len(identities) != 8 or set(identities) != {(p, r) for p in PAIRS_REQUIRED for r in (2, 3)}:
        raise ValueError("Require all eight replays")
    for job in plan["jobs"]:
        for field in ("original", "selection"):
            if digest(job[field]) != job[field + "_sha256"]:
                raise ValueError("Selector input changed")
        original = json.loads(Path(job["original"]).read_text())
        duplicate_evidence(original)
        derived = json.loads(Path(job["selection"]).read_text())
        if derived != select_original_rank(original, job["original_rank"]):
            raise ValueError("Selection changes more than ordering")
        if job["strengths"] != [item["strength"] for item in original["source_loras"]]:
            raise ValueError("Strengths changed")
        if Path(job["run"]).resolve() != path.parent.resolve() / f"{job['pair']}-rank{job['original_rank']}":
            raise ValueError("Run path escapes the prepared scope")
    return plan


def require_idle(base):
    queue = request(base, "/queue")
    if queue.get("queue_running") != [] or queue.get("queue_pending") != []:
        raise RuntimeError("Queue is not explicitly empty; no GPU work started")
    return queue


def compare_payloads(left, right, *, chunk_elements=2 ** 20):
    """Read every tensor, including alpha; metadata may legitimately differ."""
    import hashlib
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    if chunk_elements <= 0:
        raise ValueError("Positive chunk size required")
    records = []
    with safe_open(str(left), framework="pt", device="cpu") as a, \
            safe_open(str(right), framework="pt", device="cpu") as b:
        if set(a.keys()) != set(b.keys()) or not a.keys():
            raise ValueError("Export key coverage differs or is empty")
        for key in sorted(a.keys()):
            x, y = a.get_tensor(key), b.get_tensor(key)
            if x.dtype != y.dtype or x.shape != y.shape or x.numel() == 0:
                raise ValueError("Export tensor shape/dtype differs or is empty")
            shape = list(x.shape)
            x, y = x.reshape(-1), y.reshape(-1)
            hashes = [hashlib.sha256(), hashlib.sha256()]
            for start in range(0, x.numel(), chunk_elements):
                chunks = [value[start:start + chunk_elements] for value in (x, y)]
                for h, chunk in zip(hashes, chunks):
                    if not torch.isfinite(chunk).all().item():
                        raise ValueError("Non-finite export")
                    h.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
            identities = [h.hexdigest() for h in hashes]
            if identities[0] != identities[1]:
                raise ValueError(f"Stored tensor bytes differ: {key}")
            records.append(dict(key=key, shape=shape,
                                dtype=str(x.dtype), numel=x.numel(), sha256=identities[0]))
    return dict(all_tensor_bytes_equal=True, tensor_count=len(records), tensors=records,
                left_sha256=digest(left), right_sha256=digest(right), metadata_compared=False)


def run(plan_path, comfy, python):
    plan = validate_plan(plan_path)
    result_path = plan_path.parent / "result.json"
    if result_path.exists() or any(Path(j["run"]).exists() for j in plan["jobs"]):
        raise FileExistsError("Existing replay/result; inspect partial work, never restart or overwrite")
    results = []
    for job in plan["jobs"]:
        require_idle(plan["queue_base"])
        run_dir = Path(job["run"])
        command = [str(python), str(REPO / "scripts/h3_merge_study.py"), "replay",
                   "--pair", job["pair"], "--comfy", str(comfy), "--out", str(run_dir),
                   "--strengths", *map(str, job["strengths"]), "--tuner-data", job["selection"],
                   "--merge-seed", str(plan["merge_seed"])]
        subprocess.run(command, check=True)
        require_idle(plan["queue_base"])
        subprocess.run([str(python), str(REPO / "scripts/h3_native_export_check.py"),
                        "--run", str(run_dir), "--comfy", str(comfy)], check=True)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        selection = json.loads(Path(job["selection"]).read_text())["top_n"][0]
        check = json.loads((run_dir / "export_check.json").read_text())
        if (manifest["selected"] != selection
                or check.get("groups_checked") != 312 or check.get("native_targets") != 208
                or check.get("all_source_and_export_keys_consumed") is not True
                or check.get("all_targets") is not True
                or len(check["results"]) != 312
                or not all(r.get("finite") is True for r in check["results"])
                or check["manifest_sha256"] != digest(run_dir / "manifest.json")
                or check["export_sha256"] != digest(manifest["export"])):
            raise ValueError("Incomplete, changed or mismatched native replay audit")
        # The native checker independently validates full source/target coverage.
        results.append(dict(job, manifest_sha256=digest(run_dir / "manifest.json"),
                            check_sha256=digest(run_dir / "export_check.json"),
                            export=manifest["export"], selected=manifest["selected"],
                            elapsed_seconds=manifest["elapsed_seconds"],
                            peak_cuda_allocated=manifest["peak_cuda_allocated"]))
        print(json.dumps({"replay_audited": job["pair"], "original_rank": job["original_rank"]}), flush=True)
    pairs = {}
    for pair in sorted(PAIRS_REQUIRED):
        a, b = sorted((r for r in results if r["pair"] == pair), key=lambda r: r["original_rank"])
        pairs[pair] = compare_payloads(a["export"], b["export"])
    result = dict(plan_sha256=digest(plan_path), runs=results, pairs=pairs,
                  quality_claim=False, timing_controlled=False)
    save_new(result_path, result)
    print(json.dumps({"equivalent_pairs": len(pairs), "result": str(result_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--evidence", type=Path, required=True)
    prep.add_argument("--study-root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--comfy", type=Path, default=Path("/media/p5/Comfyui"))
    execute.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare(args.evidence, args.study_root, args.out)
        print(json.dumps({"replays": len(plan["jobs"]), "plan": str(args.out / "plan.json")}))
    else:
        run(args.plan, args.comfy, args.python)


if __name__ == "__main__":
    main()
