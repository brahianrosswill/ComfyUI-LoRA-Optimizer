"""Isolated, paired before/after H3 search measurements; never installs nodes.

Fresh processes, identical source tensors/settings, explicit CUDA synchronization,
one warm-up per version/pair and three alternating-order measured repetitions.
Disconnected controls are separate. No render, remote API, learned judge, cache
release, automatic retry or overwrite. Worker RSS is bounded by a parent watchdog.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
import types

try:
    from .h3_benchmark import digest, save_new
    from .h3_merge_study import PAIRS, header, check_gpu_headroom
    from .h3_candidate_equivalence import require_idle
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_merge_study import PAIRS, header, check_gpu_headroom
    from h3_candidate_equivalence import require_idle

REPO = Path(__file__).resolve().parents[1]
# Privacy-safe published commit; identical tree to historical 27bedd7621e8e53b5d9c4d87ba594c2b7aa76008.
BASELINE = "cb92bd4f6e297fc16fe41889851801dcff792767"
BASELINE_SHA = "3d245ea08cda5aa4faf49cec9a878a93a521cb2bc71088d989f30c1d63346145"
PAIR_ORDER = ("series30_combat", "sully_cinema", "series30_cinema", "sully_combat")
SNAPSHOT_FILES = ("lora_optimizer.py", "experimental_merge.py", "kernel.py")
HELPERS = ("scripts/h3_search_benchmark.py", "scripts/h3_merge_study.py",
           "scripts/h3_candidate_equivalence.py", "scripts/h3_benchmark.py", "scripts/h3_render_study.py")
ENV = dict(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
           LORA_OPTIMIZER_PROFILE_MERGE="0", PYTHONHASHSEED="0")
SETTINGS = dict(top_n=3, normalize_keys="enabled", scoring_svd="disabled",
                scoring_device="gpu", scoring_speed="full", memory_mode="disabled",
                cache_patches="disabled", community_cache="disabled", diff_cache_mode="disabled",
                record_dataset="disabled", output_mode="merge", vram_budget=.35)


def schedule():
    jobs = []
    for index, pair in enumerate(PAIR_ORDER):
        for phase, repeats in (("disconnected", 1), ("warmup", 1), ("measured", 3)):
            for repeat in range(repeats):
                versions = ("baseline", "current") if (index + repeat) % 2 == 0 else ("current", "baseline")
                for version in versions:
                    jobs.append(dict(id=f"{pair}-{phase}-{repeat}-{version}", pair=pair,
                                     phase=phase, repeat=repeat, version=version))
    return jobs


def prepare(out, comfy, loras):
    if out.exists():
        raise FileExistsError(out)
    baseline = subprocess.check_output(["git", "show", f"{BASELINE}:lora_optimizer.py"], cwd=REPO)
    if hashlib.sha256(baseline).hexdigest() != BASELINE_SHA:
        raise ValueError("Unexpected committed baseline")
    for name in ("experimental_merge.py", "kernel.py"):
        old = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=REPO)
        if old != (REPO / name).read_bytes():
            raise ValueError("Baseline and current must use identical formula/kernel sources")
    checkpoint = comfy / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    shapes = header(checkpoint)
    adapters = {}
    for pair in PAIR_ORDER:
        entries = []
        for index, name in enumerate(PAIRS[pair]):
            path = loras / (name + ".safetensors")
            sidecar = json.loads(path.with_suffix(".metadata.json").read_text())
            sha = digest(path)
            if sha != sidecar["sha256"]:
                raise ValueError("Adapter differs from recorded source identity")
            entries.append(dict(path=str(path.resolve()), name=name + ".safetensors", sha256=sha,
                                strength=1. if index == 0 and pair.startswith("sully_") else .8))
        adapters[pair] = entries
    out.mkdir(parents=True)
    snapshots = {}
    for version in ("baseline", "current"):
        directory = out / "sources" / version
        directory.mkdir(parents=True)
        files = {}
        for name in SNAPSHOT_FILES:
            payload = baseline if version == "baseline" and name == "lora_optimizer.py" else (REPO / name).read_bytes()
            # Generated, isolated source artifacts, not a worktree checkout/edit.
            with (directory / name).open("xb") as stream:
                stream.write(payload)
            files[name] = digest(directory / name)
        snapshots[version] = dict(directory=str(directory.resolve()), files=files)
    plan = dict(version=1, created_at=time.time(), baseline_commit=BASELINE,
                root=str(out.resolve()), comfy=str(comfy.resolve()), loras=str(loras.resolve()),
                queue_base="http://127.0.0.1:8189", snapshots=snapshots, adapters=adapters,
                checkpoint=str(checkpoint.resolve()), checkpoint_size=checkpoint.stat().st_size,
                checkpoint_header_sha256=hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
                settings=SETTINGS, environment=ENV, merge_seed=2026090800, jobs=schedule(),
                helper_sha256={name: digest(REPO / name) for name in HELPERS},
                protocol=dict(warmups_per_pair_version=1, measured_repetitions=3,
                    experimental_trial_budget=dict(stable=3, additive_controls=1, np_lora=1, ct_merge=1),
                    timer="CUDA synchronize; perf_counter; actual auto_tune; CUDA synchronize",
                    timer_excludes="imports, source hashing/loading, context warm-up, winner hashing, serialization",
                    persistent_cache="new isolated models/user/temp directories per process; lookup/upload disabled",
                    winner_hash="raw stored model/CLIP patches, not a perceptual or base-model forward score",
                    concurrent_study_work="none during timed repetitions; no ffmpeg, tests or other GPU study",
                    limitations="GPU clocks are observed, not locked; system-wide CPU/I/O and queue activity are not continuously isolated. "
                                "Changing candidates changes useful work, so this is not a same-candidate microbenchmark."))
    save_new(out / "plan.json", plan)
    return plan


def validate_plan(path):
    plan = json.loads(path.read_text())
    if (plan.get("version") != 1 or plan.get("baseline_commit") != BASELINE
            or plan.get("root") != str(path.parent.resolve())
            or plan.get("queue_base") != "http://127.0.0.1:8189"
            or plan.get("jobs") != schedule() or plan.get("settings") != SETTINGS
            or plan.get("environment") != ENV or plan.get("merge_seed") != 2026090800
            or set(plan["snapshots"]) != {"baseline", "current"}
            or set(plan["helper_sha256"]) != set(HELPERS)
            or set(plan["adapters"]) != set(PAIR_ORDER)):
        raise ValueError("Benchmark scope/settings changed")
    for name, sha in plan["helper_sha256"].items():
        if digest(REPO / name) != sha:
            raise ValueError("Benchmark helper changed")
    for version, snapshot in plan["snapshots"].items():
        if (Path(snapshot["directory"]) != path.parent.resolve() / "sources" / version
                or set(snapshot["files"]) != set(SNAPSHOT_FILES)):
            raise ValueError("Snapshot escaped scope")
        for name, sha in snapshot["files"].items():
            if digest(Path(snapshot["directory"]) / name) != sha:
                raise ValueError("Snapshot changed")
    if plan["snapshots"]["baseline"]["files"]["lora_optimizer.py"] != BASELINE_SHA:
        raise ValueError("Baseline identity changed")
    for name in ("experimental_merge.py", "kernel.py"):
        if plan["snapshots"]["baseline"]["files"][name] != plan["snapshots"]["current"]["files"][name]:
            raise ValueError("Formula/kernel difference confounds the comparison")
    checkpoint = Path(plan["checkpoint"])
    if (checkpoint.stat().st_size != plan["checkpoint_size"] or
            hashlib.sha256(json.dumps(header(checkpoint), sort_keys=True).encode()).hexdigest() != plan["checkpoint_header_sha256"]):
        raise ValueError("Shape-only base changed")
    for pair, entries in plan["adapters"].items():
        expected = [dict(name=name + ".safetensors", strength=1. if i == 0 and pair.startswith("sully_") else .8)
                    for i, name in enumerate(PAIRS[pair])]
        if [{k: e[k] for k in ("name", "strength")} for e in entries] != expected:
            raise ValueError("Source roles/strengths changed")
    return plan


def hash_payload(value, *, chunk_elements=2 ** 20):
    """Deterministic typed digest including alpha, key offsets and signed zero."""
    import torch
    if chunk_elements <= 0:
        raise ValueError("Positive hash chunk size required")
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(json.dumps(["tensor", str(item.dtype), list(item.shape)]).encode())
            flat = item.detach().reshape(-1)
            for start in range(0, flat.numel(), chunk_elements):
                chunk = flat[start:start + chunk_elements].to("cpu").contiguous()
                if not torch.isfinite(chunk).all().item():
                    raise ValueError("Non-finite winner payload")
                result.update(chunk.view(torch.uint8).numpy().tobytes())
        elif hasattr(item, "weights"):
            result.update(type(item).__name__.encode())
            visit(item.weights)
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
            result.update(b"end")
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for part in item:
                visit(part)
            result.update(b"end")
        elif item is None or isinstance(item, (str, int, float, bool)):
            result.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise ValueError(f"Unsupported winner payload: {type(item).__name__}")
    visit(value)
    return result.hexdigest()


def validate_candidate_budget(tuner, experimental):
    rows = tuner["top_n"]
    stable = [row for row in rows if "experimental" not in row["config"]
              and not row["config"].get("experimental_baseline")]
    controls = [row for row in rows if row["config"].get("experimental_baseline")]
    methods = [row for row in rows if "experimental" in row["config"]]
    if (tuner["analysis_summary"]["prefix_count"] != 312 or len(stable) != 3
            or any(not math.isfinite(row["score_final"]) for row in rows)):
        raise ValueError("Incomplete, non-finite or changed stable candidate budget")
    if experimental is None:
        if len(rows) != 3 or controls or methods:
            raise ValueError("Disconnected path contains experimental trials")
    elif (len(rows) != 6 or len(controls) != 1 or len(methods) != 2
          or controls[0]["config"].get("experimental_baseline") is not True
          or controls[0]["config"].get("merge_mode") != "weighted_sum"
          or controls[0]["config"].get("optimization_mode") != "global"
          or {row["config"]["merge_mode"] for row in methods} != {"np_lora", "ct_merge"}):
        raise ValueError("Experimental budget must include additive control, NP and CT")
    return dict(stable=len(stable), additive_controls=len(controls), experimental_methods=len(methods), total=len(rows))


def worker(plan_path, job_id):
    plan = validate_plan(plan_path)
    job = next(job for job in plan["jobs"] if job["id"] == job_id)
    out = plan_path.parent / "runs" / job_id
    out.mkdir(parents=True, exist_ok=False)
    require_idle(plan["queue_base"])
    for item in plan["adapters"][job["pair"]]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Actual adapter payload changed")
    sys.path.insert(0, plan["comfy"])
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import comfy.lora
    from comfy.model_patcher import ModelPatcher
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized GPU environment")
    check_gpu_headroom("tune", torch.cuda.mem_get_info()[0])
    folder_paths.set_user_directory(str(out / "user"))
    folder_paths.set_temp_directory(str(out / "temp"))
    folder_paths.models_dir = str(out / "models")
    folder_paths.add_model_folder_path("loras", plan["loras"])
    source = Path(plan["snapshots"][job["version"]]["directory"]) / "lora_optimizer.py"
    spec = importlib.util.spec_from_file_location("h3_search_optimizer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._LoRAMergeBase._get_compute_device = staticmethod(lambda: torch.device("cuda"))
    import logging
    logging.basicConfig(level=logging.INFO, force=True, handlers=[logging.FileHandler(out / "merge.log")])

    class ShapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_config = types.SimpleNamespace(unet_config={})
            self.lora_optimizer_h3_profile = dict(partition="fl2va", basis="pruned")
            for key, info in header(plan["checkpoint"]).items():
                if key == "__metadata__" or not key.endswith((".weight", ".bias")):
                    continue
                if info["dtype"] not in ("I8", "F16", "BF16", "F32"):
                    raise ValueError("Unsupported shape-only storage")
                parent = self
                parts = ("diffusion_model." + key).split(".")
                for part in parts[:-1]:
                    if part not in parent._modules:
                        parent.add_module(part, torch.nn.Module())
                    parent = parent._modules[part]
                parent.register_parameter(parts[-1], torch.nn.Parameter(
                    torch.empty(info["shape"], device="meta", dtype=torch.bfloat16), requires_grad=False))

    model = ModelPatcher(ShapeModel(), torch.device("cpu"), torch.device("cpu"))
    stack, native = [], []
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    for item in plan["adapters"][job["pair"]]:
        sd = load_file(item["path"])
        if any("dora" in key.lower() or "adaln" in key.lower() for key in sd):
            raise ValueError("Base-dependent source out of scope")
        mapped = comfy.lora.load_lora(sd, mapping)
        if set(sd) != set().union(*(p.loaded_keys for p in mapped.values())):
            raise ValueError("Native source coverage incomplete")
        native.append(set(mapped))
        stack.append(dict(name=item["name"], lora=sd, strength=item["strength"],
                          metadata=header(item["path"]).get("__metadata__", {}), h3_layout="comfy"))
    if len(set.union(*native)) != 208:
        raise ValueError("Unexpected native target union")
    del native, mapped, sd
    # Equal lightweight CUDA initialization in every fresh process. No data from
    # another trial or prior result enters the measured search.
    init = torch.ones((256, 256), device="cuda")
    init = init @ init
    torch.cuda.synchronize()
    del init
    torch.cuda.empty_cache()
    torch.manual_seed(plan["merge_seed"])
    random.seed(plan["merge_seed"])
    node = module.LoRAAutoTuner()
    experimental = None if job["phase"] == "disconnected" else dict(np_lora=True, ct_merge=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = node.auto_tune(model, stack, 1., **plan["settings"], experimental_options=experimental)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    tuner, data = output[4], output[5]
    if data is None or len(data["model_patches"]) != 208 or data["clip_patches"]:
        raise ValueError("Incomplete native winner output")
    counts = validate_candidate_budget(tuner, experimental)
    # Outside the measured interval: hash every raw winner patch, not sparse
    # targets. This can detect changed output even when selected config agrees.
    payload = hash_payload(dict(model=data["model_patches"], clip=data["clip_patches"]))
    save_new(out / "tuner_data.json", tuner)
    record = dict(job=job, plan_sha256=digest(plan_path), elapsed_seconds=elapsed,
                  peak_cuda_allocated=peak, peak_cuda_reserved=peak_reserved,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  winner_payload_sha256=payload, winner_native_targets=208,
                  candidate_counts=counts,
                  tuner_sha256=digest(out / "tuner_data.json"), log_sha256=digest(out / "merge.log"),
                  torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(0),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  matmul_precision=torch.get_float32_matmul_precision(),
                  svd_kernel_loaded=module._HAS_SVD_KERNEL, triton_available=module._HAS_TRITON,
                  triton_disabled=module._DISABLE_TRITON,
                  actual_experimental_options=tuner.get("experimental_options"),
                  snapshots=plan["snapshots"][job["version"]]["files"])
    save_new(out / "result.json", record)
    del output, model, node, stack, data


def measured_summary(records):
    if [r["job"] for r in records] != schedule():
        raise ValueError("All predeclared trials are required, including controls/warm-ups")
    pairs = {}
    for pair in PAIR_ORDER:
        stats = {}
        for version in ("baseline", "current"):
            selected = [r for r in records if r["job"]["pair"] == pair and r["job"]["version"] == version
                        and r["job"]["phase"] == "measured"]
            times = [r["elapsed_seconds"] for r in selected]
            if len(times) != 3 or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Invalid or incomplete synchronized timing")
            stats[version] = dict(seconds=times, median_seconds=statistics.median(times),
                                  peak_cuda_bytes=[r["peak_cuda_allocated"] for r in selected],
                                  winner_payload_sha256=[r["winner_payload_sha256"] for r in selected])
        stats["current_over_baseline_median_time"] = stats["current"]["median_seconds"] / stats["baseline"]["median_seconds"]
        pairs[pair] = stats
    return pairs


def candidate_identity(row):
    return json.dumps(row["config"], sort_keys=True, allow_nan=False)


def compare_searches(records, root):
    """Keep default regressions, changed admission, scores and payloads separate."""
    tuners = {}
    for record in records:
        path = root / "runs" / record["job"]["id"] / "tuner_data.json"
        if digest(path) != record["tuner_sha256"]:
            raise ValueError("Tuner record changed")
        tuners[record["job"]["id"]] = json.loads(path.read_text())
    comparisons = {}
    for pair in PAIR_ORDER:
        controls = {r["job"]["version"]: r for r in records if r["job"]["pair"] == pair
                    and r["job"]["phase"] == "disconnected"}
        a, b = [tuners[controls[v]["job"]["id"]] for v in ("baseline", "current")]
        def rows_equal(a, b):
            return (len(a) == len(b) and all(candidate_identity(x) == candidate_identity(y)
                    and x.get("per_prefix_decisions") == y.get("per_prefix_decisions")
                    and math.isclose(x["score_final"], y["score_final"], rel_tol=1e-9, abs_tol=1e-9)
                    for x, y in zip(a, b)))
        control_equal = (rows_equal(a["top_n"], b["top_n"])
                         and controls["baseline"]["winner_payload_sha256"] == controls["current"]["winner_payload_sha256"])
        runs = []
        for repeat in range(3):
            selected = {r["job"]["version"]: r for r in records if r["job"]["pair"] == pair
                        and r["job"]["phase"] == "measured" and r["job"]["repeat"] == repeat}
            baseline, current = [tuners[selected[v]["job"]["id"]] for v in ("baseline", "current")]
            old, new = [{candidate_identity(row): row for row in t["top_n"]} for t in (baseline, current)]
            common = sorted(old.keys() & new.keys())
            runs.append(dict(repeat=repeat, admitted=[new[k]["config"] for k in sorted(new.keys()-old.keys())],
                             removed=[old[k]["config"] for k in sorted(old.keys()-new.keys())],
                             baseline_winner=baseline["top_n"][0]["config"], current_winner=current["top_n"][0]["config"],
                             winner_payload_equal=selected["baseline"]["winner_payload_sha256"] == selected["current"]["winner_payload_sha256"],
                             common_candidate_score_changes=[dict(config=old[k]["config"],
                                 current_minus_baseline=new[k]["score_final"]-old[k]["score_final"],
                                 per_prefix_decisions_equal=new[k].get("per_prefix_decisions") == old[k].get("per_prefix_decisions")) for k in common],
                             current_dedup=current.get("experimental_dedup")))
        comparisons[pair] = dict(disconnected_configs_scores_and_payload_equal=control_equal, measured=runs)
    return comparisons


def gpu_telemetry():
    fields = "index,name,driver_version,memory.free,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw"
    return dict(fields=fields.split(","), csv=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], text=True).strip())


def run(plan_path):
    import psutil
    plan = validate_plan(plan_path)
    root = plan_path.parent
    if (root / "runs").exists() or (root / "summary.json").exists():
        raise FileExistsError("Existing/partial execution; inspect it instead of restarting")
    (root / "runs").mkdir()
    records = []
    env = {**os.environ, **plan["environment"]}
    for job in plan["jobs"]:
        require_idle(plan["queue_base"])
        # A conservative guard for our owned worker, never unrelated processes.
        rss_limit = min(26 * 1024 ** 3, int(psutil.virtual_memory().available * .70))
        if rss_limit < 12 * 1024 ** 3:
            raise RuntimeError("Insufficient free RAM for a controlled H3 sweep")
        before = dict(gpu=gpu_telemetry(), available_ram=psutil.virtual_memory().available,
                      time=time.time())
        print(json.dumps({"starting": job, "rss_limit_bytes": rss_limit}), flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan", str(plan_path.resolve()), "--job", job["id"]]
        with (root / "runs" / (job["id"] + ".log")).open("x") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            memory_exceeded = False
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    if rss > rss_limit:
                        memory_exceeded = True
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.5)
            returncode = child.wait()
        if returncode or memory_exceeded:
            save_new(root / "stopped.json", dict(job=job, returncode=returncode, rss_limit_bytes=rss_limit,
                                                memory_exceeded=memory_exceeded, automatic_retry=False))
            raise RuntimeError(f"Owned benchmark worker failed: {job['id']}; inspect its log, no retry")
        require_idle(plan["queue_base"])
        result_path = root / "runs" / job["id"] / "result.json"
        result = json.loads(result_path.read_text())
        if result["plan_sha256"] != digest(plan_path) or result["job"] != job:
            raise ValueError("Worker receipt does not match the scheduled trial")
        telemetry = dict(before=before, after=dict(gpu=gpu_telemetry(), available_ram=psutil.virtual_memory().available,
                                                  time=time.time()), worker_rss_limit_bytes=rss_limit)
        save_new(result_path.parent / "telemetry.json", telemetry)
        result["telemetry_sha256"] = digest(result_path.parent / "telemetry.json")
        records.append(result)
        print(json.dumps({"completed": job["id"], "seconds": result["elapsed_seconds"],
                          "peak_cuda_bytes": result["peak_cuda_allocated"]}), flush=True)
    summary = dict(plan_sha256=digest(plan_path), records=records, pairs=measured_summary(records),
                   comparisons=compare_searches(records, root),
                   quality_claim=False, note=plan["protocol"]["limitations"])
    save_new(root / "summary.json", summary)
    print(json.dumps({"completed_trials": len(records), "summary": str(root / "summary.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--comfy", type=Path, default=Path("/media/p5/Comfyui"))
    prep.add_argument("--loras", type=Path, default=Path("/media/p5/ComfyUI-Model-CIFS-Cache/loras/MiniMax H3"))
    execute = sub.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    work = sub.add_parser("worker")
    work.add_argument("--plan", type=Path, required=True)
    work.add_argument("--job", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare(args.out, args.comfy, args.loras)
        print(json.dumps({"trials": len(plan["jobs"]), "plan": str(args.out / "plan.json")}))
    elif args.action == "run":
        run(args.plan)
    else:
        worker(args.plan, args.job)


if __name__ == "__main__":
    main()
