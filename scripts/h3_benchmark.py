"""Frozen, bounded local H3 benchmark. No server start, restart or cancellation.

prepare is offline. install copies only the plan's verified exports into the
existing study LoRA directory, exclusively and without overwriting. run submits
one job at a time, stops for unrelated queued work, and resumes recorded jobs
without resubmission. Human AV preferences are a separate, required input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

try:
    from .h3_render_study import PROMPTS, graph, request
except ImportError:
    from h3_render_study import PROMPTS, graph, request

STUDY_ROOT = Path("/tmp/h3-autotuner-study-20260908")
INSTALL_ROOT = Path("/media/p5/model_temps/h3-autotuner-study-20260908")
COMBAT = "MiniMax H3/concept/H3_Combat_V2.safetensors"
CINEMA = "MiniMax H3/style/Cinema-MH3-V02_000010000.safetensors"
REPAIR = "MiniMax H3/concept/Motion_Repair.safetensors"
EXPORT_RUNS = {
    "combat_cinema": {"additive": "combat-cinema-additive-01", "winner": "combat-cinema-winner-03",
                      "np": "combat-cinema-np-01-native", "ct": "combat-cinema-ct-02-native"},
    "combat_repair": {"additive": "combat-repair-additive-01", "winner": "combat-repair-winner-02",
                      "np": "combat-repair-np-01-native", "ct": "combat-repair-ct-01-native"},
}
BLOCKS = (("calibration", "boxing", 2026090803), ("calibration", "boxing", 2026090804),
          ("heldout", "padwork", 2026090811), ("heldout", "padwork", 2026090812))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def case_matrix(exports):
    """Deduplicate shared base/Combat controls, never across prompts or seeds."""
    jobs, cases, identities = [], [], {}
    for stage, prompt, seed in BLOCKS:
        for pair, second, strength in (("combat_cinema", CINEMA, .8), ("combat_repair", REPAIR, .6)):
            variants = {"base": [], "combat": [[COMBAT, .8]], "second": [[second, strength]]}
            variants.update({mode: [[exports[pair][mode]["api_name"], 1.]]
                             for mode in EXPORT_RUNS[pair]})
            for variant, loras in variants.items():
                identity = json.dumps([prompt, seed, loras], sort_keys=True)
                if identity not in identities:
                    job_id = f"av2-{prompt}-{seed}-{pair.replace('_', '-')}-{variant}"
                    identities[identity] = job_id
                    jobs.append(dict(id=job_id, stage=stage, prompt=prompt, seed=seed, loras=loras))
                cases.append(dict(pair=pair, variant=variant, stage=stage, prompt=prompt,
                                  seed=seed, job_id=identities[identity]))
    # Fixed order before rendering, independent of measured results.
    random.Random(2026090857).shuffle(jobs)
    jobs.sort(key=lambda job: (job["stage"] != "calibration", job["seed"]))
    return jobs, cases


def prepare(root, plan_path):
    if plan_path.exists():
        raise FileExistsError(plan_path)
    exports = {}
    for pair, runs in EXPORT_RUNS.items():
        exports[pair] = {}
        for mode, run in runs.items():
            manifest_path = root / run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            source = Path(manifest["export"])
            if source.stat().st_size != manifest["export_size"]:
                raise ValueError(f"Export size mismatch: {source}")
            exports[pair][mode] = dict(source=str(source), sha256=digest(source),
                size=source.stat().st_size, manifest_sha256=digest(manifest_path),
                api_name=f"{INSTALL_ROOT.name}/{run}.safetensors",
                destination=str(INSTALL_ROOT / f"{run}.safetensors"))
    jobs, cases = case_matrix(exports)
    plan = dict(version=1, created_at=time.time(), base="http://127.0.0.1:8189",
        root=str(root), profile="local", width=640, height=384, length=124, steps=20,
        source_sha256={name: digest(Path(__file__).parent / name)
                       for name in ("h3_benchmark.py", "h3_render_study.py")},
        prompt_text={k: PROMPTS[k] for k in ("boxing", "padwork")},
        exports=exports, jobs=jobs, cases=cases,
        protocol={"calibration": "boxing seeds 2026090803/04; preferences may inform candidate changes",
                  "heldout": "padwork seeds 2026090811/12; do not inspect before freezing the candidate policy",
                  "quality_source": "blinded full-video AND audio review, not descriptive metrics",
                  "np_roles": "Combat as content; capability pairs, NOT subject-identity validation",
                  "no_default_change_without_evidence": True})
    save_new(plan_path, plan)
    return dict(plan=str(plan_path), sha256=digest(plan_path), jobs=len(jobs), cases=len(cases))


def install(plan):
    for variants in plan["exports"].values():
        for item in variants.values():
            source, destination = Path(item["source"]), Path(item["destination"])
            if destination.parent != INSTALL_ROOT or not destination.name.endswith(".safetensors"):
                raise ValueError("Installation target outside the dedicated study directory")
            if digest(source) != item["sha256"]:
                raise ValueError(f"Source changed: {source}")
            if destination.exists():
                if digest(destination) != item["sha256"]:
                    raise ValueError(f"Existing destination differs; refusing overwrite: {destination}")
                continue
            INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if digest(destination) != item["sha256"]:
                raise ValueError(f"Copy verification failed: {destination}; original preserved")
            print(json.dumps(dict(installed=str(destination), sha256=item["sha256"])), flush=True)


def successful(run):
    path = run / "history.json"
    if not path.exists():
        return False
    history = json.loads(path.read_text())
    if history.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"Recorded job failed; inspect {path}, do not auto-resubmit")
    return True


def run_batch(plan, stage, limit):
    if not 1 <= limit <= 12:
        raise ValueError("Batch limit must be 1..12")
    runner = Path(__file__).with_name("h3_render_study.py")
    if digest(runner) != plan["source_sha256"][runner.name]:
        raise ValueError("Render harness changed after plan freeze; create a new plan")
    if (plan["base"], plan["profile"], plan["width"], plan["height"], plan["length"], plan["steps"]) != (
            "http://127.0.0.1:8189", "local", 640, 384, 124, 20):
        raise ValueError("Benchmark is restricted to the authorized local profile")
    for variants in plan["exports"].values():
        for item in variants.values():
            if digest(item["destination"]) != item["sha256"]:
                raise ValueError(f"Installed export differs: {item['destination']}")
    for prompt, text in plan["prompt_text"].items():
        if PROMPTS[prompt] != text:
            raise ValueError("Prompt differs from frozen plan")
    completed = 0
    for job in plan["jobs"]:
        if job["stage"] != stage:
            continue
        run = Path(plan["root"]) / job["id"]
        if successful(run):
            continue
        if not (run / "submission.json").exists():
            queue = request(plan["base"], "/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                print(json.dumps(dict(paused="Existing queued work; nothing submitted", next_job=job["id"])), flush=True)
                return
            if not run.exists():
                command = [sys.executable, str(runner), "prepare", "--profile", "local",
                           "--out", str(run), "--prompt", job["prompt"], "--seed", str(job["seed"])]
                for name, strength in job["loras"]:
                    command.extend(("--lora", name, str(strength)))
                subprocess.run(command, check=True)
            # Verify exact prepared identity before submitting an existing run.
            manifest = json.loads((run / "manifest.json").read_text())
            actual_loras = [[name, float(strength)] for name, strength in manifest["loras"]]
            if (manifest["seed"], manifest["prompt_name"], actual_loras, manifest["base"]) != (
                    job["seed"], job["prompt"], job["loras"], plan["base"]):
                raise ValueError(f"Prepared job differs: {run}")
            expected_graph = graph(PROMPTS[job["prompt"]], job["seed"], job["loras"],
                                   manifest["output_prefix"], profile="local")
            if json.loads((run / "prompt_api.json").read_text()) != expected_graph:
                raise ValueError(f"Prepared graph differs from frozen local configuration: {run}")
            subprocess.run([sys.executable, str(runner), "submit", "--out", str(run)], check=True)
        print(json.dumps(dict(watching=job["id"])), flush=True)
        # WebSocket monitor follows the recorded prompt only; never resubmit a
        # lost/slow response. Append logs so a resumed watch preserves evidence.
        with (run / "watch.log").open("a") as stream:
            subprocess.run([sys.executable, str(runner), "watch", "--out", str(run)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not successful(run):
            raise RuntimeError(f"No terminal history: {run}")
        completed += 1
        print(json.dumps(dict(completed=job["id"], batch_completed=completed)), flush=True)
        if completed >= limit:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "run"))
    parser.add_argument("--root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "heldout"), default="calibration")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root, args.plan)))
        return
    plan = json.loads(args.plan.read_text())
    if args.action == "install":
        install(plan)
    else:
        run_batch(plan, args.stage, args.limit)


if __name__ == "__main__":
    main()
