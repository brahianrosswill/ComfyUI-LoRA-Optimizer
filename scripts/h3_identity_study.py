"""Frozen character-only qualification, separate from the unchanged AV2 benchmark.

Prepare is offline. Run uses only the existing local ComfyUI, one job at a time,
and stops for unrelated work. No restart, cancellation, download or graph edits.
These are calibration controls, not held-out merge scores or human preferences.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

try:
    from .h3_benchmark import digest, save_new
    from .h3_render_study import graph, request
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_render_study import graph, request

BASE = "http://127.0.0.1:8189"
SEEDS = (2026090821, 2026090822)
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".h3-study-artifacts/20260908"
PROBES = {
    "sully": "integrated_multimodal_description: [Shot 1] 3D CG, a medium close-up of h3sully, an adult male monster with thick green and purple fur, two short curved horns, a broad nose and expressive eyes. He stands alone in a plain softly lit room. His full head and both shoulders remain visible. He begins facing the camera, slowly turns his head toward his left into a three-quarter view, then faces the camera again and gives a small closed-mouth smile. His body remains still. One continuous static shot at normal speed, soft side light revealing individual fur strands. No speech, no text or subtitles.\n\noverall_soundscape: Low steady room ambience and faint fur movement. No voices or impacts.\n\nnon_diegetic_music: N/A",
    "series30": "integrated_multimodal_description: [Shot 1] Live-action, a medium close-up of ai30, a fictional adult Japanese woman in her thirties with long straight black hair, wearing an opaque white collared blouse. She stands alone in a plain softly lit room. Her full head and both shoulders remain visible. She begins facing the camera, slowly turns her head toward her left into a three-quarter view, then faces the camera again and gives a small closed-mouth smile. Her body remains still. One continuous static shot at normal speed, soft side light revealing natural skin and hair texture. No speech, no text or subtitles.\n\noverall_soundscape: Low steady room ambience and faint clothing rustle. No voices or impacts.\n\nnon_diegetic_music: N/A",
}


def matrix(adapters, prompt_text):
    jobs = []
    for name, adapter in adapters.items():
        for prompt in (f"{name}_source", f"{name}_turn"):
            if not prompt_text.get(prompt):
                raise ValueError("Missing full source/probe prompt")
            for seed in SEEDS:
                for variant in ("base", "character"):
                    jobs.append(dict(id=f"idq-{name}-{prompt.rsplit('_', 1)[1]}-{seed}-{variant}",
                        stage="calibration", character=name, prompt=prompt, seed=seed, variant=variant,
                        loras=[] if variant == "base" else [[adapter["api_name"], adapter["strength"]]]))
    random.Random(2026090820).shuffle(jobs)
    jobs.sort(key=lambda job: (job["seed"], job["prompt"].endswith("_turn")))
    return jobs


def expected_graph(plan, job):
    return graph(plan["prompt_text"][job["prompt"]], job["seed"], job["loras"],
                 f"h3_identity_study_20260908/{job['id']}", profile="local")


def validate_plan(plan, verify_files=True):
    if (plan["base"], plan["profile"], plan["width"], plan["height"], plan["length"], plan["steps"]) != (
            BASE, "local", 640, 384, 124, 20):
        raise ValueError("Plan differs from authorized local INT8/FP4 profile")
    if plan["jobs"] != matrix(plan["adapters"], plan["prompt_text"]):
        raise ValueError("Qualification matrix changed")
    if len(plan["jobs"]) != 16 or set(plan["adapters"]) != {"sully", "series30"}:
        raise ValueError("Qualification scope changed")
    for filename, expected in plan["source_sha256"].items():
        if Path(filename).name != filename or digest(Path(__file__).parent / filename) != expected:
            raise ValueError("Harness changed after plan freeze")
    if verify_files:
        for adapter in plan["adapters"].values():
            if digest(adapter["local_path"]) != adapter["sha256"]:
                raise ValueError("Installed adapter differs from frozen identity")


def prepare(plan_path, root, sources):
    if plan_path.exists():
        raise FileExistsError(plan_path)
    adapters, prompts = {}, {}
    for name, strength in (("sully", 1.), ("series30", .8)):
        source = json.loads((sources / f"{name}-source.json").read_text())
        prompt_path = sources / f"{name}-reference-prompts.json"
        fields = json.loads(prompt_path.read_text())
        if len(fields) != 1 or not isinstance(fields[0]["text"], str):
            raise ValueError("Resolve the complete original prompt explicitly")
        local_path = Path(source["local_path"])
        adapter_root = Path("/media/p5/ComfyUI-Model-CIFS-Cache/loras")
        api_name = local_path.relative_to(adapter_root).as_posix()
        if digest(local_path) != source["file"]["hashes"]["SHA256"].lower():
            raise ValueError("Adapter hash differs from Civitai source")
        adapters[name] = dict(local_path=str(local_path), api_name=api_name, strength=strength,
            sha256=digest(local_path), version_id=source["version_id"], source_url=source["source_url"],
            source_prompt_sha256=digest(prompt_path), source_node=fields[0]["node_id"])
        prompts[f"{name}_source"] = fields[0]["text"]
        prompts[f"{name}_turn"] = PROBES[name]
    jobs = matrix(adapters, prompts)
    plan = dict(version=1, created_at=time.time(), base=BASE, root=str(root),
        profile="local", width=640, height=384, length=124, steps=20,
        adapters=adapters, prompt_text=prompts, jobs=jobs,
        cases=[dict(pair=j["character"], variant=j["variant"], stage=j["stage"],
                    prompt=j["prompt"], seed=j["seed"], job_id=j["id"]) for j in jobs],
        source_sha256={f: digest(Path(__file__).parent / f) for f in (
            "h3_identity_study.py", "h3_render_study.py", "h3_benchmark.py")},
        protocol={"purpose": "Character-only qualification; not held-out merge evaluation",
            "source_prompts": "Exact saved originals, including trigger quirks; no exact creator reproduction claim",
            "source_reference_limits": "Series30 uses Turbo and post-processing; reference pixels/timing are not ground truth",
            "comparisons": "Prompt and seed identical within base/character pairs; no frame or LoRA strength selection after results",
            "qualification": "Inspect identity against source, reproducibility across seeds, within-clip identity and pose adherence; report failures separately",
            "ranking": "No automatic scalar quality score, preference labels or fitted ranking weights",
            "holdout": "These two characters/prompts/seeds are calibration. Later tests need separately frozen unused prompts/seeds; neither character is unseen after this qualification."})
    validate_plan(plan)
    save_new(plan_path, plan)
    print(json.dumps(dict(plan=str(plan_path), sha256=digest(plan_path), jobs=len(jobs))))


def prepare_job(plan, job):
    run = Path(plan["root"]) / job["id"]
    if not run.exists():
        run.mkdir(parents=True)
        save_new(run / "manifest.json", dict(base=BASE, client_id="h3-idq-" + uuid.uuid4().hex,
            prompt_name=job["prompt"], seed=job["seed"], loras=job["loras"],
            output_prefix=f"h3_identity_study_20260908/{job['id']}",
            turbo=False, steps=20, profile="local", width=640, height=384, length=124,
            quality_claim=None, character=job["character"], variant=job["variant"]))
        save_new(run / "prompt_api.json", expected_graph(plan, job))
        with (run / "prompt.txt").open("x") as stream:
            stream.write(plan["prompt_text"][job["prompt"]])
    if json.loads((run / "prompt_api.json").read_text()) != expected_graph(plan, job):
        raise ValueError("Prepared graph differs from frozen case")
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest["base"], manifest["seed"], manifest["loras"]) != (BASE, job["seed"], job["loras"]):
        raise ValueError("Prepared manifest differs from frozen case")
    return run, manifest


def completed(run, expected):
    if not (run / "history.json").exists():
        return False
    history = json.loads((run / "history.json").read_text())
    if history.get("status", {}).get("status_str") != "success":
        raise RuntimeError("Recorded execution failed; inspect before any rerun")
    if history["prompt"][2] != expected:
        raise ValueError("Executed graph differs from frozen case")
    return True


def run_batch(plan, limit):
    if not 1 <= limit <= 16:
        raise ValueError("Batch limit must be 1..16")
    validate_plan(plan)
    runner = Path(__file__).with_name("h3_render_study.py")
    count = 0
    for job in plan["jobs"]:
        run = Path(plan["root"]) / job["id"]
        expected = expected_graph(plan, job)
        if completed(run, expected):
            continue
        run, manifest = prepare_job(plan, job)
        if not (run / "submission.json").exists():
            if (run / "submission-intent.json").exists():
                raise RuntimeError(f"Ambiguous prior submission at {run}; reconcile the recorded client/graph with live queue/history, never resubmit blindly")
            queue = request(BASE, "/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                print(json.dumps(dict(paused="Existing queued work; nothing submitted", next_job=job["id"])), flush=True)
                return
            save_new(run / "queue_before.json", {k: [j[1] for j in v] for k, v in queue.items()})
            save_new(run / "system_stats.json", request(BASE, "/system_stats"))
            submitted_at = time.time()
            # Persist BEFORE POST: a network timeout must not cause duplication.
            save_new(run / "submission-intent.json", dict(client_id=manifest["client_id"],
                graph_sha256=digest(run / "prompt_api.json"), submitted_at=submitted_at))
            result = request(BASE, "/prompt", dict(prompt=expected, client_id=manifest["client_id"]))
            save_new(run / "submission.json", dict(**result, submitted_at=submitted_at))
        print(json.dumps(dict(watching=job["id"], prompt_id=json.loads((run / "submission.json").read_text())["prompt_id"])), flush=True)
        with (run / "watch.log").open("a") as log:
            subprocess.run([sys.executable, str(runner), "watch", "--out", str(run)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if not completed(run, expected):
            raise RuntimeError("No terminal history; inspect the same prompt handle")
        count += 1
        print(json.dumps(dict(completed=job["id"], batch_completed=count)), flush=True)
        if count >= limit:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ARTIFACTS)
    parser.add_argument("--sources", type=Path, default=ARTIFACTS / "identity-acquisition")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.plan, args.root, args.sources)
    else:
        run_batch(json.loads(args.plan.read_text()), args.limit)


if __name__ == "__main__":
    main()
