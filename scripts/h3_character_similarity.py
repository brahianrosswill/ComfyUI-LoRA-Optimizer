"""Apply the unchanged calibration probe to matched character merge controls.

CPU only. Raw cosine differences are diagnostics, not identity probabilities,
effect scores, audiovisual preference or autotuner feedback. This recipe is
restricted to the two declared calibration seeds; it never opens held-out media.
"""
import argparse
import json
import math
from pathlib import Path
import time

try:
    from . import h3_identity_probe as probe
    from .h3_character_benchmark import ARMS, PAIRS, expected_graph
    from .h3_identity_study import completed
    from .h3_av_review import output_video
    from .h3_benchmark import digest, save_new
except ImportError:
    import h3_identity_probe as probe
    from h3_character_benchmark import ARMS, PAIRS, expected_graph
    from h3_identity_study import completed
    from h3_av_review import output_video
    from h3_benchmark import digest, save_new

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/research/data"
SEEDS = (2026090831, 2026090832)
HELPERS = ("h3_character_similarity.py", "h3_identity_probe.py", "h3_character_benchmark.py",
           "h3_identity_study.py", "h3_render_study.py", "h3_av_review.py", "h3_benchmark.py")


def calibration_jobs(render_plan):
    jobs = [j for j in render_plan["jobs"] if j["stage"] == "calibration" and j["seed"] in SEEDS]
    if len(jobs) != 36 or len({j["id"] for j in jobs}) != 36:
        raise ValueError("Expected exactly 36 declared calibration jobs")
    for pair in PAIRS:
        for seed in SEEDS:
            chosen = [j for j in jobs if (j["pair"], j["seed"]) == (pair, seed)]
            if len(chosen) != 9 or {j["variant"] for j in chosen} != set(ARMS):
                raise ValueError("Incomplete matched nine-arm block")
    return jobs


def prepare(recipe_path):
    if recipe_path.exists():
        raise FileExistsError(recipe_path)
    render_path = DATA / "2026-09-08-h3-character-benchmark-plan.json"
    probe_path = DATA / "2026-09-08-h3-identity-probe-plan.json"
    render_plan = json.loads(render_path.read_text())
    old = json.loads(probe_path.read_text())
    if old["runner_sha256"] != digest(probe.__file__):
        raise ValueError("Original calibrated probe implementation changed")
    save_new(recipe_path, dict(version=1, created_at=time.time(),
        render_plan_path=str(render_path), render_plan_sha256=digest(render_path),
        probe_plan_path=str(probe_path), probe_plan_sha256=digest(probe_path),
        jobs=calibration_jobs(render_plan),
        inherited_probe={k: old[k] for k in ("model_path", "model_sha256", "config_sha256", "frames", "references", "processor", "limitations")},
        helper_sha256={name: digest(ROOT / "scripts" / name) for name in HELPERS},
        purpose="Apply the already calibrated head-region formula unchanged to both nine-arm calibration blocks; do not alter masks, frames or thresholds after outputs.",
        exposure="Seed-31 sparse visual observations were already recorded and unblinded. Seed-32 generations are in progress; no seed-32 visual observations have been made at recipe freeze.",
        reporting="Retain every sampled frame and all controls. Subtract the matched character-only and additive medians in raw cosine units, not normalized preservation percentages. Effects, visibility and AV quality remain separate.",
        heldout_allowed=False, evaluator_feedback=False, quality_labels=[]))
    print(json.dumps(dict(recipe=str(recipe_path), sha256=digest(recipe_path), jobs=36)))


def matched_deltas(rows):
    """No fitted thresholds, clipping, percentages or aggregate winner."""
    if not rows or any(type(r[field]) not in (int, float) or not math.isfinite(r[field])
            for r in rows for field in ("median_head_maxcos", "median_global_cosine")):
        raise ValueError("Finite numeric observations required")
    output = []
    groups = {(r["job"]["pair"], r["job"]["seed"]) for r in rows}
    for pair, seed in sorted(groups):
        group = [r for r in rows if (r["job"]["pair"], r["job"]["seed"]) == (pair, seed)]
        by_arm = {r["job"]["variant"]: r for r in group}
        if len(group) != 9 or set(by_arm) != set(ARMS):
            raise ValueError("Cannot compare an incomplete or duplicated nine-arm block")
        character, additive = by_arm["character_only"], by_arm["additive"]
        for arm in ARMS:
            row = by_arm[arm]
            value = row["median_head_maxcos"]
            output.append(dict(pair=pair, seed=seed, variant=arm, median_head_maxcos=value,
                delta_to_character_only=value-character["median_head_maxcos"],
                delta_to_additive=value-additive["median_head_maxcos"],
                median_global_cosine=row["median_global_cosine"]))
    return output


def evaluate(recipe_path, seed, out):
    if out.exists():
        raise FileExistsError(out)
    if seed not in SEEDS:
        raise ValueError("Only the two declared calibration seeds may be evaluated")
    recipe = json.loads(recipe_path.read_text())
    if recipe["heldout_allowed"] is not False or recipe["evaluator_feedback"] is not False:
        raise ValueError("Research-only calibration scope changed")
    for name, sha in recipe["helper_sha256"].items():
        if Path(name).name != name or digest(ROOT / "scripts" / name) != sha:
            raise ValueError("Similarity implementation changed after freeze")
    for kind in ("render", "probe"):
        if digest(recipe[f"{kind}_plan_path"]) != recipe[f"{kind}_plan_sha256"]:
            raise ValueError("Original frozen plan changed")
    render_plan = json.loads(Path(recipe["render_plan_path"]).read_text())
    old = json.loads(Path(recipe["probe_plan_path"]).read_text())
    spec = recipe["inherited_probe"]
    if (recipe["jobs"] != calibration_jobs(render_plan) or
            spec != {k: old[k] for k in spec} or spec["frames"] != list(probe.FRAMES) or spec["processor"] != probe.PROCESSOR):
        raise ValueError("Sampling, references or evaluation matrix changed")
    clips = []
    for job in [j for j in recipe["jobs"] if j["seed"] == seed]:
        run = Path(render_plan["root"]) / job["id"]
        if not completed(run, expected_graph(render_plan, job)):
            raise ValueError("Missing completed calibration case")
        video = output_video(json.loads((run / "history.json").read_text()))
        audit = json.loads((run / "media-audit/metrics.json").read_text())
        sha = digest(video)
        if sha != audit["video_sha256"] or audit["decoded_frames"] != 124:
            raise ValueError("Source output lacks a matching full-clip audit")
        clips.append(dict(job=job, path=str(video), sha256=sha))
    if len(clips) != 18:
        raise ValueError("Incomplete calibration seed")
    model_path = Path(spec["model_path"])
    if (digest(model_path / "model.safetensors") != spec["model_sha256"]
            or digest(model_path / "config.json") != spec["config_sha256"]):
        raise ValueError("Cached evaluator model changed")
    for ref in spec["references"].values():
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Original creator reference changed")
    import numpy as np
    import torch
    import transformers
    from transformers import SiglipVisionModel, SiglipImageProcessor
    torch.set_num_threads(4)
    model, loading = SiglipVisionModel.from_pretrained(model_path, local_files_only=True,
        dtype=torch.float32, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Incomplete vision checkpoint")
    model.eval().to("cpu")
    processor = SiglipImageProcessor(**spec["processor"])
    def features(frame):
        with torch.inference_mode():
            result = model(**processor(images=frame, return_tensors="pt"))
        return result.last_hidden_state[0].numpy(), result.pooler_output[0].numpy()
    references = {}
    for name, ref in spec["references"].items():
        patches, pooled = features(probe.decode(ref["path"], [ref["frame"]])[0])
        if patches.shape != (1024, 768):
            raise ValueError("Unexpected evaluator patch grid")
        references[name] = patches, pooled, probe.patch_mask(ref["normalized_head_rectangle"], 32)
    started, rows = time.monotonic(), []
    for clip in clips:
        job = clip["job"]
        ref, pooled_ref, mask = references[job["character"]]
        frames = []
        for index, frame in zip(probe.FRAMES, probe.decode(clip["path"], probe.FRAMES)):
            patches, pooled = features(frame)
            frames.append(dict(frame=index, time_seconds=index/24,
                reference_head_maxcos=probe.similarities(ref, patches, mask),
                global_cosine=float(np.dot(pooled_ref, pooled) / (np.linalg.norm(pooled_ref)*np.linalg.norm(pooled)))))
        if digest(clip["path"]) != clip["sha256"]:
            raise ValueError("Output media changed during scoring")
        rows.append(dict(job=job, video_sha256=clip["sha256"], frames=frames,
            median_head_maxcos=float(np.median([f["reference_head_maxcos"] for f in frames])),
            min_head_maxcos=min(f["reference_head_maxcos"] for f in frames),
            median_global_cosine=float(np.median([f["global_cosine"] for f in frames]))))
        print(json.dumps(dict(completed=job["id"])), flush=True)
    save_new(out, dict(recipe_sha256=digest(recipe_path), seed=seed, stage="calibration",
        model_sha256=spec["model_sha256"], results=rows, matched=matched_deltas(rows),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        device="cpu", threads=4, elapsed_scoring_seconds=time.monotonic()-started,
        source_frame_count=2, output_frame_count=len(rows)*len(probe.FRAMES),
        quality_labels=[], ranking_used=False, not_masc=True,
        visibility_judged=False, audio_listened=False, limitations=spec["limitations"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.recipe)
    elif args.seed is None or args.out is None:
        parser.error("evaluate requires --seed and --out")
    else:
        evaluate(args.recipe, args.seed, args.out)
