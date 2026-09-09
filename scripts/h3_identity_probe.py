"""Offline, calibration-only SigLIP2 head-region similarity diagnostic.

This is NOT MaSC: it uses the cached base/512 backbone and a manually declared
reference head rectangle, not SO400M-NaFlex and semantic segmentation. It does
not measure motion, audio, action success, visibility or human preference.
No scores feed the autotuner. All media and sampling are pinned before scoring.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

try:
    from .h3_benchmark import digest, save_new
    from .h3_av_review import output_video
    from .h3_identity_study import completed, expected_graph
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_av_review import output_video
    from h3_identity_study import completed, expected_graph

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".h3-study-artifacts/20260908"
MODEL = Path("/home/ethanfel/.cache/huggingface/hub/models--google--siglip2-base-patch16-512/snapshots/a89f5c5093f902bf39d3cd4d81d2c09867f0724b")
FRAMES = (0, 24, 48, 72, 96, 120)
ROIS = {"sully": [.52, .04, .86, .98], "series30": [.24, .01, .66, .97]}
PROCESSOR = dict(size=dict(height=512, width=512), resample=2, do_resize=True,
    do_rescale=True, rescale_factor=1/255, do_normalize=True,
    image_mean=[.5, .5, .5], image_std=[.5, .5, .5], do_convert_rgb=True)


def prepare(path):
    if path.exists():
        raise FileExistsError(path)
    source = ROOT / "docs/research/data/2026-09-08-h3-identity-qualification-plan.json"
    qualification = json.loads(source.read_text())
    refs = {name: dict(path=str(ARTIFACTS / f"identity-acquisition/{name}-reference.mp4"),
        frame=0, normalized_head_rectangle=roi) for name, roi in ROIS.items()}
    for ref in refs.values():
        ref["sha256"] = digest(ref["path"])
    clips = []
    for job in qualification["jobs"]:
        if job["stage"] != "calibration":
            raise ValueError("Qualification-only probe must not inspect held-out clips")
        run = Path(qualification["root"]) / job["id"]
        if not completed(run, expected_graph(qualification, job)):
            raise ValueError("Missing original qualification output")
        video = output_video(json.loads((run / "history.json").read_text()))
        clips.append(dict(job=job, path=str(video), sha256=digest(video)))
    save_new(path, dict(version=1, source_plan=str(source), source_plan_sha256=digest(source),
        model_path=str(MODEL), model_sha256=digest(MODEL / "model.safetensors"),
        config_sha256=digest(MODEL / "config.json"), runner_sha256=digest(__file__),
        frames=list(FRAMES), references=refs, clips=clips, processor=PROCESSOR,
        processor_source="https://huggingface.co/google/siglip2-base-patch16-512/blob/main/preprocessor_config.json",
        roi_policy="Single first creator frame; manually declared head rectangles from already exposed reference grids, before scoring. Approximate head regions, not segmentation; do not tune on scores.",
        scope="Six fixed frames from every old character-only calibration clip. No new merge outputs, no held-out frames, no preference labels or fitted thresholds.",
        limitations="Not MaSC, not face recognition, not an identity probability. Similarity may reflect species, hair, pose or lighting; visibility and effect must be separately reviewed. Do not discard low-scoring frames.",
        created_at=time.time()))
    print(json.dumps(dict(plan=str(path), sha256=digest(path), clips=len(clips))))


def patch_mask(rectangle, side):
    import numpy as np
    left, top, right, bottom = rectangle
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Invalid normalized rectangle")
    centers = (np.arange(side) + .5) / side
    x, y = np.meshgrid(centers, centers)
    mask = ((x >= left) & (x < right) & (y >= top) & (y < bottom)).reshape(-1)
    if not mask.any():
        raise ValueError("Reference head region contains no patches")
    return mask


def similarities(reference, output, mask):
    import numpy as np
    if (reference.ndim != 2 or reference.shape != output.shape or len(mask) != len(reference)
            or mask.dtype != np.bool_ or not mask.any()
            or not np.isfinite(reference).all() or not np.isfinite(output).all()):
        raise ValueError("Invalid patch features/mask")
    def normalized(value):
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        if (norm <= 0).any():
            raise ValueError("Zero-norm patch features")
        return value / norm
    values = (normalized(reference[mask]) @ normalized(output).T).max(axis=1)
    return float(values.mean())


def decode(path, indices):
    """Exact indexed source frames; never seek/select a flattering frame."""
    import numpy as np
    from PIL import Image
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(path)], check=True, capture_output=True).stdout)
    spec = probe["streams"][0]
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
        "-an", "-vf", "select=" + selection, "-fps_mode", "vfr", "-frames:v", str(len(indices)),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, spec["height"], spec["width"], 3)
    if len(frames) != len(indices):
        raise ValueError("Video is missing predeclared sample frames")
    return [Image.fromarray(frame) for frame in frames]


def evaluate(plan_path, out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text())
    if (plan["runner_sha256"] != digest(__file__) or plan["frames"] != list(FRAMES)
            or plan["processor"] != PROCESSOR or plan["source_plan_sha256"] != digest(plan["source_plan"])):
        raise ValueError("Probe protocol changed after freeze")
    model_path = Path(plan["model_path"])
    if (digest(model_path / "model.safetensors") != plan["model_sha256"]
            or digest(model_path / "config.json") != plan["config_sha256"]):
        raise ValueError("Cached model identity changed")
    for media in [*plan["references"].values(), *plan["clips"]]:
        if digest(media["path"]) != media["sha256"]:
            raise ValueError("Source media changed")
    import numpy as np
    import torch
    import transformers
    from transformers import SiglipVisionModel, SiglipImageProcessor
    torch.set_num_threads(4)
    model, loading = SiglipVisionModel.from_pretrained(model_path, local_files_only=True,
        dtype=torch.float32, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Vision checkpoint did not load completely")
    model.eval().to("cpu")
    processor = SiglipImageProcessor(**PROCESSOR)
    def features(frame):
        with torch.inference_mode():
            result = model(**processor(images=frame, return_tensors="pt"))
        return result.last_hidden_state[0].numpy(), result.pooler_output[0].numpy()
    references = {}
    for character, ref in plan["references"].items():
        patches, pooled = features(decode(ref["path"], [ref["frame"]])[0])
        mask = patch_mask(ref["normalized_head_rectangle"], 32)
        if patches.shape != (1024, 768):
            raise ValueError("Unexpected vision patch layout")
        references[character] = patches, pooled, mask
    started, records = time.monotonic(), []
    for clip in plan["clips"]:
        job = clip["job"]
        if job["stage"] != "calibration" or job["variant"] not in ("base", "character"):
            raise ValueError("This diagnostic is old qualification only")
        ref, ref_pool, mask = references[job["character"]]
        rows = []
        for index, frame in zip(FRAMES, decode(clip["path"], FRAMES)):
            patches, pooled = features(frame)
            rows.append(dict(frame=index, time_seconds=index/24,
                reference_head_maxcos=similarities(ref, patches, mask),
                global_cosine=float(np.dot(ref_pool, pooled) / (np.linalg.norm(ref_pool)*np.linalg.norm(pooled)))))
        record = dict(job=job, video_sha256=clip["sha256"], frames=rows,
            median_head_maxcos=float(np.median([r["reference_head_maxcos"] for r in rows])),
            min_head_maxcos=min(r["reference_head_maxcos"] for r in rows),
            median_global_cosine=float(np.median([r["global_cosine"] for r in rows])))
        records.append(record)
        print(json.dumps(dict(completed=job["id"], median_head=record["median_head_maxcos"])), flush=True)
    save_new(out, dict(plan_sha256=digest(plan_path), runner_sha256=digest(__file__),
        model_sha256=plan["model_sha256"], torch_version=torch.__version__, transformers_version=transformers.__version__,
        device="cpu", threads=4, elapsed_scoring_seconds=time.monotonic()-started,
        source_frame_count=2, output_frame_count=len(records)*len(FRAMES), results=records,
        quality_labels=[], preference_model_changed=False,
        limitations=plan["limitations"], not_masc=True, visibility_judged=False, audio_listened=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.plan)
    elif args.out is None:
        parser.error("evaluate requires --out")
    else:
        evaluate(args.plan, args.out)
