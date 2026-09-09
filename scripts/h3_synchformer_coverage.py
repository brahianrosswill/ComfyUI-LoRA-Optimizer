"""Inspect existing evaluator coverage without changing or rerunning its scores."""
from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from . import h3_synchformer as s
except ImportError:
    import h3_synchformer as s

POLICY = s.REPO / "docs/research/data/2026-09-09-h3-synchformer-coverage-policy.json"
PRIOR = s.REPO / "docs/research/data/2026-09-08-h3-av2-sync-seed03.json"
CHECKPOINT = s.REPO / "docs/research/data/2026-09-09-h3-synchformer-checkpoint.json"
OUT = s.ROOT / "coverage-01"


def covered_indices(mapping, ranges, source_count):
    if (type(source_count) is not int or source_count < 1 or not mapping or not ranges
            or any(type(i) is not int or i < 0 or i >= source_count for i in mapping)):
        raise ValueError("Invalid source mapping")
    if any(b < a for a, b in zip(mapping, mapping[1:])):
        raise ValueError("Chronological mapping required; duplicates are permitted")
    targets = set()
    for a, b in ranges:
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(mapping):
            raise ValueError("Invalid half-open segment range")
        targets.update(range(a, b))
    sources = sorted({mapping[i] for i in targets})
    return {"target_slots": sorted(targets), "source_frames": sources,
            "excluded_source_frames": sorted(set(range(source_count)) - set(sources))}


def coverage_fraction(width, height, crop):
    if any(type(x) is not int for x in (width, height, *crop)) or width <= 0 or height <= 0 or len(crop) != 4:
        raise ValueError("Integer image/crop geometry required")
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("Crop outside image")
    return w * h / (width * height)


def make_panels(video, video_info, case, out):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 13)
    mapping = video_info["source_frame_indices"]
    result = []
    for left, right in case["frame_ranges_inclusive"]:
        indices = list(range(left, right + 1))
        for first in range(0, len(indices), 4):
            page_indices = indices[first:first + 4]
            page = Image.new("RGB", (1376, 600), "#15191f")
            draw = ImageDraw.Draw(page)
            for slot, i in enumerate(page_indices):
                target = mapping.index(i)
                rgb = video[target].permute(1, 2, 0).numpy()
                full = Image.fromarray(rgb)
                # This rectangle is a display overlay only; crop uses untouched pixels.
                crop = full.crop((101, 16, 325, 240))
                full = full.copy()
                ImageDraw.Draw(full).rectangle((101, 16, 324, 239), outline="#ff3232", width=2)
                x, y = (slot % 2) * 688 + 5, (slot // 2) * 300 + 32
                page.paste(full, (x, y))
                page.paste(crop, (x + 438, y + 16))
                t = video_info["source_pts_seconds"][i]
                draw.text((x, y - 25), f'{case["case_id"]} source f{i:03d} {t:.6f}s | full resized + exact crop',
                          font=font, fill="white")
            path = out / f'{case["case_id"]}-f{page_indices[0]:03d}-{page_indices[-1]:03d}.png'
            page.save(path)
            result.append({"path": str(path), "sha256": s.digest(path), "source_frames": page_indices})
    return result


def prepare():
    s.isolate()
    policy, prior = s.read_json(POLICY), s.read_json(PRIOR)
    if s.digest(PRIOR) != policy["prior_event_observations_sha256"] or s.digest(CHECKPOINT) != policy["prior_checkpoint_sha256"]:
        raise ValueError("Prior evidence changed")
    checkpoint = s.read_json(CHECKPOINT)
    manifest = s.read_json(s.REPO / s.read_json(s.POLICY)["manifest"])
    OUT.mkdir(parents=True, exist_ok=False)
    pins = {str(POLICY): s.digest(POLICY), str(PRIOR): s.digest(PRIOR), str(CHECKPOINT): s.digest(CHECKPOINT),
            str(Path(__file__).resolve()): s.digest(__file__)}
    records = []
    control = s.read_json(s.ROOT / "runs/controls-01/A09-original-r0.input.json")
    ranges = control["input"]["segment_video_ranges"]
    for case in policy["cases"]:
        source = next(c for c in manifest["cases"] if c["case_id"] == case["case_id"])
        previous = next(c for c in prior["clips"] if c["job_id"] == source["source_job"])
        sync_path = s.REPO / ".h3-study-artifacts/20260908" / case["prior_diagnostic_directory"] / "sync.json"
        model_path = s.ROOT / "runs/rated-calibration-01" / (case["case_id"] + ".json")
        if (s.digest(sync_path) != previous["sync_json_sha256"] or s.digest(source["path"]) != source["sha256"]
                or source["sha256"] != previous["video_sha256"]
                or s.digest(model_path) != checkpoint["artifact_sha256"][str(model_path)]):
            raise ValueError("Model, diagnostic or media provenance changed")
        model, diagnostic = s.read_json(model_path), s.read_json(sync_path)
        rgb, vi = s.decode_video(source["path"])
        if vi != model["input"]["video"]:
            raise ValueError("Reconstructed resize/mapping differs from evaluated input")
        coverage = covered_indices(vi["source_frame_indices"], ranges, len(vi["source_pts_seconds"]))
        selected = sorted({f for a, b in case["frame_ranges_inclusive"] for f in range(a, b + 1)})
        panels = make_panels(rgb, vi, case, OUT)
        record = {"case_id": case["case_id"], "source_sha256": source["sha256"], "coverage": coverage,
                  "selected_source_frames": selected,
                  "selected_frames_excluded": sorted(set(selected) - set(coverage["source_frames"])),
                  "crop_area_fraction": coverage_fraction(426, 256, vi["center_crop_xywh"]),
                  "audio_window_seconds": [.08, 4.88], "prior_observations": previous["observations"],
                  "prior_anonymous_energy_maxima": diagnostic["burst_candidates"], "panels": panels,
                  "visual_contact_coverage_review": "pending", "new_sync_score": None}
        records.append(record)
        for p in (sync_path, model_path, Path(source["path"])):
            pins[str(p)] = s.digest(p)
        pins.update({p["path"]: p["sha256"] for p in panels})
    record = {"prepared_at_utc": s.now(), "cases": records, "artifact_sha256": pins,
              "new_model_inferences": 0, "new_human_labels": 0, "heldout_access": False,
              "visual_review_complete": False, "production_changed": False}
    s.save_new(OUT / "prepared.json", record)
    print(json.dumps({"cases": len(records), "panels": sum(len(c["panels"]) for c in records),
                      "unique_selected_frames": sum(len(c["selected_source_frames"]) for c in records),
                      "all_selected_frames_temporally_covered": not any(c["selected_frames_excluded"] for c in records)}))


if __name__ == "__main__":
    prepare()
