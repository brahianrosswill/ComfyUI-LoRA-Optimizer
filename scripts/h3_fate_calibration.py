"""One frozen twelve-original FATE calibration; never changes production nodes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
import traceback

try:
    from . import h3_fate_controls as c
    from .h3_synchformer_calibration import pairwise
except ImportError:
    import h3_fate_controls as c
    from h3_synchformer_calibration import pairwise

POLICY = c.REPO / "docs/research/data/2026-09-09-h3-fate-calibration-policy.json"
PARENT = c.REPO / "docs/research/data/2026-09-09-h3-fate-checkpoint.json"
RUN = c.ROOT / "runs/rated-calibration-01"
PAIRS = ("combat_cinema", "combat_repair")
VARIANTS = {"base", "combat", "additive", "np", "ct", "winner", "second"}


def validate_cases(cases):
    if ([r["case_id"] for r in cases] != [f"A{i:02d}" for i in range(1, 13)]
            or len({r["sha256"] for r in cases}) != 12):
        raise ValueError("Exact twelve unique calibration sources in A01..A12 order required")


def verify():
    policy = c.read_json(POLICY)
    audit = c.ROOT / "runs/controls-02/independent-audit.json"
    manifest = c.REPO / policy["manifest"]
    if (c.digest(PARENT) != policy["parent_checkpoint_sha256"]
            or c.digest(audit) != policy["control_audit_sha256"]
            or c.digest(manifest) != policy["manifest_sha256"]):
        raise ValueError("Frozen prerequisite changed")
    parent = c.read_json(PARENT)
    if not parent["control_gates"]["necessary_controls_pass"]:
        raise ValueError("Necessary controls did not pass")
    for path, wanted in parent["artifact_sha256"].items():
        if c.digest(c.REPO / path) != wanted:
            raise ValueError(f"Parent evidence changed: {path}")
    cases = c.read_json(manifest)["cases"]
    validate_cases(cases)
    return policy, cases


def join_labels(cases, predictions, labels):
    """Preserve every contextual R1 entry; never average duplicate-source labels."""
    validate_cases(cases)
    by_hash = {x["sha256"]: x["case_id"] for x in cases}
    if set(predictions) != set(by_hash) or len(labels) != 14:
        raise ValueError("Twelve predictions and all fourteen original ratings required")
    if {x["original_sha256"] for x in labels} != set(by_hash):
        raise ValueError("Ratings must cover exactly the original media hashes")
    for pair in PAIRS:
        group = [r for r in labels if r["pair"] == pair]
        if len(group) != 7 or {r["variant"] for r in group} != VARIANTS:
            raise ValueError("Each fixed pairing needs its seven original arms")
    if len({(r["pair"], r["blind_id"]) for r in labels}) != 14:
        raise ValueError("Duplicate contextual rating identity")
    result = []
    for label in labels:
        score = label["scores"]["synchronization"]
        metric = predictions[label["original_sha256"]]
        if (label["reviewer_id"] != "R1" or label["seed"] != 2026090803 or label["stage"] != "calibration"
                or type(score) is not int or not 0 <= score <= 4
                or type(metric) not in (int, float) or not math.isfinite(metric)):
            raise ValueError("Unexpected label scope or invalid score")
        result.append({"pair": label["pair"], "variant": label["variant"], "blind_id": label["blind_id"],
            "case_id": by_hash[label["original_sha256"]], "source_sha256": label["original_sha256"],
            "human_sync": score, "fate_score": metric})
    return result


def run():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    c.isolate()
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    policy, cases = verify()
    c.check_generation_queue()
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("Require idle CUDA GPU with >=20 GiB free")
    RUN.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(c.__file__), c.REPO / "scripts/h3_fate.py", c.REPO / "scripts/h3_fate_setup.py",
             c.REPO / "scripts/h3_synchformer_calibration.py", POLICY, PARENT]
    c.save_new(RUN / "intent.json", {"utc": c.now(), "policy": policy, "cases": cases, "packages": c.inventory(),
        "artifact_sha256": {str(p.relative_to(c.REPO)): c.digest(p) for p in files}})
    for p in files[:5]:
        shutil.copyfile(p, RUN / p.name)
    rows = []
    try:
        # All source checks occur before CUDA model loading, as in the amended control run.
        decoded = {case["case_id"]: c.decode_source(case) for case in cases}
        for case_id, (_, _, source) in decoded.items():
            c.save_new(RUN / f"{case_id}-source.json", source)
        c.check_generation_queue()
        if torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise RuntimeError("GPU availability changed before model load")
        model, processor, loading = c.load_model()
        c.save_new(RUN / "loading.json", loading)
        model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        for case in cases:
            case_id = case["case_id"]
            rgb, wave, _ = decoded[case_id]
            row = {"case_id": case_id, "source_sha256": case["sha256"], "windows": []}
            started = time.monotonic()
            for window_id, (start, end) in enumerate(policy["window_seconds"]):
                c.check_generation_queue()
                video = torch.from_numpy(rgb[round(start * 24):round(end * 24)].copy()).permute(0, 3, 1, 2)
                audio = wave[round(start * 48000):round(end * 48000)].copy()
                inputs = processor(videos=video, audio=audio, sampling_rate=48000, padding=True, return_tensors="pt")
                records = {k: c.tensor_record(v) for k, v in inputs.items()}
                if (list(inputs["input_values"].shape) != [1, 1, 96000]
                        or list(inputs["pixel_values_videos"].shape) != [1, 48, 3, 336, 336]
                        or not bool(torch.all(inputs["padding_mask"] == 1))
                        or not all(v["finite"] for v in records.values())):
                    raise ValueError("Actual processor inputs fail unchanged contract")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**{k: v.to("cuda") for k, v in inputs.items()})
                a, v = output.audio_frame_embeds.float(), output.video_frame_embeds.float()
                if (list(a.shape) != [1, 50, 1024] or v.shape != a.shape
                        or not bool(torch.isfinite(a).all() and torch.isfinite(v).all())
                        or bool(torch.any(a.norm(dim=-1) == 0) or torch.any(v.norm(dim=-1) == 0))):
                    raise ValueError("Output embeddings fail unchanged contract")
                a, v = F.normalize(a, dim=-1).cpu(), F.normalize(v, dim=-1).cpu()
                diagonal = (a * v).sum(-1)[0]
                stem = f"{case_id}-w{window_id}"
                path = RUN / (stem + ".safetensors")
                save_file({"audio": a.contiguous(), "video": v.contiguous()}, path)
                window = {"window_id": window_id, "seconds": [start, end],
                    "source_frame_indices": list(range(round(start * 24), round(end * 24))),
                    "source_audio_sample_range": [round(start * 48000), round(end * 48000)],
                    "input_tensors": records, "audio_peak": float(np.abs(audio).max()),
                    "alignment": model.get_base_model().last_alignment,
                    "diagonal": diagonal.tolist(), "score": float(diagonal.mean()),
                    "embeddings_file": path.name, "embeddings_sha256": c.digest(path)}
                c.save_new(RUN / (stem + ".json"), window)
                row["windows"].append(window)
                del inputs, output, a, v
            row.update(raw_score=sum(w["score"] for w in row["windows"]) / 3,
                       assessable=any(w["audio_peak"] > 0 for w in row["windows"]),
                       seconds_elapsed=time.monotonic() - started)
            row["reported_score"] = row["raw_score"] if row["assessable"] else None
            c.save_new(RUN / f"{case_id}.json", row)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
        c.save_new(RUN / "summary.json", {"utc": c.now(), "status": "complete", "rows": rows,
            "intent_sha256": c.digest(RUN / "intent.json"), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "qualified_for_av_quality": False, "heldout_admitted": False, "production_changed": False})
    except Exception:
        c.save_new(RUN / "failure.json", {"utc": c.now(), "completed_cases": len(rows), "traceback": traceback.format_exc(),
            "qualified_for_av_quality": False, "production_changed": False})
        raise


def compare():
    policy, cases = verify()
    summary = c.read_json(RUN / "summary.json")
    if summary["status"] != "complete" or len(summary["rows"]) != 12:
        raise ValueError("Complete twelve-clip run required")
    ratings_path = c.REPO / policy["human_ratings"]
    if c.digest(ratings_path) != policy["human_ratings_sha256"]:
        raise ValueError("Original R1 ratings changed")
    predictions = {}
    for row in summary["rows"]:
        if row != c.read_json(RUN / f"{row['case_id']}.json"):
            raise ValueError("Stored case differs from summary")
        predictions[row["source_sha256"]] = row["reported_score"]
    labels = c.read_json(ratings_path)["ratings"]
    rows = join_labels(cases, predictions, labels)
    comparisons = {pair: pairwise([r for r in rows if r["pair"] == pair], "fate_score") for pair in PAIRS}
    result = {"utc": c.now(), "policy_sha256": c.digest(POLICY), "ratings_sha256": c.digest(ratings_path),
        "summary_sha256": c.digest(RUN / "summary.json"), "rows": rows, "comparisons": comparisons,
        "one_reviewer_only": True, "shared_clips_are_correlated": True, "scores_fitted": False,
        "qualified_for_av_quality": False, "heldout_admitted": False, "production_changed": False}
    c.save_new(RUN / "human-comparison.json", result)
    print(json.dumps({p: {k:v for k,v in r.items() if k != "comparisons"} for p,r in comparisons.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "compare"))
    {"run": run, "compare": compare}[parser.parse_args().action]()
