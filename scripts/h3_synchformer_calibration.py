"""Frozen natural-clip calibration; machine sync estimates are not human labels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import traceback

try:
    from . import h3_synchformer as s
except ImportError:
    import h3_synchformer as s

POLICY = s.REPO / "docs/research/data/2026-09-09-h3-synchformer-calibration-policy.json"
RUN = s.ROOT / "runs/rated-calibration-01"
RATINGS = s.REPO / "docs/research/data/2026-09-08-h3-av2-ratings-seed03.json"


def pairwise(rows, metric):
    result = []
    for a, b in itertools.combinations(rows, 2):
        h1, h2, m1, m2 = a["human_sync"], b["human_sync"], a[metric], b[metric]
        if any(x is None for x in (h1, h2, m1, m2)):
            relation = "missing"
        elif any(type(x) not in (int, float) or not math.isfinite(x) for x in (h1, h2, m1, m2)):
            raise ValueError("Finite numeric scores or explicit null required")
        elif h1 == h2:
            relation = "both_tied" if m1 == m2 else "human_tied"
        elif m1 == m2:
            relation = "model_tied"
        else:
            relation = "concordant" if (h1 - h2) * (m1 - m2) > 0 else "discordant"
        result.append({"a": a["blind_id"], "b": b["blind_id"], "relation": relation,
                       "human": [h1, h2], "model": [m1, m2]})
    counts = {k: sum(r["relation"] == k for r in result) for k in
              ("concordant", "discordant", "model_tied", "human_tied", "both_tied", "missing")}
    eligible = counts["concordant"] + counts["discordant"] + counts["model_tied"]
    return {"comparisons": result, "counts": counts, "human_untied_eligible": eligible,
            "tie_adjusted_descriptive_fraction": ((counts["concordant"] + .5 * counts["model_tied"]) / eligible
                                                   if eligible else None)}


def verify():
    policy = s.read_json(POLICY)
    audit = s.ROOT / "runs/controls-01/control-audit.json"
    manifest_path = s.REPO / s.read_json(s.POLICY)["manifest"]
    if (s.digest(s.POLICY) != policy["parent_policy_sha256"] or s.digest(audit) != policy["control_audit_sha256"]
            or not s.read_json(audit)["gates"]["necessary_controls_pass"]
            or s.digest(manifest_path) != policy["manifest_sha256"] or s.digest(RATINGS) != policy["human_ratings_sha256"]):
        raise ValueError("Frozen calibration prerequisite mismatch")
    return s.read_json(manifest_path)["cases"]


def run():
    import numpy as np
    import torch
    s.isolate()
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    cases = verify()
    if [c["case_id"] for c in cases] != [f"A{i:02d}" for i in range(1, 13)]:
        raise ValueError("Exact 12-clip calibration order required")
    receipt = s.read_json(s.ROOT / "setup-receipt.json")
    if s.digest(s.CKPT) != s.WEIGHT_SHA or s.digest(s.CFG) != s.CFG_SHA or s.source_files() != receipt["source_files"]:
        raise ValueError("Verified model/source changed")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("Insufficient idle GPU memory")
    s.check_generation_queue()
    RUN.mkdir(parents=True, exist_ok=False)
    s.save_new(RUN / "start.json", {"started_at_utc": s.now(), "policy_sha256": s.digest(POLICY),
        "runner_sha256": s.digest(__file__), "input_helper_sha256": s.digest(s.__file__),
        "audio_helper_sha256": s.digest(s.REPO / "scripts/h3_local_av_evaluator.py"),
        "receipt_sha256": s.digest(s.ROOT / "setup-receipt.json"), "packages": s.inventory(), "cases": cases})
    count = 0
    try:
        cfg, instantiate = s.initialize_source()
        from torchvision.transforms import Compose
        transforms = Compose([instantiate(x) for x in cfg.transform_sequence_test])
        state, _ = s.safe_checkpoint()
        model = instantiate(cfg.model)
        if {k: tuple(v.shape) for k, v in state.items()} != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
            raise ValueError("Exact model state required")
        model.load_state_dict(state, strict=True)
        del state
        model.eval().to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        for case in cases:
            s.check_generation_queue()
            began = time.monotonic()
            if s.digest(case["path"]) != case["sha256"]:
                raise ValueError("Source media changed")
            rgb, vi = s.decode_video(case["path"])
            audio, ai = s.decode_audio(case["path"])
            item = transforms({"video": rgb, "audio": torch.from_numpy(audio), "path": "calibration-input",
                "split": "test", "targets": {"offset_sec": 0., "v_start_i_sec": 0.},
                "meta": {"video": {"fps": [25.]}, "audio": {"framerate": [16000.]}}})
            vid, aud = item["video"].unsqueeze(0), item["audio"].unsqueeze(0)
            vr, ar = s.tensor_record(vid), s.tensor_record(aud)
            if (not vr["finite"] or not ar["finite"] or vr["shape"] != [1, 14, 16, 3, 224, 224]
                    or ar["shape"] != [1, 14, 1, 128, 66]):
                raise ValueError("Unexpected model input")
            row = {"case_id": case["case_id"], "source_sha256": case["sha256"], "input": {
                "audio": ai, "video": vi, "audio_peak": float(np.max(np.abs(audio))),
                "waveform_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "video_tensor": vr, "audio_tensor": ar}}
            s.save_new(RUN / (case["case_id"] + ".input.json"), row)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _, logits = model(vid.to("cuda:0"), aud.to("cuda:0"))
            row.update(status="ok", prediction=s.distribution(logits[0].float().cpu().tolist(), row["input"]["audio_peak"] == 0),
                       wall_seconds=time.monotonic() - began, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            s.save_new(RUN / (case["case_id"] + ".json"), row)
            count += 1
            print(json.dumps({"event": "case_complete", "case": case["case_id"],
                "offset": row["prediction"]["reported_offset_seconds"], "near_zero_mass": row["prediction"]["raw_near_zero_mass"]}), flush=True)
        s.save_new(RUN / "complete.json", {"completed_at_utc": s.now(), "cases": count,
            "queue": s.check_generation_queue(), "qualified_for_av_quality": False, "qualified_for_heldout": False})
    except Exception:
        s.save_new(RUN / "failure.json", {"utc": s.now(), "completed_cases": count, "traceback": traceback.format_exc()})
        raise


def compare():
    cases = verify()
    start, complete = s.read_json(RUN / "start.json"), s.read_json(RUN / "complete.json")
    if complete["cases"] != 12 or start["policy_sha256"] != s.digest(POLICY):
        raise ValueError("Incomplete/mismatched run")
    pins = {str(POLICY): s.digest(POLICY), str(RATINGS): s.digest(RATINGS)}
    predictions = {}
    for case in cases:
        path, ipath = RUN / (case["case_id"] + ".json"), RUN / (case["case_id"] + ".input.json")
        row, before = s.read_json(path), s.read_json(ipath)
        if row["status"] != "ok" or row["source_sha256"] != case["sha256"] or any(row[k] != v for k, v in before.items()):
            raise ValueError("Invalid or changed inference")
        d = s.distribution(row["prediction"]["logits"], row["input"]["audio_peak"] == 0)
        if d != row["prediction"]:
            raise ValueError("Prediction does not match logits")
        predictions[case["sha256"]] = d
        pins.update({str(path): s.digest(path), str(ipath): s.digest(ipath)})
    labels = s.read_json(RATINGS)["ratings"]
    if len(labels) != 14:
        raise ValueError("All original rating entries required")
    rows = []
    for label in labels:
        d = predictions[label["original_sha256"]]
        rows.append({"pair": label["pair"], "variant": label["variant"], "blind_id": label["blind_id"],
            "source_sha256": label["original_sha256"], "human_sync": label["scores"]["synchronization"],
            "near_zero_mass": d["raw_near_zero_mass"] if d["assessable"] else None,
            "negative_abs_offset": -abs(d["reported_offset_seconds"]) if d["assessable"] else None,
            "modal_offset": d["reported_offset_seconds"]})
    comparisons = {pair: {metric: pairwise([r for r in rows if r["pair"] == pair], metric)
        for metric in ("near_zero_mass", "negative_abs_offset")} for pair in sorted({r["pair"] for r in rows})}
    result = {"completed_at_utc": s.now(), "rows": rows, "comparisons": comparisons,
        "artifact_sha256": pins, "qualified_for_av_quality": False, "qualified_for_heldout": False,
        "one_reviewer_only": True, "shared_clips_are_correlated": True, "scores_fitted": False}
    s.save_new(RUN / "human-comparison.json", result)
    print(json.dumps({p: {m: {k: v for k, v in r.items() if k != "comparisons"} for m, r in ms.items()}
                      for p, ms in comparisons.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "compare"))
    run() if parser.parse_args().mode == "run" else compare()
