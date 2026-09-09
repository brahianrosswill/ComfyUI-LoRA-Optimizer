"""Isolated, calibration-only Synchformer; never imports production nodes."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

try:
    from .h3_local_av_setup import REPO, digest, inventory, now, read_json, save_new
    from .h3_local_av_evaluator import audio_control, check_generation_queue, decode_audio
except ImportError:
    from h3_local_av_setup import REPO, digest, inventory, now, read_json, save_new
    from h3_local_av_evaluator import audio_control, check_generation_queue, decode_audio

ROOT = REPO / ".h3-study-artifacts/20260909/synchformer"
SOURCE = ROOT / "source"
POLICY = REPO / "docs/research/data/2026-09-09-h3-synchformer-policy.json"
CKPT = ROOT / "24-01-04T16-39-21.pt"
CFG = ROOT / "cfg-24-01-04T16-39-21.yaml"
WEIGHT_SHA = "5b4b3557fbd96b61aaffa8bc70b28f9ff53f8fa98edc202655c5d94ab3c719ee"
CFG_SHA = "4785fa7f341e038c1a947d55eebc056a0807975731d71fede813a5d65924a60f"
GRID = [round(-2 + i * .2, 1) for i in range(21)]
MANIFEST_SHA = "a51212a23027089f5958b69a80418d510f029d4f5f2328064fcc95905dcfc967"


def isolate():
    for name, folder in (("HF_HOME", "hf-home"), ("XDG_CACHE_HOME", "xdg-cache"),
                         ("NUMBA_CACHE_DIR", "numba-cache")):
        os.environ[name] = str(ROOT / folder)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
                 "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_XET"):
        os.environ[name] = "1"
    (ROOT / "xdg-cache/torch/kernels").mkdir(parents=True, exist_ok=True)


def nearest_frame_map(pts, count=125, fps=25):
    if (len(pts) < 2 or any(not math.isfinite(t) for t in pts)
            or abs(pts[0]) > 1e-6 or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("Expected finite, strictly increasing source PTS starting at zero")
    if type(count) is not int or count < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid target timing")
    if (count - 1) / fps > pts[-1]:
        raise ValueError("No extrapolation beyond source PTS")
    result = []
    for i in range(count):
        t = i / fps
        j = bisect.bisect_left(pts, t)
        candidates = [k for k in (j - 1, j) if 0 <= k < len(pts)]
        result.append(min(candidates, key=lambda k: (abs(pts[k] - t), k)))
    return result


def distribution(logits, silent=False):
    if len(logits) != 21 or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
        raise ValueError("21 finite numeric logits required")
    values = [math.exp(x - max(logits)) for x in logits]
    probs = [x / sum(values) for x in values]
    modal = max(range(21), key=lambda i: probs[i])
    return {"logits": logits, "probabilities": probs, "modal_class": modal,
            "raw_modal_offset_seconds": GRID[modal],
            "raw_near_zero_mass": sum(probs[9:12]),
            "assessable": not silent, "reported_offset_seconds": None if silent else GRID[modal],
            "abstention": "exact silent model waveform" if silent else None}


def control_gates(rows):
    expected = {(c, k, 0) for c in ("A05", "A09") for k in ("original", "muted", "delay750")}
    expected |= {("A05", k, 1) for k in ("original", "muted", "delay750")}
    keys = [(r["case_id"], r["condition"], r["repeat"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Exact frozen nine-case scope required")
    if (any(r["status"] != "ok" for r in rows)
            or any(not math.isfinite(r["input"]["audio_peak"]) or r["input"]["audio_peak"] < 0
                   or (r["condition"] != "muted" and r["input"]["audio_peak"] == 0) for r in rows)):
        return {"technical_gate": False, "necessary_controls_pass": False,
                "qualified_for_av_quality": False}
    values = {}
    for key, row in zip(keys, rows):
        d = distribution(row["prediction"]["logits"], row["input"]["audio_peak"] == 0)
        if d != row["prediction"]:
            raise ValueError("Stored predictions do not match raw logits or silence guard")
        values[key] = d
    original, delayed = [values[("A05", k, 0)] for k in ("original", "delay750")]
    delta = delayed["raw_modal_offset_seconds"] - original["raw_modal_offset_seconds"]
    direction = delta < 0 and abs(delta + .75) <= .4
    ranking = delayed["raw_near_zero_mass"] < original["raw_near_zero_mass"]
    repeats = all(values[("A05", k, 0)]["modal_class"] == values[("A05", k, 1)]["modal_class"]
                  and max(abs(a - b) for a, b in zip(values[("A05", k, 0)]["probabilities"],
                                                    values[("A05", k, 1)]["probabilities"])) <= 1e-5
                  for k in ("original", "muted", "delay750"))
    silence = all(not d["assessable"] and d["reported_offset_seconds"] is None
                  for (c, k, r), d in values.items() if k == "muted")
    return {"technical_gate": True, "direction_gate": direction, "ranking_gate": ranking,
            "repeatability_gate": repeats, "harness_silence_abstention_gate": silence,
            "modal_offset_delta_seconds": delta,
            "necessary_controls_pass": direction and ranking and repeats and silence,
            "qualified_for_av_quality": False, "qualified_for_heldout": False}


def source_files():
    names = subprocess.check_output(["git", "-C", str(SOURCE), "ls-files"], text=True).splitlines()
    return {str(SOURCE / p): digest(SOURCE / p) for p in names if p.endswith((".py", ".yaml", ".yml"))}


def safe_checkpoint():
    """Reviewed configuration/scalar containers only, not arbitrary pickle fallback."""
    import collections
    import typing
    import numpy as np
    import torch
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    expected = {"omegaconf.dictconfig.DictConfig", "omegaconf.listconfig.ListConfig",
                "omegaconf.base.ContainerMetadata", "omegaconf.base.Metadata", "omegaconf.nodes.AnyNode",
                "numpy.core.multiarray.scalar", "numpy.dtype", "builtins.dict", "builtins.list",
                "builtins.int", "typing.Any", "collections.defaultdict"}
    actual = set(torch.serialization.get_unsafe_globals_in_checkpoint(CKPT))
    if actual != expected:
        raise ValueError(f"Unexpected checkpoint globals: {actual ^ expected}")
    allowed = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, dict, list, int,
               typing.Any, collections.defaultdict, np.dtype, np.dtypes.Float64DType,
               (np._core.multiarray.scalar, "numpy.core.multiarray.scalar")]
    with torch.serialization.safe_globals(allowed):
        value = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = value["model"]
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Tensor-only model state required")
    # Do not evaluate checkpoint args, interpolation strings, optimizer, or metric objects.
    return state, sorted(actual)


def setup():
    isolate()
    import torch
    policy = read_json(POLICY)
    if digest(CKPT) != WEIGHT_SHA or digest(CFG) != CFG_SHA:
        raise ValueError("Checkpoint/config hash mismatch")
    with CKPT.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if CKPT.stat().st_size != policy["checkpoint_bytes"] or md5 != policy["checkpoint_published_md5"]:
        raise ValueError("Official checkpoint size/MD5 mismatch")
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if head != policy["source_revision"]:
        raise ValueError("Unexpected source revision")
    state, globals_ = safe_checkpoint()
    torch.set_num_threads(4)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite checkpoint tensor")
    record = {"completed_at_utc": now(), "policy_sha256": digest(POLICY), "source_revision": head,
              "checkpoint_sha256": WEIGHT_SHA, "checkpoint_md5": md5, "checkpoint_bytes": CKPT.stat().st_size,
              "config_sha256": CFG_SHA, "source_files": source_files(), "safe_globals": globals_,
              "weights_only": True, "model_tensors": len(state), "model_parameters": sum(v.numel() for v in state.values()),
              "packages": inventory(), "python": sys.version, "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True),
              "requirements_sha256": digest(REPO / "docs/research/h3-synchformer-requirements.txt"),
              "helper_sha256": digest(__file__), "model_quality_qualified": False}
    save_new(ROOT / "setup-receipt.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("source_files", "packages", "ffmpeg")}))


def initialize_source():
    isolate()
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / "model/modules/feat_extractors/visual"))
    os.chdir(SOURCE)
    from utils import utils
    def require_local(path, *args, **kwargs):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Offline inference cannot fetch {path}")
    utils.check_if_file_exists_else_download = require_local
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(CFG)
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    return cfg, utils.instantiate_from_config


def decode_video(path):
    import numpy as np
    import torch
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,width,height", "-of", "json", str(path)], text=True))
    frames = probe["frames"]
    pts = [float(f["best_effort_timestamp_time"]) for f in frames]
    if any((f["width"], f["height"]) != (640, 384) for f in frames) or len(frames) != 124:
        raise ValueError("Unexpected calibration geometry/frame count")
    indices = nearest_frame_map(pts)
    errors = [abs(pts[j] - i / 25) for i, j in enumerate(indices)]
    if max(errors) > 1 / 48 + 1e-6:
        raise ValueError("Source PTS mismatch exceeds half a 24-fps frame")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", str(path),
        "-map", "0:v:0", "-vf", "scale=426:256:flags=bicubic", "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 256, 426, 3)
    if len(decoded) != len(pts):
        raise ValueError("FFmpeg decoded frames differ from PTS records")
    mapped = decoded[indices].copy()
    video = torch.from_numpy(mapped).permute(0, 3, 1, 2)
    meta = {"source_pts_seconds": pts, "source_frame_indices": indices,
            "target_pts_seconds": [i / 25 for i in range(125)], "max_time_error_seconds": max(errors),
            "resized_geometry": [426, 256], "resize": "ffmpeg bicubic, no video re-encode",
            "center_crop_xywh": [101, 16, 224, 224],
            "raw_resized_sha256": hashlib.sha256(raw).hexdigest(),
            "mapped_rgb_sha256": hashlib.sha256(mapped.tobytes()).hexdigest()}
    return video, meta


def tensor_record(value):
    import torch
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all()),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def run(name):
    import numpy as np
    import torch
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a simple new run name")
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=False)
    isolate()
    policy, receipt = read_json(POLICY), read_json(ROOT / "setup-receipt.json")
    if receipt["policy_sha256"] != digest(POLICY) or digest(CKPT) != WEIGHT_SHA or digest(CFG) != CFG_SHA:
        raise ValueError("Frozen asset/policy mismatch")
    if source_files() != receipt["source_files"]:
        raise ValueError("Source changed since audit")
    manifest_path = REPO / policy["manifest"]
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("Frozen source manifest changed")
    manifest = read_json(manifest_path)
    if manifest["policy_sha256"] != policy["parent_av_policy_sha256"]:
        raise ValueError("Wrong calibration policy")
    cases = [c for c in manifest["cases"] if c["case_id"] in ("A05", "A09")]
    if len(cases) != 2 or any("2026090803" not in c["source_job"] for c in cases):
        raise ValueError("Wrong calibration scope")
    jobs = [(c, k, 0) for c in cases for k in ("original", "muted", "delay750")]
    jobs += [(cases[0], k, 1) for k in ("original", "muted", "delay750")]
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("Insufficient free CUDA memory; no eviction/CPU fallback")
    check_generation_queue()
    save_new(run_dir / "start.json", {"started_at_utc": now(), "policy_sha256": digest(POLICY),
        "manifest_sha256": digest(manifest_path), "receipt_sha256": digest(ROOT / "setup-receipt.json"),
        "runner_sha256": digest(__file__), "audio_helper_sha256": digest(REPO / "scripts/h3_local_av_evaluator.py"),
        "jobs": [(c["case_id"], k, r) for c, k, r in jobs], "packages": inventory()})
    rows = []
    try:
        cfg, instantiate = initialize_source()
        from torchvision.transforms import Compose
        transforms = Compose([instantiate(x) for x in cfg.transform_sequence_test])
        segmenter = next(x for x in transforms.transforms if type(x).__name__ == "GenerateMultipleSegments")
        vranges, aranges = segmenter.get_sequential_seg_ranges(125, 80000, 25, 16000, 14, 10240)
        if (vranges.tolist() != [[2 + 8 * i, 18 + 8 * i] for i in range(14)]
                or aranges.tolist() != [[1280 + 5120 * i, 11520 + 5120 * i] for i in range(14)]):
            raise ValueError("Official segment ranges differ from declared coverage")
        state, _ = safe_checkpoint()
        model = instantiate(cfg.model)
        keys_before = {k: tuple(v.shape) for k, v in state.items()}
        if keys_before != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
            raise ValueError("Exact state keys/shapes required; no silent positional trimming")
        model.load_state_dict(state, strict=True)
        del state
        model.eval().to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        for case, condition, repeat in jobs:
            check_generation_queue()
            began = time.monotonic()
            if digest(case["path"]) != case["sha256"]:
                raise ValueError("Source video changed")
            rgb, video_info = decode_video(case["path"])
            audio, audio_info = decode_audio(case["path"])
            controlled = audio_control(audio, condition)
            item = {"video": rgb, "audio": torch.from_numpy(controlled), "path": "calibration-input",
                    "split": "test", "targets": {"offset_sec": 0., "v_start_i_sec": 0.},
                    "meta": {"video": {"fps": [25.]}, "audio": {"framerate": [16000.]}}}
            transformed = transforms(item)
            vid, aud = transformed["video"].unsqueeze(0), transformed["audio"].unsqueeze(0)
            vrec, arec = tensor_record(vid), tensor_record(aud)
            if not vrec["finite"] or not arec["finite"] or list(vid.shape) != [1, 14, 16, 3, 224, 224] or list(aud.shape) != [1, 14, 1, 128, 66]:
                raise ValueError("Unexpected model input")
            info = {"audio": audio_info, "video": video_info,
                    "waveform_sha256": hashlib.sha256(controlled.tobytes()).hexdigest(),
                    "audio_peak": float(np.max(np.abs(controlled))),
                    "video_tensor": vrec, "audio_tensor": arec,
                    "segment_video_ranges": vranges.tolist(),
                    "segment_audio_ranges": aranges.tolist()}
            row = {"case_id": case["case_id"], "condition": condition, "repeat": repeat,
                   "source_sha256": case["sha256"], "input": info}
            stem = f'{case["case_id"]}-{condition}-r{repeat}'
            save_new(run_dir / (stem + ".input.json"), row)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _, logits = model(vid.to("cuda:0"), aud.to("cuda:0"))
            prediction = distribution(logits[0].float().cpu().tolist(), info["audio_peak"] == 0)
            row.update(status="ok", prediction=prediction, wall_seconds=time.monotonic() - began,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), completed_at_utc=now())
            save_new(run_dir / (stem + ".json"), row)
            rows.append(row)
            print(json.dumps({"event": "case_complete", "case": stem, "prediction": prediction,
                              "wall_seconds": row["wall_seconds"]}), flush=True)
        save_new(run_dir / "complete.json", {"completed_at_utc": now(), "cases": len(rows),
            "gates": control_gates(rows), "queue": check_generation_queue()})
        print(json.dumps({"event": "complete", "gates": control_gates(rows)}), flush=True)
    except Exception:
        save_new(run_dir / "failure.json", {"utc": now(), "completed_cases": len(rows), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "controls"))
    parser.add_argument("--run-name", default="controls-01")
    args = parser.parse_args()
    setup() if args.mode == "setup" else run(args.run_name)
