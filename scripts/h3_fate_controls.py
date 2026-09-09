"""Prospective, bounded local FATE controls on two already-rated H3 sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import traceback

try:
    from .h3_fate import load_model
    from .h3_fate_setup import ROOT, REPO, digest, inventory, isolate, now, read_json, save_new
    from .h3_local_av_evaluator import check_generation_queue
    from .h3_synchformer import tensor_record
except ImportError:
    from h3_fate import load_model
    from h3_fate_setup import ROOT, REPO, digest, inventory, isolate, now, read_json, save_new
    from h3_local_av_evaluator import check_generation_queue
    from h3_synchformer import tensor_record

POLICY = REPO / "docs/research/data/2026-09-09-h3-fate-controls-policy.json"
AMENDMENT = REPO / "docs/research/data/2026-09-09-h3-fate-controls-amendment-02.json"
CONDITIONS = ("original", "muted", "delay750", "advance750", "delay125", "advance125", "delay80", "advance80", "tone440")


def controlled_audio(wave, condition, sr=48000):
    import numpy as np
    if wave.ndim != 1 or len(wave) < sr * 5 or not np.isfinite(wave).all():
        raise ValueError("At least five seconds of finite mono PCM required")
    if condition not in CONDITIONS or sr != 48000:
        raise ValueError("Condition/rate outside frozen scope")
    wave = np.asarray(wave, dtype=np.float32)
    result = np.zeros_like(wave)
    if condition == "original":
        return wave.copy()
    if condition == "muted":
        return result
    if condition == "tone440":
        energy = np.pad(wave.astype(np.float64) ** 2, (240, 239))
        cumulative = np.concatenate(([0.0], np.cumsum(energy)))
        envelope = np.sqrt(np.maximum((cumulative[480:] - cumulative[:-480]) / 480, 0))
        tone = envelope * np.sin(2 * np.pi * 440 * np.arange(len(wave)) / sr)
        denominator = math.sqrt(float(np.mean(tone ** 2)))
        if denominator <= 0:
            raise ValueError("Cannot create a tonal control from silence")
        tone *= math.sqrt(float(np.mean(wave.astype(np.float64) ** 2))) / denominator
        return tone.astype(np.float32)
    delay = condition.startswith("delay")
    n = round(int(condition.removeprefix("delay").removeprefix("advance")) * sr / 1000)
    if delay:
        result[n:] = wave[:-n]
    else:
        result[:-n] = wave[n:]
    return result


def decode_source(case):
    import numpy as np
    import torch
    import torchaudio
    if digest(case["path"]) != case["sha256"]:
        raise ValueError("Calibration media hash mismatch")
    path = case["path"]
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path], text=True))
    streams = probe["streams"]
    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise ValueError("Exactly one video and one audio stream required")
    video, audio = video_streams[0], audio_streams[0]
    if (video.get("start_time") is None or audio.get("start_time") is None
            or abs(float(video["start_time"])) > 1e-7 or abs(float(audio["start_time"])) > 1e-7
            or (video["width"], video["height"], video["avg_frame_rate"]) != (640, 384, "24/1")
            or (int(audio["sample_rate"]), audio["channels"]) != (32000, 2)):
        raise ValueError("Source format/clocks outside frozen calibration scope")
    frames = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,width,height", "-of", "json", path], text=True))["frames"]
    pts = [float(f["best_effort_timestamp_time"]) for f in frames]
    if (len(pts) != 124 or any(not math.isfinite(p) or abs(p - i / 24) > 1e-6 for i, p in enumerate(pts))
            or any((f["width"], f["height"]) != (640, 384) for f in frames)):
        raise ValueError("Actual frame PTS/count/geometry mismatch")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
        "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    if len(raw) != 124 * 384 * 640 * 3:
        raise ValueError("Video decode incomplete")
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(124, 384, 640, 3)
    pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
        "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    stereo = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2)
    if len(stereo) < 160000 or not np.isfinite(stereo).all():
        raise ValueError("Audio decode incomplete or nonfinite")
    native_mono = stereo.mean(axis=1)
    mono = torchaudio.functional.resample(torch.from_numpy(native_mono), 32000, 48000,
        lowpass_filter_width=6, rolloff=.99, resampling_method="sinc_interp_hann", beta=None).numpy()
    if len(mono) != math.ceil(len(native_mono) * 3 / 2) or not np.isfinite(mono).all():
        raise ValueError("Resampler sample count/values do not preserve the declared clock")
    if not float(np.abs(mono).max()) > 0:
        raise ValueError("Original control source unexpectedly silent")
    return rgb, mono, {"source_sha256": case["sha256"], "streams": streams,
        "source_pts": pts, "raw_rgb_sha256": hashlib.sha256(raw).hexdigest(),
        "native_stereo_pcm_sha256": hashlib.sha256(pcm).hexdigest(), "native_samples": len(native_mono),
        "resampled_mono_sha256": hashlib.sha256(mono.tobytes()).hexdigest(), "samples": len(mono),
        "downmix": "arithmetic channel mean", "resampling": "torchaudio sinc_interp_hann 32000 to 48000; width=6, rolloff=.99, beta=None",
        "gain_normalization": False}


def gates(rows):
    expected = {(c, k, 0) for c in ("A05", "A09") for k in CONDITIONS}
    expected |= {("A05", k, 1) for k in ("original", "delay750")}
    keys = [(r["case_id"], r["condition"], r["repeat"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Exact twenty-case scope required")
    by_key = dict(zip(keys, rows))
    for row in rows:
        if len(row["windows"]) != 3:
            raise ValueError("Three windows per case required")
        for window in row["windows"]:
            d = window["diagonal"]
            if len(d) != 50 or not all(type(x) in (int, float) and math.isfinite(x) for x in d):
                raise ValueError("Fifty finite scores per window required")
            if abs(sum(d) / 50 - window["score"]) > 1e-6:
                raise ValueError("Window score not supported by raw diagonal")
        if abs(sum(w["score"] for w in row["windows"]) / 3 - row["raw_score"]) > 1e-8:
            raise ValueError("Case score not supported by windows")
    base = by_key[("A05", "original", 0)]["raw_score"]
    margins = {c: base - by_key[("A05", c, 0)]["raw_score"] for c in ("delay750", "advance750", "tone440")}
    repeat_diffs = {}
    repeats = True
    for condition in ("original", "delay750"):
        first, second = [by_key[("A05", condition, i)] for i in (0, 1)]
        delta = max(abs(x - y) for a, b in zip(first["windows"], second["windows"])
                    for x, y in zip(a["diagonal"], b["diagonal"]))
        same_inputs = all(a["input_tensors"] == b["input_tensors"] for a, b in zip(first["windows"], second["windows"]))
        repeat_diffs[condition] = delta
        repeats &= same_inputs and delta <= 1e-5
    silence = all(not r["assessable"] and r["reported_score"] is None
                  for r in rows if r["condition"] == "muted")
    sensitivity = all(x > .0001 for x in margins.values())
    return {"technical_gate": True, "necessary_sensitivity_gate": sensitivity,
            "a05_original_minus_control": margins, "repeatability_gate": bool(repeats),
            "repeat_max_diagonal_difference": repeat_diffs, "harness_silence_abstention_gate": silence,
            "necessary_controls_pass": sensitivity and bool(repeats) and silence,
            "qualified_for_av_quality": False, "heldout_admitted": False}


def run(name):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError("A simple new run directory name is required")
    isolate()
    policy = read_json(POLICY)
    amendment = read_json(AMENDMENT)
    if digest(POLICY) != amendment["parent_policy_sha256"]:
        raise ValueError("Prospective parent policy changed")
    manifest = REPO / policy["manifest"]
    if digest(manifest) != policy["manifest_sha256"]:
        raise ValueError("Calibration manifest changed")
    scope = {r["case_id"]: r for r in read_json(manifest)["cases"] if r["case_id"] in policy["source_ids"]}
    if set(scope) != {"A05", "A09"}:
        raise ValueError("Wrong calibration source scope")
    target = ROOT / "runs" / name
    target.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), REPO / "scripts/h3_fate.py", REPO / "scripts/h3_fate_setup.py",
             REPO / "scripts/h3_local_av_evaluator.py", REPO / "scripts/h3_synchformer.py",
             REPO / "scripts/h3_local_av_setup.py", POLICY, AMENDMENT, ROOT / "loading-audit.json"]
    intent = {"utc": now(), "policy": policy, "amendment": amendment, "packages": inventory(),
              "artifact_sha256": {str(p.relative_to(REPO)): digest(p) for p in files}}
    save_new(target / "intent.json", intent)
    for p in files[:3]:
        shutil.copyfile(p, target / p.name)
    rows = []
    try:
        check_generation_queue()
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise RuntimeError("Require idle CUDA device with >=20 GiB free")
        torch.manual_seed(20260909)
        torch.set_num_threads(4)
        decoded = {case_id: decode_source(scope[case_id]) for case_id in policy["source_ids"]}
        for case_id, (_, _, source) in decoded.items():
            save_new(target / f"{case_id}-source.json", source)
        check_generation_queue()
        model, processor, loading = load_model()
        save_new(target / "loading.json", loading)
        model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        for case_id in policy["source_ids"]:
            rgb, original, source = decoded[case_id]
            conditions = [(c, 0) for c in CONDITIONS]
            if case_id == "A05":
                conditions += [(c, 1) for c in policy["repeat_conditions_on_A05"]]
            for condition, repeat in conditions:
                started = time.monotonic()
                wave = controlled_audio(original, condition)
                row = {"case_id": case_id, "condition": condition, "repeat": repeat, "windows": []}
                for window_id, (start, end) in enumerate(policy["window_seconds"]):
                    check_generation_queue()
                    video = torch.from_numpy(rgb[round(start * 24):round(end * 24)].copy()).permute(0, 3, 1, 2)
                    audio = wave[round(start * 48000):round(end * 48000)].copy()
                    inputs = processor(videos=video, audio=audio, sampling_rate=48000, padding=True, return_tensors="pt")
                    records = {k: tensor_record(v) for k, v in inputs.items()}
                    if (list(inputs["input_values"].shape) != [1, 1, 96000]
                            or list(inputs["pixel_values_videos"].shape) != [1, 48, 3, 336, 336]
                            or not bool(torch.all(inputs["padding_mask"] == 1))
                            or not all(v["finite"] for v in records.values())):
                        raise ValueError("Actual processor inputs fail shape/validity contract")
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**{k: v.to("cuda") for k, v in inputs.items()})
                    a, v = output.audio_frame_embeds.float(), output.video_frame_embeds.float()
                    if (list(a.shape) != [1, 50, 1024] or v.shape != a.shape
                            or not bool(torch.isfinite(a).all() and torch.isfinite(v).all())
                            or bool(torch.any(a.norm(dim=-1) == 0) or torch.any(v.norm(dim=-1) == 0))):
                        raise ValueError("Output embeddings fail shape/finite/norm contract")
                    a, v = F.normalize(a, dim=-1).cpu(), F.normalize(v, dim=-1).cpu()
                    diagonal = (a * v).sum(-1)[0]
                    stem = f"{case_id}-{condition}-{repeat}-w{window_id}"
                    path = target / (stem + ".safetensors")
                    save_file({"audio": a.contiguous(), "video": v.contiguous()}, path)
                    window = {"window_id": window_id, "seconds": [start, end],
                        "source_frame_indices": list(range(round(start * 24), round(end * 24))),
                        "source_audio_sample_range": [round(start * 48000), round(end * 48000)],
                        "input_tensors": records, "audio_peak": float(np.abs(audio).max()),
                        "alignment": model.get_base_model().last_alignment,
                        "diagonal": diagonal.tolist(), "score": float(diagonal.mean()),
                        "embeddings_file": path.name, "embeddings_sha256": digest(path)}
                    save_new(target / (stem + ".json"), window)
                    row["windows"].append(window)
                    del inputs, output, a, v
                row["raw_score"] = sum(w["score"] for w in row["windows"]) / 3
                row["assessable"] = any(w["audio_peak"] > 0 for w in row["windows"])
                row["reported_score"] = row["raw_score"] if row["assessable"] else None
                row["seconds_elapsed"] = time.monotonic() - started
                save_new(target / f"{case_id}-{condition}-{repeat}.json", row)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
        outcome = gates(rows)
        save_new(target / "summary.json", {"utc": now(), "status": "complete", "rows": rows, "gates": outcome,
            "intent_sha256": digest(target / "intent.json"), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "production_changed": False, "human_ratings_created": False})
        print(json.dumps(outcome), flush=True)
    except Exception:
        save_new(target / "failure.json", {"utc": now(), "completed_cases": len(rows), "traceback": traceback.format_exc(),
            "qualified_for_av_quality": False, "production_changed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    run(parser.parse_args().run_name)
