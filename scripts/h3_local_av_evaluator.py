"""Calibration-only local AV judge harness; never imports or changes the optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from urllib.request import urlopen

try:
    from .h3_local_av_setup import REPO, ROOT, POLICY, digest, inventory, isolate, now, read_json, save_new
except ImportError:
    from h3_local_av_setup import REPO, ROOT, POLICY, digest, inventory, isolate, now, read_json, save_new

DIMENSIONS = ("action", "temporal", "appearance", "audio", "synchronization", "overall")
RATINGS = REPO / "docs/research/data/2026-09-08-h3-av2-ratings-seed03.json"
CALIBRATION = REPO / "docs/research/data/2026-09-08-h3-av2-calibration-seed03.json"
AMENDMENT = REPO / "docs/research/data/2026-09-09-h3-local-av-evaluator-amendment-02.json"
MANIFEST = ROOT / "calibration-inputs-02.json"
SYSTEM = "You are a careful audiovisual observer. Report only evidence present in the supplied media. Express uncertainty instead of inventing sounds, contacts or precise timing. Return JSON only."
INSTRUCTION = """Inspect the complete audio and sampled video. First describe what is actually audible and visible, independent of the intended scene. Never infer that a sound exists merely because a visible action would usually produce it. A quiet or silent recording is possible. Do not invent speech, music or exact timing. Video is sparsely sampled, so uncertain contact timing/counts must remain uncertain.

The intended scene is a single stationary side view of an adult boxer in a grey tracksuit and red gloves at a heavy bag in an empty gym. At normal speed: two distinct straight punches, return both gloves to guard after each, then one step back and stop. The bag should react to contacts. Intended audio: two dull synchronized impacts, shoe squeaks, chain rattling and quiet gym ambience; no speech or music. This description is a target, NOT evidence that those events happened.

Use the original review's ordinal scale: 0 fails badly, 1 major faults, 2 mixed, 3 good with minor faults, 4 convincing. Scores: action=requested order/contact/response/ending; temporal=stable anatomy and continuous motion; appearance=coherent subjects/texture/lighting; audio=plausible ambience/impacts without unwanted speech/music; synchronization=audible impacts match visible contacts; overall=your AV preference, not a computed average. Use null for any unassessable dimension; do not turn missing evidence into a zero or a perfect score.

Return exactly one JSON object with these keys:
"audible_sound": boolean or null,
"speech_present": boolean or null,
"music_present": boolean or null,
"visible_contact_count": nonnegative integer or null,
"scores": object with action, temporal, appearance, audio, synchronization, overall (each integer 0..4 or null),
"evidence": object with those same six keys, each at most twelve words grounded in what you observed,
"sound_events": at most THREE summary objects with "start_seconds", "end_seconds" (number or null) and "description" (at most eight words),
"uncertainty": at most thirty words explaining limitations.
Do not produce a detailed event timeline. Use null times when timing is uncertain. If silent, sound_events must be an empty list. Do not repeat event summaries. Keep the entire response under 300 words."""


def stopping_ids(tokenizer):
    """Do not inherit missing EOS/pad IDs from the top-level Omni generation file."""
    ids = {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    if any(type(value) is not int or value < 0 for value in ids.values()):
        raise ValueError("Explicit valid tokenizer EOS and pad IDs required")
    return ids


def parse_response_json(text):
    # Fences are presentation only; trailing prose/NaN/duplicate keys are errors.
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=bad_constant)


def validate_response(text):
    value = parse_response_json(text)
    expected = {"audible_sound", "speech_present", "music_present", "sound_events",
                "visible_contact_count", "scores", "evidence", "uncertainty"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected response schema")
    for key in ("audible_sound", "speech_present", "music_present"):
        if value[key] is not None and type(value[key]) is not bool:
            raise ValueError("Audio observations must be boolean or null")
    count = value["visible_contact_count"]
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("Invalid contact count")
    if not isinstance(value["scores"], dict) or set(value["scores"]) != set(DIMENSIONS):
        raise ValueError("All six scores required")
    for score in value["scores"].values():
        if score is not None and (type(score) is not int or not 0 <= score <= 4):
            raise ValueError("Scores must be integers 0..4 or null")
    if not isinstance(value["evidence"], dict) or set(value["evidence"]) != set(DIMENSIONS):
        raise ValueError("All six evidence fields required")
    if any(not isinstance(s, str) or not s.strip() for s in value["evidence"].values()):
        raise ValueError("Nonempty evidence required")
    if not isinstance(value["uncertainty"], str) or not isinstance(value["sound_events"], list):
        raise ValueError("Invalid uncertainty/events")
    for event in value["sound_events"]:
        if not isinstance(event, dict) or set(event) != {"start_seconds", "end_seconds", "description"}:
            raise ValueError("Invalid sound event")
        if not isinstance(event["description"], str) or not event["description"].strip():
            raise ValueError("Sound description required")
        for key in ("start_seconds", "end_seconds"):
            t = event[key]
            if t is not None and (type(t) not in (int, float) or not math.isfinite(t) or t < 0):
                raise ValueError("Invalid sound time")
        if all(event[k] is not None for k in ("start_seconds", "end_seconds")):
            if event["end_seconds"] < event["start_seconds"]:
                raise ValueError("Reversed event time")
    return value


def validate_flat_response(text):
    value = parse_response_json(text)
    text_keys = ("observed_video", "observed_audio", "sync_evidence", "uncertainty")
    bool_keys = ("audible_sound", "speech_present", "music_present")
    score_keys = tuple(k + "_score" for k in DIMENSIONS)
    if not isinstance(value, dict) or set(value) != set(text_keys + bool_keys + score_keys):
        raise ValueError("Unexpected flat response schema")
    if any(not isinstance(value[k], str) or not value[k].strip() for k in text_keys):
        raise ValueError("Nonempty observation/evidence required")
    for k in bool_keys:
        if value[k] is not None and type(value[k]) is not bool:
            raise ValueError("Boolean observation or null required")
    for k in score_keys:
        if value[k] is not None and (type(value[k]) is not int or not 0 <= value[k] <= 4):
            raise ValueError("Ordinal score must be an integer 0..4 or null")
    return value


def evaluation_profile(name):
    if name == "legacy":
        return AMENDMENT, MANIFEST, SYSTEM, INSTRUCTION, validate_response
    if name != "joint_flat":
        raise ValueError("Unknown evaluator profile")
    amendment = REPO / "docs/research/data/2026-09-09-h3-local-av-evaluator-amendment-06.json"
    data = read_json(amendment)
    return (amendment, ROOT / "calibration-inputs-flat-01.json", data["system"],
            data["instruction"], validate_flat_response)


def effective_video_fps(indices, source_fps):
    """Align the scalar model time grid to actual sampled frame endpoints.

    Qwen's reader returns N/duration, but selects N linspace endpoint frames.
    Using (N-1)/sampled_span avoids cumulative AV timing drift. Rounding error
    of individual source indices is measured separately, not claimed exact.
    """
    if not math.isfinite(source_fps) or source_fps <= 0 or len(indices) < 2:
        raise ValueError("Invalid source timing")
    if indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Expected strictly increasing whole-clip frame indices starting at zero")
    return (len(indices) - 1) * source_fps / (indices[-1] - indices[0])


def prepare(profile="legacy"):
    amendment_path, manifest_path, system, instruction, _ = evaluation_profile(profile)
    policy = read_json(POLICY)
    if digest(RATINGS) != policy["human_ratings_sha256"]:
        raise ValueError("Calibration ratings changed")
    data, labels = read_json(CALIBRATION), read_json(RATINGS)
    if data["seed"] != 2026090803 or data["stage"] != "calibration":
        raise ValueError("Only already-rated seed03 calibration is allowed")
    renders = {r["run"]: r for r in data["renders"]}
    rows = {}
    for label in labels["ratings"]:
        if label["seed"] != 2026090803 or label["stage"] != "calibration":
            raise ValueError("Unrated/held-out source rejected")
        source = renders[label["job_id"]]
        if source["video_sha256"] != label["original_sha256"]:
            raise ValueError("Calibration media/ratings mismatch")
        path = Path(source["video"]).resolve()
        if not path.is_relative_to(Path("/media/unraid/comfyui/output")):
            raise ValueError("Unexpected calibration source path")
        if label["job_id"] not in rows:
            if digest(path) != source["video_sha256"]:
                raise ValueError("Calibration clip changed")
            rows[label["job_id"]] = {"source_job": label["job_id"], "path": str(path),
                                      "sha256": source["video_sha256"]}
    if len(rows) != 12 or len(labels["ratings"]) != 14:
        raise ValueError("Unexpected calibration scope")
    cases = [dict(row, case_id=f"A{i:02d}") for i, row in enumerate(
        sorted(rows.values(), key=lambda r: r["sha256"]), 1)]
    amendment = read_json(amendment_path)
    if amendment["parent_policy_sha256"] != digest(POLICY):
        raise ValueError("Unexpected parent policy")
    record = {"created_at_utc": now(), "stage": "calibration_only", "policy_sha256": digest(POLICY),
              "amendment_sha256": digest(amendment_path), "profile": profile,
              "ratings_sha256": digest(RATINGS), "calibration_sha256": digest(CALIBRATION),
              "cases": cases, "system": system, "instruction": instruction,
              "model_sees_labels": False, "model_sees_paths": False}
    save_new(manifest_path, record)
    print(json.dumps({"prepared": len(cases), "path": str(manifest_path)}), flush=True)


def audio_control(samples, condition, sample_rate=16000):
    import numpy as np
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Finite nonempty mono waveform required")
    if condition == "original":
        return samples.copy()
    result = np.zeros_like(samples)
    if condition == "muted":
        return result
    if condition != "delay750":
        raise ValueError("Unknown control")
    shift = round(sample_rate * 0.75)
    if shift >= len(samples):
        raise ValueError("Clip too short for delay control")
    result[shift:] = samples[:-shift]
    return result


def decode_audio(path):
    import numpy as np
    import librosa
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], text=True))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if abs(float(stream.get("start_time", 0)) - float(video.get("start_time", 0))) > 1 / 16000:
        raise ValueError("Unexpected source A/V start offset; must handle explicitly")
    channels, sr = int(stream["channels"]), int(stream["sample_rate"])
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    native = np.frombuffer(raw, dtype="<f4").reshape(-1, channels)
    mono = native.mean(axis=1)
    samples = librosa.resample(mono, orig_sr=sr, target_sr=16000, res_type="soxr_hq").astype(np.float32)
    metadata = {"source_sample_rate": sr, "source_channels": channels,
                "source_samples_per_channel": len(native), "input_sample_rate": 16000,
                "input_samples": len(samples), "downmix": "arithmetic channel mean",
                "resample": "librosa/soxr_hq", "gain_normalization": False,
                "native_pcm_sha256": hashlib.sha256(raw).hexdigest(),
                "source_streams": probe["streams"]}
    return samples, metadata


def check_generation_queue():
    """Fail closed if the known existing render session has work or is unknown."""
    with urlopen("http://127.0.0.1:8189/queue", timeout=3) as reply:
        queue = json.load(reply)
    if not isinstance(queue, dict) or any(not isinstance(queue.get(k), list)
                                          for k in ("queue_running", "queue_pending")):
        raise ValueError("Cannot verify existing ComfyUI queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Existing ComfyUI work detected; stop evaluator without interrupting it")
    return queue


def run(mode, run_name, profile="legacy"):
    amendment_path, manifest_path, system, instruction, validate = evaluation_profile(profile)
    isolate(offline=True)
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from qwen_omni_utils import fetch_video
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a simple new run directory name")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    policy, manifest = read_json(POLICY), read_json(manifest_path)
    amendment = read_json(amendment_path)
    receipt = read_json(ROOT / "setup-receipt.json")
    if manifest["policy_sha256"] != digest(POLICY) or receipt["policy_sha256"] != digest(POLICY):
        raise ValueError("Frozen policy mismatch")
    if manifest["amendment_sha256"] != digest(amendment_path) or amendment["parent_policy_sha256"] != digest(POLICY):
        raise ValueError("Frozen calibration amendment mismatch")
    if manifest["instruction"] != instruction or manifest["system"] != system:
        raise ValueError("Frozen instruction mismatch")
    model_dir = Path(receipt["model_dir"])
    for asset in receipt["files"]:
        stat = (model_dir / asset["name"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (asset["bytes"], asset["mtime_ns"]):
            raise ValueError("Previously hash-verified model asset changed")
    cases = manifest["cases"]
    controls = [c for c in cases if c["source_job"] in policy["control_sources"]]
    best = next(c for c in controls if c["source_job"].endswith("-combat"))
    if mode == "smoke":
        jobs = [(best, "original", 0)]
    elif mode == "controls":
        jobs = [(c, k, 0) for c in controls for k in ("original", "muted", "delay750")]
        jobs += [(best, k, 1) for k in ("original", "muted")]
    else:
        jobs = [(c, "original", 0) for c in cases]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute CPU")
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    free, total = torch.cuda.mem_get_info()
    if free < 24 * 1024**3:
        raise RuntimeError("Less than 24 GiB free; defer instead of evicting another GPU job")
    start_record = {"started_at_utc": now(), "mode": mode, "policy_sha256": digest(POLICY),
                    "manifest_sha256": digest(manifest_path), "amendment_sha256": digest(amendment_path), "profile": profile,
                    "setup_receipt_sha256": digest(ROOT / "setup-receipt.json"),
                    "helper_sha256": digest(__file__), "setup_helper_sha256": digest(REPO / "scripts/h3_local_av_setup.py"),
                    "generation": amendment["generation"], "packages": inventory(),
                    "gpu": torch.cuda.get_device_name(), "free_bytes_before": free,
                    "total_bytes": total, "system": system, "instruction": instruction,
                    "jobs": [{"case_id": c["case_id"], "condition": k, "repeat": r} for c, k, r in jobs],
                    "processor_timing": "Endpoint-derived effective fps from actual frame indices; residual rounding error recorded.",
                    "controls": "In-memory float32 audio at 16 kHz. Original video tensors retained; source media untouched.",
                    "media_uploaded": False, "human_labels_in_prompt": False}
    start_record["queue_before_loading"] = check_generation_queue()
    save_new(run_dir / "start.json", start_record)
    print(json.dumps({"event": "loading_model", "run": run_name}), flush=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
        use_safetensors=True).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    torch.cuda.synchronize()
    print(json.dumps({"event": "model_loaded", "allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
    for case, condition, repeat in jobs:
        ident = f"{case['case_id']}-{condition}-r{repeat}"
        started = time.perf_counter()
        result = {"case_id": case["case_id"], "condition": condition, "repeat": repeat,
                  "source_sha256": case["sha256"], "started_at_utc": now(), "human_rating": False}
        try:
            result["queue_before_case"] = check_generation_queue()
            path = Path(case["path"])
            if digest(path) != case["sha256"]:
                raise ValueError("Calibration source changed")
            media = {"type": "video", "video": str(path), "fps": 4.0,
                     "min_pixels": 100352, "max_pixels": 200704}
            (video, metadata), reader_fps = fetch_video(media, return_video_sample_fps=True, return_video_metadata=True)
            indices = metadata["frames_indices"].tolist()
            effective_fps = effective_video_fps(indices, float(metadata["fps"]))
            timing_error = max(abs(index / metadata["fps"] - i / effective_fps) for i, index in enumerate(indices))
            samples, audio_info = decode_audio(path)
            waveform = audio_control(samples, condition)
            conversation = [{"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": [media, {"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            if str(path) in text or case["source_job"] in text:
                raise ValueError("Model prompt leaks source identity")
            inputs = processor(text=text, audio=[waveform], videos=[video],
                return_tensors="pt", padding=True,
                videos_kwargs={"fps": effective_fps, "use_audio_in_video": True,
                               "do_sample_frames": False, "min_pixels": 100352, "max_pixels": 200704})
            required = {"input_features", "feature_attention_mask", "pixel_values_videos", "video_grid_thw"}
            if not required.issubset(inputs) or any(inputs[k].numel() == 0 for k in required):
                raise ValueError("Audio/video inputs are missing")
            result["inputs"] = {"video_shape": list(video.shape), "frame_indices": indices,
                "video_tensor_sha256": hashlib.sha256(video.contiguous().numpy().tobytes()).hexdigest(),
                "source_fps": metadata["fps"], "reader_reported_fps": reader_fps,
                "model_effective_fps": effective_fps, "max_grid_rounding_error_seconds": timing_error,
                "audio": audio_info, "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                "waveform_peak": float(np.abs(waveform).max()), "waveform_rms": float(np.sqrt(np.mean(waveform**2))),
                "tensor_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
                "audio_features_sha256": hashlib.sha256(inputs["input_features"].contiguous().numpy().tobytes()).hexdigest(),
                "video_features_input_sha256": hashlib.sha256(inputs["pixel_values_videos"].contiguous().numpy().tobytes()).hexdigest(),
                "video_second_per_grid": inputs["video_second_per_grid"].tolist(),
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
            save_new(run_dir / f"{ident}.input.json", result)
            inputs = inputs.to(model.device).to(model.dtype)
            # Qwen stores RoPE deltas as model state; fresh independent conversations.
            model.rope_deltas = None
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, use_audio_in_video=True, use_cache=True,
                                        **stopping_ids(processor.tokenizer),
                                        **amendment["generation"])
            torch.cuda.synchronize()
            new_ids = output[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result.update(raw_response=raw, generated_tokens=new_ids.shape[1],
                          generated_token_ids=new_ids[0].tolist(),
                          eos_token_id=processor.tokenizer.eos_token_id,
                          pad_token_id=processor.tokenizer.pad_token_id,
                          inference_seconds=time.perf_counter() - infer_start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            try:
                result["response"] = validate(raw)
                result["status"] = "valid_response"
            except (ValueError, TypeError) as error:
                result["status"] = "invalid_response"
                result["validation_error"] = str(error)
            del inputs, output, new_ids
        except Exception:
            result["status"] = "runtime_failure"
            result["error"] = traceback.format_exc()
            result["elapsed_seconds"] = time.perf_counter() - started
            save_new(run_dir / f"{ident}.json", result)
            raise
        result["elapsed_seconds"] = time.perf_counter() - started
        save_new(run_dir / f"{ident}.json", result)
        print(json.dumps({"event": "case_complete", "case": ident, "status": result["status"],
                          "seconds": result["elapsed_seconds"]}), flush=True)
    save_new(run_dir / "complete.json", {"completed_at_utc": now(), "cases": len(jobs),
                                         "model_validated_as_judge": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "controls", "calibration"))
    parser.add_argument("--run-name")
    parser.add_argument("--profile", choices=("legacy", "joint_flat"), default="legacy")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.profile)
    elif not args.run_name:
        parser.error("--run-name is required for inference")
    else:
        run(args.action, args.run_name, args.profile)
