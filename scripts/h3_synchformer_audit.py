"""Reconstruct Synchformer model inputs from original clips without GPU inference."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from .h3_synchformer import (CFG, CKPT, POLICY, REPO, ROOT, audio_control, control_gates,
        decode_audio, decode_video, digest, isolate, now, read_json, save_new, tensor_record)
except ImportError:
    from h3_synchformer import (CFG, CKPT, POLICY, REPO, ROOT, audio_control, control_gates,
        decode_audio, decode_video, digest, isolate, now, read_json, save_new, tensor_record)


def audit(name):
    import numpy as np
    import torch
    import torchaudio
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Simple run name required")
    isolate()
    torch.set_num_threads(4)
    run = ROOT / "runs" / name
    start, complete = read_json(run / "start.json"), read_json(run / "complete.json")
    policy = read_json(POLICY)
    manifest_path = REPO / policy["manifest"]
    manifest = read_json(manifest_path)
    pins = {str(POLICY): start["policy_sha256"], str(manifest_path): start["manifest_sha256"],
            str(ROOT / "setup-receipt.json"): start["receipt_sha256"],
            str(REPO / "scripts/h3_synchformer.py"): start["runner_sha256"],
            str(REPO / "scripts/h3_local_av_evaluator.py"): start["audio_helper_sha256"]}
    for path, sha in pins.items():
        if digest(path) != sha:
            raise ValueError(f"Changed pinned artifact: {path}")
    receipt = read_json(ROOT / "setup-receipt.json")
    pins.update({str(CFG): receipt["config_sha256"], str(CKPT): receipt["checkpoint_sha256"]})
    for path in (CFG, CKPT):
        if digest(path) != pins[str(path)]:
            raise ValueError("Changed model asset")
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400,
        hop_length=160, n_fft=1024, n_mels=128)
    by_case = {c["case_id"]: c for c in manifest["cases"]}
    reconstructed = {}
    rows = []
    for case_id, condition, repeat in start["jobs"]:
        case = by_case[case_id]
        if case_id not in reconstructed:
            if digest(case["path"]) != case["sha256"]:
                raise ValueError("Original media changed")
            pins[case["path"]] = case["sha256"]
            video, vi = decode_video(case["path"])
            if vi["source_frame_indices"] != [round(i * 24 / 25) for i in range(125)]:
                raise ValueError("PTS mapping does not match independently calculated 24-to-25-fps mapping")
            audio, ai = decode_audio(case["path"])
            reconstructed[case_id] = video, vi, audio, ai
        video, vi, audio, ai = reconstructed[case_id]
        # Reconstruct from declared ranges, without invoking Synchformer's Compose.
        controlled = audio_control(audio, condition)
        v = torch.stack([video[2 + i * 8:18 + i * 8, :, 16:240, 101:325] for i in range(14)])
        v = ((v.half().div(255.) - .5) / .5).unsqueeze(0)
        a = torch.from_numpy(np.stack([controlled[1280 + i * 5120:11520 + i * 5120] for i in range(14)]))
        a = torch.log(mel(a) + 1e-6)
        if a.shape != (14, 128, 65):
            raise ValueError("Unexpected raw spectrogram shape")
        a = torch.nn.functional.pad(a, (0, 1), value=0.)
        a = ((a + 4.2677393) / (2 * 4.5689974)).unsqueeze(0).unsqueeze(2)
        stem = f"{case_id}-{condition}-r{repeat}"
        path, ipath = run / (stem + ".json"), run / (stem + ".input.json")
        row, before = read_json(path), read_json(ipath)
        if any(row[k] != value for k, value in before.items()):
            raise ValueError("Input record altered after inference")
        info = row["input"]
        if (info["video_tensor"] != tensor_record(v) or info["audio_tensor"] != tensor_record(a)
                or info["waveform_sha256"] != hashlib.sha256(controlled.tobytes()).hexdigest()
                or info["audio_peak"] != float(np.max(np.abs(controlled)))
                or info["video"] != vi or info["audio"] != ai
                or info["segment_video_ranges"] != [[2 + i * 8, 18 + i * 8] for i in range(14)]
                or info["segment_audio_ranges"] != [[1280 + i * 5120, 11520 + i * 5120] for i in range(14)]):
            raise ValueError(f"Reconstructed model input mismatch: {stem}")
        pins[str(path)] = digest(path)
        pins[str(ipath)] = digest(ipath)
        rows.append(row)
    gates = control_gates(rows)
    if gates != complete["gates"] or len(rows) != complete["cases"]:
        raise ValueError("Completion summary does not match raw evidence")
    for path in (run / "start.json", run / "complete.json", Path(__file__).resolve()):
        pins[str(path)] = digest(path)
    report = {"completed_at_utc": now(), "cases": len(rows), "gates": gates,
              "independently_reconstructed_model_inputs": True, "gpu_inference_rerun": False,
              "original_media_unchanged": True, "artifact_sha256": pins}
    save_new(run / "control-audit.json", report)
    print({k: v for k, v in report.items() if k != "artifact_sha256"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    audit(parser.parse_args().run_name)
