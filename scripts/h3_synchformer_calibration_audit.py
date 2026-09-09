"""Independent CPU input reconstruction for the twelve rated Synchformer cases."""
from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from . import h3_synchformer as s
    from .h3_synchformer_calibration import POLICY, RATINGS, RUN, pairwise, verify
except ImportError:
    import h3_synchformer as s
    from h3_synchformer_calibration import POLICY, RATINGS, RUN, pairwise, verify


def main():
    import numpy as np
    import torch
    import torchaudio
    s.isolate()
    torch.set_num_threads(4)
    cases = verify()
    start = s.read_json(RUN / "start.json")
    comparison = s.read_json(RUN / "human-comparison.json")
    pins = {str(POLICY): start["policy_sha256"],
            str(s.REPO / "scripts/h3_synchformer_calibration.py"): start["runner_sha256"],
            str(s.REPO / "scripts/h3_synchformer.py"): start["input_helper_sha256"],
            str(s.REPO / "scripts/h3_local_av_evaluator.py"): start["audio_helper_sha256"],
            str(s.ROOT / "setup-receipt.json"): start["receipt_sha256"],
            **comparison["artifact_sha256"]}
    for path, sha in pins.items():
        if s.digest(path) != sha:
            raise ValueError(f"Changed pinned artifact: {path}")
    if start["cases"] != cases:
        raise ValueError("Run scope differs from frozen source manifest")
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400,
        hop_length=160, n_fft=1024, n_mels=128)
    predictions = {}
    repeated = {}
    for case in cases:
        if s.digest(case["path"]) != case["sha256"]:
            raise ValueError("Source media changed")
        pins[case["path"]] = case["sha256"]
        video, vi = s.decode_video(case["path"])
        audio, ai = s.decode_audio(case["path"])
        if vi["source_frame_indices"] != [round(i * 24 / 25) for i in range(125)]:
            raise ValueError("Independent time mapping mismatch")
        v = torch.stack([video[2 + 8 * i:18 + 8 * i, :, 16:240, 101:325] for i in range(14)])
        v = ((v.half().div(255.) - .5) / .5).unsqueeze(0)
        a = torch.from_numpy(np.stack([audio[1280 + 5120 * i:11520 + 5120 * i] for i in range(14)]))
        a = torch.log(mel(a) + 1e-6)
        if a.shape != (14, 128, 65):
            raise ValueError("Unexpected unpadded spectrogram")
        a = (torch.nn.functional.pad(a, (0, 1), value=0.) + 4.2677393) / (2 * 4.5689974)
        a = a.unsqueeze(0).unsqueeze(2)
        path, ipath = RUN / (case["case_id"] + ".json"), RUN / (case["case_id"] + ".input.json")
        row, before = s.read_json(path), s.read_json(ipath)
        info = row["input"]
        if (any(row[k] != value for k, value in before.items()) or row["status"] != "ok"
                or row["source_sha256"] != case["sha256"] or info["audio"] != ai or info["video"] != vi
                or info["video_tensor"] != s.tensor_record(v) or info["audio_tensor"] != s.tensor_record(a)
                or info["waveform_sha256"] != hashlib.sha256(audio.tobytes()).hexdigest()
                or info["audio_peak"] != float(np.max(np.abs(audio)))):
            raise ValueError(f"Actual model input reconstruction failed: {case['case_id']}")
        d = s.distribution(row["prediction"]["logits"], info["audio_peak"] == 0)
        if row["prediction"] != d:
            raise ValueError("Raw logits differ from stored prediction")
        predictions[case["sha256"]] = d
        if case["case_id"] in ("A05", "A09"):
            control_path = s.ROOT / "runs/controls-01" / (case["case_id"] + "-original-r0.json")
            control = s.read_json(control_path)
            repeated[case["case_id"]] = d == control["prediction"] and all(
                info[k] == control["input"][k] for k in info)
            pins[str(control_path)] = s.digest(control_path)
    reconstructed_rows = []
    for label in s.read_json(RATINGS)["ratings"]:
        d = predictions[label["original_sha256"]]
        reconstructed_rows.append({"pair": label["pair"], "variant": label["variant"], "blind_id": label["blind_id"],
            "source_sha256": label["original_sha256"], "human_sync": label["scores"]["synchronization"],
            "near_zero_mass": d["raw_near_zero_mass"] if d["assessable"] else None,
            "negative_abs_offset": -abs(d["reported_offset_seconds"]) if d["assessable"] else None,
            "modal_offset": d["reported_offset_seconds"]})
    comparisons = {pair: {metric: pairwise([r for r in reconstructed_rows if r["pair"] == pair], metric)
        for metric in ("near_zero_mass", "negative_abs_offset")} for pair in {r["pair"] for r in reconstructed_rows}}
    if reconstructed_rows != comparison["rows"] or comparisons != comparison["comparisons"]:
        raise ValueError("Human-label mapping or comparison changed")
    for p in (RUN / "start.json", RUN / "complete.json", RUN / "human-comparison.json", Path(__file__).resolve()):
        pins[str(p)] = s.digest(p)
    result = {"completed_at_utc": s.now(), "cases": len(cases), "human_entries": len(reconstructed_rows),
        "reconstructed_model_inputs": True, "control_original_reproduction": repeated,
        "gpu_inference_rerun": False, "comparison_reproduced": True, "artifact_sha256": pins,
        "qualified_for_av_quality": False, "qualified_for_heldout": False}
    s.save_new(RUN / "calibration-audit.json", result)
    print({k: v for k, v in result.items() if k != "artifact_sha256"})


if __name__ == "__main__":
    main()
