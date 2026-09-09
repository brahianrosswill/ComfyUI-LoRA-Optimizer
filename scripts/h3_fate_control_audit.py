"""CPU-only independent input/embedding audit; never imports the FATE runner/model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

try:
    from .h3_local_av_setup import REPO, digest, now, read_json, save_new
except ImportError:
    from h3_local_av_setup import REPO, digest, now, read_json, save_new

ROOT = REPO / ".h3-study-artifacts/20260909/fate"


def record(x):
    import torch
    x = x.detach().cpu().contiguous()
    return {"shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(torch.isfinite(x).all()),
            "sha256": hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}


def reconstruct_audio(mono, condition):
    import numpy as np
    if condition == "original": return mono.copy()
    if condition == "muted": return np.zeros_like(mono)
    if condition == "tone440":
        energy = np.r_[np.zeros(240), mono.astype(np.float64) ** 2, np.zeros(239)]
        integral = np.cumsum(np.r_[0., energy])
        rms = np.sqrt(np.maximum((integral[480:] - integral[:-480]) / 480, 0))
        tone = rms * np.sin(2 * np.pi * 440 * np.arange(mono.size) / 48000)
        tone *= math.sqrt(float(np.mean(mono.astype(np.float64) ** 2))) / math.sqrt(float(np.mean(tone ** 2)))
        return tone.astype(np.float32)
    offsets = {"delay750": 36000, "advance750": -36000, "delay125": 6000,
               "advance125": -6000, "delay80": 3840, "advance80": -3840}
    n = offsets[condition]
    if n > 0: return np.concatenate((np.zeros(n, dtype=np.float32), mono[:-n]))
    return np.concatenate((mono[-n:], np.zeros(-n, dtype=np.float32)))


def audit(name):
    import numpy as np
    import torch
    import torchaudio
    from torchvision.transforms.v2 import functional as VF
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    if Path(name).name != name or name.startswith("."):
        raise ValueError("Simple run name required")
    folder = ROOT / "runs" / name
    output = folder / "independent-audit.json"
    if output.exists(): raise FileExistsError(output)
    intent, summary = [read_json(folder / n) for n in ("intent.json", "summary.json")]
    assert summary["status"] == "complete"
    assert digest(folder / "intent.json") == summary["intent_sha256"]
    for path, sha in intent["artifact_sha256"].items():
        assert digest(REPO / path) == sha, path
    for p in folder.glob("h3_fate*.py"):
        assert digest(p) == intent["artifact_sha256"]["scripts/" + p.name]
    manifest = REPO / intent["policy"]["manifest"]
    assert digest(manifest) == intent["policy"]["manifest_sha256"]
    cases = {x["case_id"]: x for x in read_json(manifest)["cases"]}
    expected_keys = {(c, k, 0) for c in ("A05", "A09") for k in intent["policy"]["conditions"]}
    expected_keys |= {("A05", k, 1) for k in ("original", "delay750")}
    rows = summary["rows"]
    keys = [(r["case_id"], r["condition"], r["repeat"]) for r in rows]
    assert len(keys) == 20 and set(keys) == expected_keys
    checked = []
    for case_id in ("A05", "A09"):
        case, source = cases[case_id], read_json(folder / f"{case_id}-source.json")
        path = case["path"]
        assert digest(path) == source["source_sha256"] == case["sha256"]
        pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
            "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
        assert hashlib.sha256(pcm).hexdigest() == source["native_stereo_pcm_sha256"]
        native = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2).mean(axis=1)
        mono = torchaudio.functional.resample(torch.from_numpy(native), 32000, 48000,
            lowpass_filter_width=6, rolloff=.99, resampling_method="sinc_interp_hann", beta=None).numpy()
        assert len(native) == source["native_samples"]
        assert len(mono) == source["samples"] == math.ceil(len(native) * 1.5)
        assert hashlib.sha256(mono.tobytes()).hexdigest() == source["resampled_mono_sha256"]
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
            "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        assert hashlib.sha256(raw).hexdigest() == source["raw_rgb_sha256"]
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(124, 384, 640, 3)
        prepared_video = []
        for start, end in intent["policy"]["window_seconds"]:
            batch = torch.from_numpy(rgb[round(start * 24):round(end * 24)].copy()).permute(0, 3, 1, 2)
            resized = VF.resize(batch, [336, 336], interpolation=VF.InterpolationMode.BILINEAR, antialias=True)
            prepared_video.append(((resized.float() - 127.5) / 127.5).unsqueeze(0))
        for row in [r for r in rows if r["case_id"] == case_id]:
            c, repeat = row["condition"], row["repeat"]
            assert row == read_json(folder / f"{case_id}-{c}-{repeat}.json")
            wave = reconstruct_audio(mono, c)
            window_scores = []
            for j, (start, end) in enumerate(intent["policy"]["window_seconds"]):
                w = row["windows"][j]
                assert w == read_json(folder / f"{case_id}-{c}-{repeat}-w{j}.json")
                assert w["seconds"] == [start, end] and w["window_id"] == j
                assert w["source_frame_indices"] == list(range(round(start * 24), round(end * 24)))
                assert w["source_audio_sample_range"] == [round(start * 48000), round(end * 48000)]
                audio = wave[round(start * 48000):round(end * 48000)]
                expected = {"pixel_values_videos": record(prepared_video[j]),
                            "input_values": record(torch.from_numpy(audio.copy()).reshape(1, 1, -1)),
                            "padding_mask": record(torch.ones(1, 96000, dtype=torch.int32))}
                assert expected == w["input_tensors"], (case_id, c, j, "actual model inputs")
                assert float(np.abs(audio).max()) == w["audio_peak"]
                align = w["alignment"]
                assert align["video_shape"] == [1, 48, 768] and align["audio_shape"] == [1, 50, 768]
                assert align["all_masks_valid"] and align["unchanged_upstream_alignment"]
                assert align["video_index_for_audio_feature"] == [i * 48 // 50 for i in range(50)]
                assert Path(w["embeddings_file"]).name == w["embeddings_file"]
                embedding_path = folder / w["embeddings_file"]
                assert digest(embedding_path) == w["embeddings_sha256"]
                data = load_file(embedding_path)
                assert set(data) == {"audio", "video"}
                for t in data.values():
                    assert t.shape == (1, 50, 1024) and bool(torch.isfinite(t).all())
                    assert float((t.norm(dim=-1) - 1).abs().max()) < 1e-5
                # Independent double-precision contraction of saved normalized embeddings.
                diagonal = np.einsum("btd,btd->bt", data["audio"].numpy().astype(np.float64),
                                     data["video"].numpy().astype(np.float64))[0]
                assert float(np.max(np.abs(diagonal - w["diagonal"]))) < 1e-6
                assert abs(float(diagonal.mean()) - w["score"]) < 1e-6
                window_scores.append(float(diagonal.mean()))
            score = sum(window_scores) / 3
            assert abs(score - row["raw_score"]) < 1e-6
            assert row["assessable"] == (c != "muted")
            assert row["reported_score"] == (row["raw_score"] if c != "muted" else None)
            checked.append({"case_id": case_id, "condition": c, "repeat": repeat, "reconstructed_score": score})
    by_key = dict(zip(keys, rows))
    margins = {c: by_key[("A05", "original", 0)]["raw_score"] - by_key[("A05", c, 0)]["raw_score"]
               for c in ("delay750", "advance750", "tone440")}
    assert margins == summary["gates"]["a05_original_minus_control"]
    assert summary["gates"]["necessary_sensitivity_gate"] == all(x > .0001 for x in margins.values())
    for c in ("original", "delay750"):
        a, b = [by_key[("A05", c, r)] for r in (0, 1)]
        delta = max(abs(x-y) for u,v in zip(a["windows"],b["windows"]) for x,y in zip(u["diagonal"],v["diagonal"]))
        assert delta == summary["gates"]["repeat_max_diagonal_difference"][c] == 0
    nominal_errors = [i * 48 // 50 / 24 - i / 25 for i in range(50)]
    save_new(output, {"utc": now(), "auditor_sha256": digest(__file__), "summary_sha256": digest(folder / "summary.json"),
        "independent_input_reconstruction": True, "all_cases": checked, "windows_checked": 60,
        "all_180_input_tensor_records_exact": True, "saved_embedding_scores_reproduced": True,
        "all_required_controls_and_repeats_reproduced": True,
        "nominal_video_minus_audio_feature_grid_range_seconds": [min(nominal_errors), max(nominal_errors)],
        "clock_limit": "These are feature-grid coordinates, not independently observed event onsets; embeddings contain temporal context.",
        "gpu_used": False, "model_forward": False, "qualified_for_quality": False})
    print(json.dumps({"audit": str(output), "sha256": digest(output), "cases": len(checked), "windows": 60}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    audit(parser.parse_args().run_name)
