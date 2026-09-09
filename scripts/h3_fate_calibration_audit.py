"""Independent CPU reconstruction of the twelve-original FATE calibration.

Does not import the model, inference runner, or comparison implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

try:
    from .h3_local_av_setup import REPO, digest, now, read_json, save_new
    from .h3_fate_control_audit import record
except ImportError:
    from h3_local_av_setup import REPO, digest, now, read_json, save_new
    from h3_fate_control_audit import record

ROOT = REPO / ".h3-study-artifacts/20260909/fate"
RUN = ROOT / "runs/rated-calibration-01"


def audit():
    import numpy as np
    import torch
    import torchaudio
    from torchvision.transforms.v2 import functional as VF
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    output = RUN / "independent-audit.json"
    if output.exists(): raise FileExistsError(output)
    intent, summary, comparison = [read_json(RUN / n) for n in ("intent.json", "summary.json", "human-comparison.json")]
    assert summary["status"] == "complete"
    assert digest(RUN / "intent.json") == summary["intent_sha256"]
    assert digest(RUN / "summary.json") == comparison["summary_sha256"]
    for path, sha in intent["artifact_sha256"].items():
        assert digest(REPO / path) == sha, path
    for p in RUN.glob("*.py"):
        assert digest(p) == intent["artifact_sha256"]["scripts/" + p.name]
    manifest = REPO / intent["policy"]["manifest"]
    assert digest(manifest) == intent["policy"]["manifest_sha256"]
    cases = read_json(manifest)["cases"]
    assert cases == intent["cases"]
    assert [x["case_id"] for x in cases] == [f"A{i:02d}" for i in range(1, 13)]
    rows = summary["rows"]
    assert len(rows) == 12 and [x["case_id"] for x in rows] == [c["case_id"] for c in cases]
    source_scores, checked, anchors = {}, [], {}
    for case, row in zip(cases, rows):
        cid, path = case["case_id"], case["path"]
        source = read_json(RUN / f"{cid}-source.json")
        assert row == read_json(RUN / f"{cid}.json")
        assert digest(path) == source["source_sha256"] == row["source_sha256"] == case["sha256"]
        streams = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path], text=True))["streams"]
        assert streams == source["streams"]
        audio_stream = [s for s in streams if s["codec_type"] == "audio"]
        video_stream = [s for s in streams if s["codec_type"] == "video"]
        assert len(audio_stream) == len(video_stream) == 1
        assert int(audio_stream[0]["sample_rate"]) == 32000 and audio_stream[0]["channels"] == 2
        assert float(audio_stream[0]["start_time"]) == float(video_stream[0]["start_time"]) == 0
        frames = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time,width,height", "-of", "json", path], text=True))["frames"]
        pts = [float(f["best_effort_timestamp_time"]) for f in frames]
        assert pts == source["source_pts"] and len(pts) == 124
        assert all(math.isfinite(p) and abs(p-i/24) <= 1e-6 for i,p in enumerate(pts))
        assert all((f["width"],f["height"]) == (640,384) for f in frames)
        pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
            "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
        assert hashlib.sha256(pcm).hexdigest() == source["native_stereo_pcm_sha256"]
        native = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2).mean(axis=1)
        mono = torchaudio.functional.resample(torch.from_numpy(native), 32000, 48000,
            lowpass_filter_width=6, rolloff=.99, resampling_method="sinc_interp_hann", beta=None).numpy()
        assert len(native) == source["native_samples"] and len(mono) == source["samples"] == math.ceil(len(native) * 1.5)
        assert hashlib.sha256(mono.tobytes()).hexdigest() == source["resampled_mono_sha256"]
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "4", "-i", path,
            "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        assert hashlib.sha256(raw).hexdigest() == source["raw_rgb_sha256"]
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(124, 384, 640, 3)
        assert len(row["windows"]) == 3
        window_scores = []
        for j, (start, end) in enumerate(intent["policy"]["window_seconds"]):
            w = row["windows"][j]
            assert w == read_json(RUN / f"{cid}-w{j}.json")
            assert w["seconds"] == [start,end] and w["window_id"] == j
            assert w["source_frame_indices"] == list(range(round(start*24),round(end*24)))
            assert w["source_audio_sample_range"] == [round(start*48000),round(end*48000)]
            batch = torch.from_numpy(rgb[round(start*24):round(end*24)].copy()).permute(0,3,1,2)
            resized = VF.resize(batch, [336,336], interpolation=VF.InterpolationMode.BILINEAR, antialias=True)
            video = ((resized.float()-127.5)/127.5).unsqueeze(0)
            audio = mono[round(start*48000):round(end*48000)]
            expected = {"pixel_values_videos": record(video),
                        "input_values": record(torch.from_numpy(audio.copy()).reshape(1,1,-1)),
                        "padding_mask": record(torch.ones(1,96000,dtype=torch.int32))}
            assert expected == w["input_tensors"], (cid,j,"model inputs")
            assert float(np.abs(audio).max()) == w["audio_peak"]
            align = w["alignment"]
            assert align["video_shape"] == [1,48,768] and align["audio_shape"] == [1,50,768]
            assert align["all_masks_valid"] and align["unchanged_upstream_alignment"]
            assert align["video_index_for_audio_feature"] == [i*48//50 for i in range(50)]
            assert Path(w["embeddings_file"]).name == w["embeddings_file"]
            embedding_path = RUN / w["embeddings_file"]
            assert digest(embedding_path) == w["embeddings_sha256"]
            data = load_file(embedding_path)
            assert set(data) == {"audio","video"}
            for t in data.values():
                assert t.shape == (1,50,1024) and bool(torch.isfinite(t).all())
                assert float((t.norm(dim=-1)-1).abs().max()) < 1e-5
            diagonal = np.einsum("btd,btd->bt", data["audio"].numpy().astype(np.float64), data["video"].numpy().astype(np.float64))[0]
            assert float(np.max(np.abs(diagonal-w["diagonal"]))) < 1e-6
            assert abs(float(diagonal.mean())-w["score"]) < 1e-6
            window_scores.append(float(diagonal.mean()))
            if cid in ("A05","A09"):
                previous = ROOT / f"runs/controls-02/{cid}-original-0-w{j}.json"
                before = read_json(previous)
                for key in ("input_tensors","alignment","diagonal","score","audio_peak","embeddings_sha256"):
                    assert w[key] == before[key], (cid,j,key,"control anchor drift")
                anchors[f"{cid}-w{j}"] = {"inputs_exact":True,"embeddings_exact":True,"diagonal_exact":True}
        score = sum(window_scores)/3
        assert abs(score-row["raw_score"]) < 1e-6
        assert row["assessable"] and row["reported_score"] == row["raw_score"]
        source_scores[case["sha256"]] = row["raw_score"]
        checked.append({"case_id":cid,"score":row["raw_score"],"reconstructed_score":score})
    ratings_path = REPO / intent["policy"]["human_ratings"]
    assert digest(ratings_path) == intent["policy"]["human_ratings_sha256"] == comparison["ratings_sha256"]
    labels = read_json(ratings_path)["ratings"]
    assert len(labels) == len(comparison["rows"]) == 14
    for label, row in zip(labels,comparison["rows"]):
        assert label["original_sha256"] == row["source_sha256"]
        assert label["scores"]["synchronization"] == row["human_sync"]
        assert (label["pair"],label["variant"],label["blind_id"]) == (row["pair"],row["variant"],row["blind_id"])
        assert source_scores[row["source_sha256"]] == row["fate_score"]
        assert next(c["case_id"] for c in cases if c["sha256"] == row["source_sha256"]) == row["case_id"]
    counts_by_pair = {}
    for pair in ("combat_cinema","combat_repair"):
        group = [r for r in comparison["rows"] if r["pair"]==pair]
        assert len(group)==7
        comparisons=[]
        for i,a in enumerate(group):
            for b in group[i+1:]:
                h,m = a["human_sync"]-b["human_sync"], a["fate_score"]-b["fate_score"]
                relation = ("both_tied" if m==0 else "human_tied") if h==0 else ("model_tied" if m==0 else ("concordant" if h*m>0 else "discordant"))
                comparisons.append({"a":a["blind_id"],"b":b["blind_id"],"relation":relation,
                                    "human":[a["human_sync"],b["human_sync"]],"model":[a["fate_score"],b["fate_score"]]})
        actual = comparison["comparisons"][pair]
        assert len(comparisons)==21 and comparisons==actual["comparisons"]
        counts={k:sum(r["relation"]==k for r in comparisons) for k in ("concordant","discordant","model_tied","human_tied","both_tied","missing")}
        eligible=counts["concordant"]+counts["discordant"]+counts["model_tied"]
        fraction=(counts["concordant"]+.5*counts["model_tied"])/eligible if eligible else None
        assert counts==actual["counts"] and eligible==actual["human_untied_eligible"]
        assert fraction==actual["tie_adjusted_descriptive_fraction"]
        counts_by_pair[pair]={"counts":counts,"fraction":fraction}
    for flag in ("qualified_for_av_quality","heldout_admitted","production_changed"):
        assert summary[flag] is False and comparison[flag] is False
    save_new(output,{"utc":now(),"auditor_sha256":digest(__file__),"record_helper_sha256":digest(REPO/"scripts/h3_fate_control_audit.py"),
        "summary_sha256":digest(RUN/"summary.json"),"human_comparison_sha256":digest(RUN/"human-comparison.json"),
        "cases":checked,"windows_checked":36,"exact_input_tensor_records":108,"all_embedding_scores_reconstructed":True,
        "original_control_anchors":anchors,"all_fourteen_contextual_labels_preserved":True,
        "all_42_comparisons_reproduced":True,"counts_by_pair":counts_by_pair,
        "model_forward":False,"gpu_used":False,"heldout_admitted":False,"qualified_for_av_quality":False,"production_changed":False})
    print(json.dumps({"audit_sha256":digest(output),"cases":len(checked),"windows":36,"comparisons":counts_by_pair}),flush=True)


if __name__=="__main__":
    audit()
