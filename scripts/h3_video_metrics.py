"""Descriptive full-clip measurements, not a learned video-quality score.

Reads a confirmed local output. Keeps original media untouched and writes
diagnostic artifacts to a new directory. Requires ffmpeg, ffprobe and numpy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    def run(*cmd):
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    probe = json.loads(run("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.video)))
    raw = run("ffmpeg", "-v", "error", "-i", str(args.video), "-an", "-vf", "scale=160:96,format=gray",
              "-f", "rawvideo", "pipe:1")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 96, 160).astype(np.float32) / 255.
    step = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    metrics = {"video": str(args.video), "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
               "decoded_frames": len(frames), "frame_difference_mean": float(step.mean()),
               "frame_difference_p95": float(np.percentile(step, 95)),
               "near_static_frame_fraction": float((step < .001).mean()),
               "luma_mean": float(frames.mean()), "luma_std": float(frames.std()),
               "quality_score": None, "note": "Motion/luma/audio statistics are descriptive, not preference or synchronization scores."}
    if any(s["codec_type"] == "audio" for s in probe["streams"]):
        pcm = run("ffmpeg", "-v", "error", "-i", str(args.video), "-vn", "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1")
        samples = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2)
        windows = samples[:len(samples) // 640 * 640].reshape(-1, 640, 2)
        rms = np.sqrt(np.mean(windows**2, axis=(1, 2)))
        metrics["audio"] = {"samples": len(samples), "sample_rate": 32000,
            "peak": float(np.abs(samples).max()), "rms": float(np.sqrt(np.mean(samples**2))),
            "clipped_fraction": float((np.abs(samples) >= 1).mean()),
            "quiet_20ms_fraction": float((rms < .001).mean()),
            "rms_20ms": rms.tolist(), "listened": False, "synchronization_score": None}
    run("ffmpeg", "-v", "error", "-i", str(args.video), "-vf", "fps=4,scale=320:-1,tile=5x5",
        "-frames:v", "1", "-n", str(args.out / "frames-4fps.jpg"))
    (args.out / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "audio"}))


if __name__ == "__main__":
    main()
