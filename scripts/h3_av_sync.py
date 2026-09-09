"""Frame/waveform diagnostics for previously exposed H3 calibration videos.

No audio classification, listening judgment, synchronization score, gain change
or automatic offset correction. Uses decoded frame PTS, not packet order or an
assumed frame rate. Rejects discontinuous audio instead of hiding timing gaps.
The original rated-AV path is retained. Character observations use a separate
explicit provenance path; assistant frame observations are never human ratings.
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np

try:
    from .h3_av_review import output_video
    from .h3_benchmark import digest, save_new
except ImportError:
    from h3_av_review import output_video
    from h3_benchmark import digest, save_new


def run(*command):
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def frame_times(frames):
    times = np.asarray([float(f["best_effort_timestamp_time"]) for f in frames])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Need finite, strictly increasing decoded frame timestamps")
    return times


def audio_start(frames, sample_rate, sample_count):
    if sample_rate <= 0 or not frames or sample_count <= 0:
        raise ValueError("Empty audio or invalid sample rate")
    starts = np.asarray([float(f["best_effort_timestamp_time"]) for f in frames])
    sizes = np.asarray([int(f["nb_samples"]) for f in frames])
    if not np.isfinite(starts).all() or np.any(sizes <= 0) or sizes.sum() != sample_count:
        raise ValueError("Audio decode/probe sample-count or timestamp mismatch")
    expected = starts[0] + np.r_[0, np.cumsum(sizes[:-1])] / sample_rate
    if np.any(np.abs(starts - expected) > max(1. / sample_rate, 2e-6)):
        raise ValueError("Discontinuous audio timestamps; do not collapse gaps into a waveform")
    return float(starts[0])


def envelope(samples, sample_rate, start, window_ms=5):
    """Channel-preserving linear min/max and RMS; retain the last partial bin."""
    samples = np.asarray(samples, dtype=np.float64)
    if (samples.ndim != 2 or not len(samples) or not samples.shape[1] or
            not np.isfinite(samples).all() or not math.isfinite(start) or
            not math.isfinite(sample_rate) or sample_rate <= 0 or
            not math.isfinite(window_ms) or window_ms <= 0):
        raise ValueError("Invalid waveform input")
    width = max(1, round(sample_rate * window_ms / 1000))
    bins = []
    for left in range(0, len(samples), width):
        right = min(left + width, len(samples))
        block = samples[left:right]
        bins.append(dict(start=start + left / sample_rate, end=start + right / sample_rate,
                         minimum=block.min(axis=0).tolist(), maximum=block.max(axis=0).tolist(),
                         rms=np.sqrt(np.mean(block ** 2, axis=0)).tolist()))
    return bins


def burst_candidates(bins, separation_seconds=.08, relative_floor=.2):
    """Broadband energy peaks, NOT identified impacts or perceptual events."""
    if not bins:
        return []
    values = np.asarray([np.sqrt(np.mean(np.square(b["rms"]))) for b in bins])
    if values.max() == 0:
        return []
    maxima = [i for i in range(1, len(values) - 1)
              if values[i] > values[i - 1] and values[i] >= values[i + 1]
              and values[i] >= relative_floor * values.max()]
    selected = []
    for i in sorted(maxima, key=lambda i: (-values[i], i)):
        t = (bins[i]["start"] + bins[i]["end"]) / 2
        if all(abs(t - row["time"]) >= separation_seconds for row in selected):
            selected.append(dict(time=t, window_start=bins[i]["start"], window_end=bins[i]["end"],
                                 rms=float(values[i]), sound_identity=None))
    return sorted(selected, key=lambda row: row["time"])


def make_sheets(frames, times, bins, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=2e-6, rtol=0):
        raise ValueError("Uniform-width frame sheets require constant frame spacing")
    mid = np.asarray([(b["start"] + b["end"]) / 2 for b in bins])
    minimum = np.asarray([b["minimum"] for b in bins])
    maximum = np.asarray([b["maximum"] for b in bins])
    # One fixed absolute scale across this clip, both channels and all pages.
    limit = max(1.05, float(np.max(np.abs([minimum, maximum]))) * 1.03)
    paths = []
    for first in range(0, len(frames), 32):
        count = min(32, len(frames) - first)
        rows = math.ceil(count / 8)
        fig = plt.figure(figsize=(20, 3.6 * rows), layout="constrained")
        grid = fig.add_gridspec(rows * 2, 8, height_ratios=[2.2, 1.4] * rows)
        for row in range(rows):
            left, right = first + row * 8, min(first + row * 8 + 8, len(frames))
            for index in range(left, right):
                ax = fig.add_subplot(grid[row * 2, index - left])
                ax.imshow(frames[index])
                ax.set_title(f"f{index:03d}  {times[index]:.3f}s", fontsize=10)
                ax.axis("off")
            ax = fig.add_subplot(grid[row * 2 + 1, :right - left])
            for channel, color in enumerate(("#087fa3", "#bd506b")):
                ax.fill_between(mid, minimum[:, channel], maximum[:, channel],
                                alpha=.5, color=color, label=("Left", "Right")[channel])
            for index in range(left, right):
                ax.axvline(times[index], linewidth=.6, color="black", alpha=.3)
            ax.set_xlim(times[left], times[right - 1] + dt)
            ax.set_ylim(-limit, limit)
            ax.set_ylabel("Linear amplitude")
            ax.set_xlabel("Container timeline (seconds); vertical lines = frame presentation times")
            ax.grid(alpha=.2)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Frames {first}–{first + count - 1} with stereo waveform | no gain or timing correction", fontsize=13)
        target = out / f"frames-{first:03d}-{first + count - 1:03d}.png"
        fig.savefig(target, dpi=100)
        plt.close(fig)
        paths.append(target.name)
    return paths


def inspect(plan_path, ratings_path, job_id, out):
    if out.exists():
        raise FileExistsError(out)
    plan, ratings = json.loads(plan_path.read_text()), json.loads(ratings_path.read_text())
    if ratings["plan_sha256"] != digest(plan_path) or ratings["stage"] != "calibration":
        raise ValueError("Only the matching, already-reviewed calibration split may be inspected")
    rated = [r for r in ratings["ratings"] if r["job_id"] == job_id]
    jobs = [j for j in plan["jobs"] if j["id"] == job_id and j["stage"] == "calibration"]
    if not rated or len(jobs) != 1:
        raise ValueError("Requested job has not been rated in calibration")
    history = json.loads((Path(plan["root"]) / job_id / "history.json").read_text())
    video = output_video(history)
    source_hash = digest(video)
    if any(r["original_sha256"] != source_hash for r in rated):
        raise ValueError("Original video differs from the reviewed media")
    return inspect_video(video, source_hash, out, dict(job_id=job_id,
        plan_sha256=digest(plan_path), ratings_sha256=digest(ratings_path)))


def inspect_observed(plan_path, observations_path, key_path, job_id, out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text())
    observations = json.loads(observations_path.read_text())
    key = json.loads(key_path.read_text())
    public_path = key_path.with_name("public.json")
    public = json.loads(public_path.read_text())
    plan_sha = digest(plan_path)
    if (observations["plan_sha256"] != plan_sha or key["plan_sha256"] != plan_sha
            or public["plan_sha256"] != plan_sha or observations["stage"] != "calibration"
            or key["stage"] != "calibration" or public["stage"] != "calibration"
            or observations["review_id"] != key["review_id"] or public["review_id"] != key["review_id"]
            or digest(public_path) != observations["public_manifest_sha256"]):
        raise ValueError("Only matching, already-observed character calibration may be inspected")
    cases = [c for c in key["cases"] if c["job_id"] == job_id]
    jobs = [j for j in plan["jobs"] if j["id"] == job_id and j["stage"] == "calibration"]
    if len(cases) != 1 or len(jobs) != 1:
        raise ValueError("Requested calibration case is missing or ambiguous")
    case, job = cases[0], jobs[0]
    rows = [r for r in observations["observations"] if r["blind_id"] == case["blind_id"]]
    visible = [c for c in public["cases"] if c["blind_id"] == case["blind_id"]]
    if (len(rows) != 1 or len(visible) != 1 or rows[0]["pair"] != job["pair"]
            or visible[0]["media_sha256"] != case["media_sha256"]
            or (job["seed"], case["seed"], key["seed"]) != (observations["seed"],) * 3):
        raise ValueError("Observation/seed does not identify the requested case")
    history = json.loads((Path(plan["root"]) / job_id / "history.json").read_text())
    try:
        from .h3_character_benchmark import expected_graph
    except ImportError:
        from h3_character_benchmark import expected_graph
    if history["prompt"][2] != expected_graph(plan, job):
        raise ValueError("Executed graph differs from frozen character case")
    video = output_video(history)
    source_hash = digest(video)
    if source_hash != case["original_sha256"]:
        raise ValueError("Original video differs from the observed media")
    return inspect_video(video, source_hash, out, dict(job_id=job_id, plan_sha256=plan_sha,
        observations_sha256=digest(observations_path), review_key_sha256=digest(key_path),
        blind_id=case["blind_id"], evidence_kind="assistant_frame_observations",
        human_ratings_supplied=False))


def inspect_video(video, source_hash, out, provenance):
    """Shared decode/clock math; callers establish exposure and exact identity."""
    if out.exists():
        raise FileExistsError(out)
    probe = json.loads(run("ffprobe", "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(video)))
    visual = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audio = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if len(visual) != 1 or len(audio) != 1 or audio[0]["channels"] != 2:
        raise ValueError("Expected one video and one stereo audio stream")
    v, a = visual[0], audio[0]
    vf = [f for f in probe["frames"] if f["media_type"] == "video"]
    af = [f for f in probe["frames"] if f["media_type"] == "audio"]
    times = frame_times(vf)
    if len(times) > 400 or v["width"] * v["height"] > 640 * 384:
        raise ValueError("Diagnostic is limited to the small H3 study profile")
    raw = run("ffmpeg", "-v", "error", "-threads", "1", "-i", str(video), "-map", "0:v:0",
              "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, v["height"], v["width"], 3)
    if len(frames) != len(times):
        raise ValueError("Video decode/probe frame-count mismatch")
    raw = run("ffmpeg", "-v", "error", "-threads", "1", "-i", str(video), "-map", "0:a:0",
              "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    rate = int(a["sample_rate"])
    start = audio_start(af, rate, len(samples))
    bins = envelope(samples, rate, start)
    out.mkdir(parents=True, exist_ok=False)
    sheets = make_sheets(frames, times, bins, out)
    # Full-resolution extracted stills make frame stepping exact; browser video
    # seeking alone cannot guarantee which decoded frame is displayed.
    from PIL import Image
    images = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    display = dict(times=times.tolist(), images=images, waveform=bins)
    page = PAGE.replace("__DATA__", json.dumps(display).replace("<", "\\u003c"))
    with (out / "inspect.html").open("x") as stream:
        stream.write(page)
    if digest(video) != source_hash:
        raise ValueError("Source video changed during diagnostic extraction")
    record = dict(provenance, video_sha256=source_hash, script_sha256=digest(__file__),
        ffmpeg_version=run("ffmpeg", "-version").decode().splitlines()[0],
        video_frame_pts=times.tolist(), audio_start=start, audio_sample_rate=rate,
        audio_decoded_samples=len(samples), envelope_window_ms=5, waveform=bins,
        burst_candidates=burst_candidates(bins),
        burst_detector={"min_separation_seconds": .08, "relative_rms_floor": .2,
                        "interpretation": "Broadband energy maxima, not identified punch sounds or onsets"},
        frame_sheets=sheets, full_motion_reviewed=False, audio_listened=False,
        synchronization_score=None, automatic_offset_correction=None,
        note="Timestamped diagnostics only. Visible contact and sound identity need separate annotation; no inferred preference.")
    save_new(out / "sync.json", record)
    return dict(job=provenance["job_id"], frames=len(frames), sheets=len(sheets), audio_start=start,
                candidates=record["burst_candidates"], inspector=str(out / "inspect.html"))


PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 frame and waveform inspector</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;background:#17191d;color:#eee}
img{width:640px;max-width:100%;display:block}canvas{width:100%;height:230px;background:#fafafa}
input{width:65%}button{font:inherit;padding:8px}</style>
<h1>Frame and waveform inspector</h1><p>Calibration diagnostic, not a blind rating page. Exact decoded stills;
stereo 5 ms min/max envelopes, original linear amplitude. No sound classification or timing correction.</p>
<img id="frame" alt="Selected decoded frame"><p><button id="previous">Previous frame</button>
<input id="seek" type="range" min="0" value="0"><button id="next">Next frame</button></p>
<p id="time"></p><canvas id="wave" width="1100" height="230"></canvas>
<p>Blue: left channel; pink: right channel; orange: selected frame presentation time.
Click the waveform or use arrow keys to select the nearest frame. Sound peaks alone do not prove impacts.</p>
<script>
const data=__DATA__, seek=document.querySelector('#seek'), canvas=document.querySelector('#wave');
seek.max=data.times.length-1;
const start=Math.min(data.times[0],data.waveform[0].start);
const end=Math.max(data.times.at(-1),data.waveform.at(-1).end);
const amplitude=Math.max(1.05,...data.waveform.flatMap(b=>[...b.minimum,...b.maximum].map(Math.abs)));
function draw(){
  const i=Number(seek.value), t=data.times[i], ctx=canvas.getContext('2d');
  document.querySelector('#frame').src=data.images[i];
  document.querySelector('#time').textContent='Frame '+i+' / '+(data.times.length-1)+' — PTS '+t.toFixed(6)+' s';
  ctx.clearRect(0,0,1100,230);
  const x=t=>30+(t-start)/(end-start)*1040;
  for(let c=0;c<2;c++){
    const center=55+c*95;ctx.strokeStyle=['#087fa3','#bd506b'][c];ctx.beginPath();
    for(const b of data.waveform){const at=x((b.start+b.end)/2);ctx.moveTo(at,center-b.minimum[c]/amplitude*40);ctx.lineTo(at,center-b.maximum[c]/amplitude*40);}
    ctx.stroke();ctx.fillStyle='#333';ctx.fillText(['L','R'][c],5,center);
  }
  ctx.strokeStyle='#c66c00';ctx.beginPath();ctx.moveTo(x(t),5);ctx.lineTo(x(t),195);ctx.stroke();
  ctx.fillStyle='#333';for(let tick=Math.ceil(start);tick<=end;tick++)ctx.fillText(tick+'s',x(tick),218);
}
function step(d){seek.value=Math.max(0,Math.min(data.times.length-1,Number(seek.value)+d));draw();}
seek.oninput=draw;document.querySelector('#previous').onclick=()=>step(-1);document.querySelector('#next').onclick=()=>step(1);
document.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}};
canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(), t=start+((e.clientX-rect.left)/rect.width*1100-30)/1040*(end-start);
let best=0;data.times.forEach((value,i)=>{if(Math.abs(value-t)<Math.abs(data.times[best]-t))best=i;});seek.value=best;draw();};
draw();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--ratings", type=Path)
    evidence.add_argument("--observations", type=Path)
    parser.add_argument("--review-key", type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.observations:
        if args.review_key is None:
            parser.error("--observations requires --review-key")
        result = inspect_observed(args.plan, args.observations, args.review_key, args.job, args.out)
    else:
        result = inspect(args.plan, args.ratings, args.job, args.out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
