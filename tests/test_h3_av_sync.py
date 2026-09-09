"""Synthetic timestamp/audio fixtures; never perceptual benchmark labels."""
import shutil
import subprocess
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.h3_av_sync import audio_start, burst_candidates, envelope, frame_times, run, PAGE
from scripts import h3_av_sync as sync


@pytest.fixture
def observed_case(tmp_path, monkeypatch):
    from scripts.h3_character_benchmark import expected_graph
    job = dict(id="observed-case", stage="calibration", seed=31, pair="sully_combat",
               prompt="p", loras=[])
    plan = dict(root=str(tmp_path), jobs=[job], prompt_text={"p": "Frozen synthetic prompt"})
    plan_path, obs_path, key_path = [tmp_path / name for name in ("plan.json", "observations.json", "private-key.json")]
    plan_path.write_text(json.dumps(plan))
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"synthetic fixture")
    case = dict(blind_id="C01", job_id=job["id"], seed=31, media_sha256="copy",
                original_sha256=sync.digest(video), pair="sully_combat")
    key = dict(plan_sha256=sync.digest(plan_path), review_id="r", stage="calibration", seed=31, cases=[case])
    key_path.write_text(json.dumps(key))
    public_path = tmp_path / "public.json"
    public_path.write_text(json.dumps(dict(plan_sha256=key["plan_sha256"], review_id="r", stage="calibration",
        cases=[dict(blind_id="C01", media_sha256="copy", pair="sully_combat")])))
    observations = dict(plan_sha256=key["plan_sha256"], review_id="r", stage="calibration", seed=31,
        public_manifest_sha256=sync.digest(public_path), observations=[dict(blind_id="C01", pair="sully_combat")])
    obs_path.write_text(json.dumps(observations))
    run_dir = tmp_path / job["id"]
    run_dir.mkdir()
    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(dict(prompt=[None, None, expected_graph(plan, job)])))
    monkeypatch.setattr(sync, "output_video", lambda h: video)
    monkeypatch.setattr(sync, "inspect_video", lambda video, sha, out, provenance: provenance)
    return plan_path, obs_path, key_path, job["id"], tmp_path / "out"


def test_observed_character_path_does_not_fabricate_human_ratings(observed_case):
    provenance = sync.inspect_observed(*observed_case)
    assert provenance["evidence_kind"] == "assistant_frame_observations"
    assert provenance["human_ratings_supplied"] is False
    assert "ratings_sha256" not in provenance
    assert provenance["observations_sha256"] == sync.digest(observed_case[1])


@pytest.mark.parametrize("change", ["heldout", "missing", "seed", "review", "media", "public", "graph"])
def test_observed_character_scope_and_provenance_are_required(observed_case, change):
    plan_path, obs_path, key_path, job, out = observed_case
    if change in ("heldout", "missing", "seed", "review"):
        value = json.loads(obs_path.read_text())
        if change == "heldout":
            value["stage"] = "heldout"
        elif change == "missing":
            value["observations"] = []
        elif change == "seed":
            value["seed"] = 32
        else:
            value["review_id"] = "different"
        obs_path.write_text(json.dumps(value))
    elif change == "media":
        value = json.loads(key_path.read_text())
        value["cases"][0]["original_sha256"] = "changed"
        key_path.write_text(json.dumps(value))
    elif change == "public":
        key_path.with_name("public.json").write_text('{}')
    else:
        (plan_path.parent / job / "history.json").write_text(json.dumps(dict(prompt=[None, None, {}])))
    with pytest.raises((ValueError, KeyError)):
        sync.inspect_observed(*observed_case)
    assert not out.exists()


def test_original_rated_av_path_retains_ratings_provenance(observed_case):
    plan_path, obs_path, key_path, job, out = observed_case
    case = json.loads(key_path.read_text())["cases"][0]
    ratings = obs_path.with_name("ratings.json")
    ratings.write_text(json.dumps(dict(plan_sha256=sync.digest(plan_path), stage="calibration",
        ratings=[dict(job_id=job, original_sha256=case["original_sha256"])])))
    provenance = sync.inspect(plan_path, ratings, job, out)
    assert provenance["ratings_sha256"] == sync.digest(ratings)
    assert "observations_sha256" not in provenance


def test_audio_clock_preserves_nonzero_start_and_rejects_gaps():
    frames = [{"best_effort_timestamp_time": .1, "nb_samples": 10},
              {"best_effort_timestamp_time": .11, "nb_samples": 10}]
    assert audio_start(frames, 1000, 20) == .1
    with pytest.raises(ValueError, match="sample-count"):
        audio_start(frames, 1000, 19)
    frames[1]["best_effort_timestamp_time"] = .13
    with pytest.raises(ValueError, match="Discontinuous"):
        audio_start(frames, 1000, 20)


def test_frame_times_keep_pts_not_assumed_framerate():
    assert frame_times([{"best_effort_timestamp_time": t} for t in (.2, .24, .3)]).tolist() == [.2, .24, .3]
    for times in ((0, 0), (1, 0), (0, float("nan"))):
        with pytest.raises(ValueError, match="timestamps"):
            frame_times([{"best_effort_timestamp_time": t} for t in times])


def test_waveform_preserves_phase_channels_and_partial_last_bin():
    samples = np.array([[1., -1.], [.5, -.5], [0., 0.]])
    bins = envelope(samples, 1000, .125, 2)
    assert len(bins) == 2
    assert bins[0]["start"] == .125 and bins[-1]["end"] == .128
    assert bins[0]["minimum"] == [.5, -1.]
    assert bins[0]["maximum"] == [1., -.5]
    assert bins[0]["rms"] == pytest.approx([np.sqrt(.625), np.sqrt(.625)])
    assert bins[1]["rms"] == [0., 0.]


def test_peak_candidates_do_not_classify_sounds_or_invent_silence_events():
    samples = np.zeros((500, 2))
    assert burst_candidates(envelope(samples, 1000, 0)) == []
    samples[100:105, 0] = 1
    samples[150:155, 0] = .5  # Suppressed by the declared 80 ms separation.
    samples[300:305, 1] = .8
    peaks = burst_candidates(envelope(samples, 1000, .2))
    assert [p["time"] for p in peaks] == pytest.approx([.3025, .5025])
    assert all(p["sound_identity"] is None for p in peaks)


def test_waveform_rejects_nonfinite_and_empty_input():
    for samples in (np.empty((0, 2)), np.array([[float("inf"), 0]]), np.array([1., 2.])):
        with pytest.raises(ValueError, match="Invalid"):
            envelope(samples, 1000, 0)


def test_inspector_javascript_syntax():
    if not shutil.which("node"):
        pytest.skip("Node.js is unavailable")
    script = PAGE.replace("__DATA__", "{}").split("<script>", 1)[1].split("</script>", 1)[0]
    subprocess.run(["node", "--check"], input=script, text=True, check=True, capture_output=True)


def test_real_decode_keeps_a_known_audio_offset(tmp_path):
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        pytest.skip("FFmpeg tools unavailable")
    video = tmp_path / "synthetic-offset.mov"
    run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=24:d=0.5",
        "-f", "lavfi", "-i", r"aevalsrc=if(between(t\,0.2\,0.205)\,0.8\,0)|0:s=32000:d=0.5",
        "-filter_complex", "[1:a]asetpts=PTS+0.125/TB[a]", "-map", "0:v:0", "-map", "[a]",
        "-c:v", "png", "-c:a", "pcm_f32le", "-f", "mov", "-n", str(video))
    probe = json.loads(run("ffprobe", "-v", "error", "-show_frames", "-of", "json", str(video)))
    audio = [f for f in probe["frames"] if f["media_type"] == "audio"]
    visual = [f for f in probe["frames"] if f["media_type"] == "video"]
    pcm = np.frombuffer(run("ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0",
                           "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"), dtype="<f4").reshape(-1, 2)
    start = audio_start(audio, 32000, len(pcm))
    assert start == pytest.approx(.125)
    assert frame_times(visual)[0] == pytest.approx(0)
    bins = envelope(pcm, 32000, start)
    first = next(b for b in bins if b["maximum"][0] > .1)
    assert first["start"] == pytest.approx(.325)
    assert all(b["maximum"][1] == 0 for b in bins)


def test_inspector_frame_controls_and_waveform_seek():
    if not shutil.which("node"):
        pytest.skip("Node.js is unavailable")
    data = dict(times=[.1, .2, .3], images=["f0", "f1", "f2"],
                waveform=[dict(start=0., end=.4, minimum=[-.5, -.4], maximum=[.5, .4])])
    script = PAGE.replace("__DATA__", json.dumps(data)).split("<script>", 1)[1].split("</script>", 1)[0]
    prelude = '''
const assert=require('node:assert/strict');
const elements={seek:{value:0},frame:{},time:{},previous:{},next:{},wave:{
  getContext:()=>new Proxy({}, {get:()=>()=>{}}), getBoundingClientRect:()=>({left:0,width:1100})}};
const document={querySelector:s=>elements[s.slice(1)]};
'''
    checks = '''
assert.equal(elements.frame.src,'f0');
elements.next.onclick();assert.equal(elements.frame.src,'f1');
elements.previous.onclick();assert.equal(elements.frame.src,'f0');
elements.previous.onclick();assert.equal(elements.frame.src,'f0');
elements.wave.onclick({clientX:1070});assert.equal(elements.frame.src,'f2');
elements.wave.onclick({clientX:0});assert.equal(elements.frame.src,'f0');
'''
    subprocess.run(["node"], input=prelude + script + checks, text=True, check=True, capture_output=True)
