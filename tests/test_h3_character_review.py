import copy
import json
from pathlib import Path

import pytest

from scripts import h3_character_review as review


def test_page_does_not_claim_one_shared_prompt_or_fully_blinded_controls():
    page = review.page(dict(review_id="</script>unsafe", cases=[]), ["<p>Group-specific prompt</p>"])
    assert "two pairs have different targets" in page
    assert "not a fully double-blind trial" in page
    assert "Identity: head/face/feature fidelity" in page
    assert "visibility failures" in page
    assert "</script>unsafe" not in page
    assert "Group-specific prompt" in page


def test_character_labels_are_distinct_from_old_av_ratings():
    key = dict(review_id="r", plan_sha256="p", cases=[dict(blind_id="C01", media_sha256="m")])
    row = dict(blind_id="C01", media_sha256="m", full_video_reviewed=True, audio_listened=True,
        scores={d: 3 for d in review.DIMENSIONS})
    labels = dict(review_id="r", plan_sha256="p", reviewer="synthetic fixture", ratings=[row])
    assert review.validate_labels(labels, key)[0]["scores"]["identity"] == 3
    for field, value in (("audio_listened", False), ("full_video_reviewed", False), ("media_sha256", "wrong")):
        changed = copy.deepcopy(labels)
        changed["ratings"][0][field] = value
        with pytest.raises(ValueError):
            review.validate_labels(changed, key)
    for value in (True, float("nan"), 5, -.1):
        changed = copy.deepcopy(labels)
        changed["ratings"][0]["scores"]["identity"] = value
        with pytest.raises(ValueError, match="integer"):
            review.validate_labels(changed, key)
    with pytest.raises(ValueError, match="repeated"):
        review.validate_labels(dict(labels, ratings=[row, row]), key)
    with pytest.raises(ValueError, match="No completed"):
        review.validate_labels(dict(labels, ratings=[]), key)


def test_build_has_distinct_pair_prompts_anchors_and_no_public_method_names(tmp_path, monkeypatch):
    from scripts.h3_character_benchmark import ARMS
    root, sources = tmp_path / "runs", tmp_path / "sources"
    sources.mkdir()
    hashes = {}
    for name in ("sully", "series30"):
        ref = sources / f"{name}-reference.mp4"
        ref.write_bytes(name.encode())
        hashes[name] = review.digest(ref)
    monkeypatch.setattr(review, "REFERENCE_HASHES", hashes)
    monkeypatch.setattr(review, "expected_graph", lambda plan, job: {"job": job["id"]})
    monkeypatch.setattr(review, "output_video", lambda history: Path(history["fixture_video"]))
    def remux(command, **kwargs):
        source = Path(command[command.index("-i") + 1])
        Path(command[-1]).write_bytes(source.read_bytes())
    monkeypatch.setattr(review.subprocess, "run", remux)
    cases, jobs = [], []
    for pair in ("series30_cinema", "sully_combat"):
        for arm in ARMS:
            job = dict(id=f"{pair}-{arm}", prompt=pair, pair=pair, variant=arm, stage="calibration", seed=31)
            jobs.append(job)
            cases.append(dict(job_id=job["id"], **{k: job[k] for k in ("pair", "prompt", "variant", "stage", "seed")}))
            run = root / job["id"]
            (run / "media-audit").mkdir(parents=True)
            video = run / "fixture.mp4"
            video.write_bytes(job["id"].encode())
            (run / "history.json").write_text(json.dumps(dict(status=dict(status_str="success"),
                prompt=[None, None, {"job": job["id"]}], fixture_video=str(video))))
            (run / "media-audit/metrics.json").write_text(json.dumps(dict(video_sha256=review.digest(video), decoded_frames=124)))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(root=str(root), pairs=["series30_cinema", "sully_combat"],
        jobs=jobs, cases=cases, prompt_text=dict(series30_cinema="Cinema-specific target <script>unsafe</script>", sully_combat="Combat-specific target"))))
    out = tmp_path / "review"
    review.build(plan, sources, out, "calibration", 31)
    text = (out / "review.html").read_text()
    assert "Cinema-specific target &lt;script&gt;unsafe&lt;/script&gt;" in text
    assert "Combat-specific target" in text
    assert text.count("data-dim=\"identity\"") == 18
    assert text.count("Creator identity anchor") == 2
    assert "stable_tuner_winner" not in text
    key = json.loads((out / "private-key.json").read_text())
    assert len(key["cases"]) == 18 and len(key["anchors"]) == 2
    assert key["ratings"] == []
    assert len(list(out.glob("*.mp4"))) == 20
    with pytest.raises(FileExistsError):
        review.build(plan, sources, out, "calibration", 31)
