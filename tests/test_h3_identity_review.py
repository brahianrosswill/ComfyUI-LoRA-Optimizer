import json
import shutil
import subprocess

import pytest

from scripts import h3_identity_review as review


def test_script_json_cannot_escape_into_executable_html():
    hostile = {"prompt": "</script><script>alert(1)</script>&\u2028"}
    encoded = review.script_json(hostile)
    assert "<" not in encoded and "&" not in encoded
    assert json.loads(encoded) == hostile


def test_review_preserves_media_and_requires_all_completed_cases(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"synthetic not-rendered fixture")
    sources = tmp_path / "sources"
    sources.mkdir()
    jobs, prompts = [], {}
    for name in ("sully", "series30"):
        (sources / f"{name}-reference.mp4").write_bytes(name.encode())
        (sources / f"{name}-source.json").write_text(json.dumps({"source_url": "https://example.invalid/"}))
        for suffix in ("source", "turn"):
            prompt = f"{name}_{suffix}"
            prompts[prompt] = '<script>untrusted reference</script>'
            for seed in review.SEEDS:
                for variant in ("base", "character"):
                    job_id = f"{prompt}-{seed}-{variant}"
                    jobs.append(dict(id=job_id, prompt=prompt, seed=seed, variant=variant))
                    run = tmp_path / job_id
                    run.mkdir()
                    (run / "history.json").write_text('{}')
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(dict(root=str(tmp_path), jobs=jobs,
        prompt_text=prompts, adapters=dict(sully={}, series30={}))))
    monkeypatch.setattr(review, "expected_graph", lambda *args: {})
    monkeypatch.setattr(review, "output_video", lambda *args: source)
    monkeypatch.setattr(review, "completed", lambda *args: False)
    with pytest.raises(ValueError, match="Incomplete"):
        review.build(plan_path, sources, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    monkeypatch.setattr(review, "completed", lambda *args: True)
    result = review.build(plan_path, sources, tmp_path / "review")
    assert result["cases"] == 16 and result["unique_media"] == 3
    page = (tmp_path / "review/review.html").read_text()
    assert '<script>untrusted' not in page
    assert '&lt;script&gt;untrusted' in page
    assert 'controls muted' in page and 'other.pause()' in page
    manifest = json.loads((tmp_path / "review/manifest.json").read_text())
    assert manifest["ratings"] == [] and manifest["unblinded"] is True
    assert manifest["audio_listened"] is False
    assert source.read_bytes() == b"synthetic not-rendered fixture"
    if shutil.which("node"):
        script = page.split('<script>', 1)[1].split('</script>', 1)[0]
        subprocess.run(['node', '--check'], input=script, text=True, check=True, capture_output=True)
