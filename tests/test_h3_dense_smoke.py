import json

import pytest

from scripts import h3_dense_smoke as m


def prepared(tmp_path, strength=1.):
    loras = [["h3-autotuner-study-20260908/series30-cinema-ties-01-mapped.safetensors", strength]]
    manifest = dict(base="http://127.0.0.1:8189", profile="local", prompt_name="cup",
        seed=2026090801, loras=loras, output_prefix="smoke", client_id="fixture")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "prompt_api.json").write_text(json.dumps(m.graph(
        m.PROMPTS["cup"], 2026090801, loras, "smoke", profile="local")))
    return tmp_path


@pytest.mark.parametrize("strength", [1., "1.0"])
def test_smoke_never_submits_over_unrelated_queue(tmp_path, monkeypatch, strength):
    run = prepared(tmp_path, strength)
    calls = []
    def request(base, path, body=None):
        calls.append((base, path, body))
        assert path == "/queue" and body is None
        return {"queue_running": [[0, "unrelated"]], "queue_pending": []}
    monkeypatch.setattr(m, "request", request)
    m.run_smoke(run)
    assert len(calls) == 1 and not (run / "submission-intent.json").exists()


def test_smoke_refuses_ambiguous_receipt_and_changed_graph(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    monkeypatch.setattr(m, "request", lambda *args: pytest.fail("No network permitted"))
    (run / "submission-intent.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        m.run_smoke(run)
    api = json.loads((run / "prompt_api.json").read_text())
    api["6"]["inputs"]["noise_seed"] = 2026090841
    (run / "prompt_api.json").write_text(json.dumps(api))
    with pytest.raises(ValueError, match="graph changed"):
        m.run_smoke(run)


def test_smoke_persists_intent_before_post_timeout(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    posts = []
    def request(base, path, body=None):
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if path.startswith("/object_info"):
            name = json.loads((run / "manifest.json").read_text())["loras"][0][0]
            return {"LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[name]]}}}}
        if path == "/system_stats":
            return {}
        assert path == "/prompt" and (run / "submission-intent.json").exists()
        posts.append(body)
        raise TimeoutError("Network result unknown")
    monkeypatch.setattr(m, "request", request)
    with pytest.raises(TimeoutError):
        m.run_smoke(run)
    assert not (run / "submission.json").exists()
    with pytest.raises(RuntimeError, match="Ambiguous"):
        m.run_smoke(run)
    assert len(posts) == 1
