"""Offline scheduling and experimental-design checks; no real quality labels."""
import copy
import json
from pathlib import Path

import pytest

from scripts import h3_identity_study as study


@pytest.fixture
def plan(tmp_path):
    adapters = {n: dict(api_name=f"{n}.safetensors", strength=s)
                for n, s in (("sully", 1.), ("series30", .8))}
    prompts = {f"{n}_{p}": f"Original unchanged text: {n} {p}"
               for n in adapters for p in ("source", "turn")}
    return dict(base=study.BASE, profile="local", width=640, height=384, length=124,
                steps=20, root=str(tmp_path), adapters=adapters, prompt_text=prompts,
                source_sha256={}, jobs=study.matrix(adapters, prompts))


def test_matrix_is_frozen_matched_and_calibration_only(plan):
    assert len(plan["jobs"]) == 16
    assert len({j["id"] for j in plan["jobs"]}) == 16
    assert study.matrix(plan["adapters"], plan["prompt_text"]) == plan["jobs"]
    assert {j["stage"] for j in plan["jobs"]} == {"calibration"}
    for character in plan["adapters"]:
        for prompt in (f"{character}_source", f"{character}_turn"):
            for seed in study.SEEDS:
                pair = [j for j in plan["jobs"] if (j["prompt"], j["seed"]) == (prompt, seed)]
                assert {j["variant"] for j in pair} == {"base", "character"}
                assert [j for j in pair if j["variant"] == "base"][0]["loras"] == []


def test_graph_preserves_prompt_and_small_local_profile(plan):
    for job in plan["jobs"]:
        graph = study.expected_graph(plan, job)
        assert graph["5"]["inputs"]["prompt"] == plan["prompt_text"][job["prompt"]]
        assert (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]) == (640, 384)
        assert graph["2"]["inputs"]["clip_name"].endswith("nvfp4_awq.safetensors")
        assert graph["1"]["inputs"]["unet_name"].endswith("int8_convrot.safetensors")
        assert graph["8"]["inputs"]["steps"] == 20
        assert graph["6"]["inputs"]["noise_seed"] == job["seed"]


@pytest.mark.parametrize("field,value", [("base", "http://other:8188"), ("width", 1280), ("steps", 4)])
def test_rejects_unapproved_profile(plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError, match="profile"):
        study.validate_plan(plan, verify_files=False)


def test_rejects_changed_matrix_and_installed_hash(plan, monkeypatch):
    study.validate_plan(plan, verify_files=False)
    changed = copy.deepcopy(plan)
    changed["jobs"][0]["seed"] += 1
    with pytest.raises(ValueError, match="matrix"):
        study.validate_plan(changed, verify_files=False)
    for adapter in plan["adapters"].values():
        adapter.update(local_path="synthetic", sha256="expected")
    monkeypatch.setattr(study, "digest", lambda p: "changed")
    with pytest.raises(ValueError, match="Installed adapter"):
        study.validate_plan(plan)


def test_prepare_job_never_overwrites_changed_graph(plan):
    job = plan["jobs"][0]
    run, _ = study.prepare_job(plan, job)
    graph_path = run / "prompt_api.json"
    graph_path.write_text('{}')
    with pytest.raises(ValueError, match="Prepared graph"):
        study.prepare_job(plan, job)
    assert graph_path.read_text() == '{}'


def test_completed_requires_success_and_exact_graph(tmp_path):
    assert not study.completed(tmp_path, {})
    path = tmp_path / "history.json"
    path.write_text(json.dumps(dict(status=dict(status_str="error"))))
    with pytest.raises(RuntimeError, match="failed"):
        study.completed(tmp_path, {})
    path.write_text(json.dumps(dict(status=dict(status_str="success"), prompt=[None, None, {"wrong": 1}])))
    with pytest.raises(ValueError, match="Executed graph"):
        study.completed(tmp_path, {})


def test_unrelated_queue_prevents_submission(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda p: None)
    requests = []
    def request(base, path, body=None):
        requests.append(path)
        return {"queue_running": [[0, "user-job"]], "queue_pending": []}
    monkeypatch.setattr(study, "request", request)
    study.run_batch(plan, 1)
    assert requests == ["/queue"]
    assert not list(Path(plan["root"]).glob("*/submission*.json"))


def test_timeout_intent_prevents_duplicate_post(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda p: None)
    posts = []
    def request(base, path, body=None):
        if path == "/prompt":
            posts.append(body)
            raise TimeoutError("uncertain receipt")
        return {"queue_running": [], "queue_pending": []} if path == "/queue" else {}
    monkeypatch.setattr(study, "request", request)
    with pytest.raises(TimeoutError):
        study.run_batch(plan, 1)
    assert len(posts) == 1
    with pytest.raises(RuntimeError, match="Ambiguous prior submission"):
        study.run_batch(plan, 1)
    assert len(posts) == 1


def test_existing_submission_is_watched_not_reposted(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda p: None)
    job = plan["jobs"][0]
    run, _ = study.prepare_job(plan, job)
    (run / "submission.json").write_text(json.dumps({"prompt_id": "existing-live-handle"}))
    monkeypatch.setattr(study, "request", lambda *a, **k: pytest.fail("Must not submit or poll another queue"))
    def watch(command, **kwargs):
        assert command[2] == "watch"
        (run / "history.json").write_text(json.dumps({"status": {"status_str": "success"},
            "prompt": [None, None, study.expected_graph(plan, job)]}))
    monkeypatch.setattr(study.subprocess, "run", watch)
    study.run_batch(plan, 1)


def test_authored_probes_have_no_reference_socket_or_speech():
    for prompt in study.PROBES.values():
        assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
        assert "<Picture" not in prompt and "No speech" in prompt
        assert prompt.index("overall_soundscape:") < prompt.index("non_diegetic_music:")
