"""Offline safeguards for the nine-arm study; never submit real generations."""
import json
from pathlib import Path

import pytest

from scripts import h3_character_benchmark as study


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ARTIFACTS", tmp_path / "runs")
    monkeypatch.setattr(study, "INSTALL", tmp_path / "installed")
    study.INSTALL.mkdir()
    protocol_path = study.DATA / "2026-09-08-h3-character-merge-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assets = {key: dict(sha256=value["sha256"], strength=value["strength"],
        api_name=value.get("api_name", key + ".safetensors"))
        for key, value in {**protocol["characters"], **protocol["effects"]}.items()}
    merges = {}
    for pair in study.PAIRS:
        for variant, suffix in study.SUFFIXES.items():
            key = f"{pair}:{variant}"
            filename = f"{pair.replace('_', '-')}-{suffix}.safetensors"
            assets[key] = dict(sha256="synthetic", strength=1.,
                api_name=f"h3-autotuner-study-20260908/{filename}",
                installed_path=str(study.INSTALL / filename))
            merges[key] = dict(numerical=dict(export_sha256="synthetic"))
    prompts = {f"{c}_{e}_{stage}": protocol["templates"][f"{e}_{split['prompt_suffix']}"].format(**protocol["characters"][c])
        for pair in study.PAIRS for c, e in [pair.split("_")] for stage, split in protocol["splits"].items()}
    return dict(base=study.BASE, root=str(study.ARTIFACTS), pairs=list(study.PAIRS),
        variants=list(study.ARMS), assets=assets, merge_evidence=merges, evidence_records={},
        protocol_path=str(protocol_path), protocol_sha256=study.PROTOCOL_SHA,
        profile=protocol["profile"], prompt_text=prompts, jobs=study.matrix(protocol, assets),
        harness_sha256={})


def test_balanced_matrix_and_disjoint_holdout(plan):
    study.validate_plan(plan, verify_files=False)
    assert len(plan["jobs"]) == len({j["id"] for j in plan["jobs"]}) == 72
    for pair in study.PAIRS:
        for seed in (2026090831, 2026090832, 2026090841, 2026090842):
            cases = [j for j in plan["jobs"] if (j["pair"], j["seed"]) == (pair, seed)]
            assert {j["variant"] for j in cases} == set(study.ARMS)
            assert len({j["prompt"] for j in cases}) == 1
    cal = [j for j in plan["jobs"] if j["stage"] == "calibration"]
    held = [j for j in plan["jobs"] if j["stage"] == "heldout"]
    assert {j["prompt"] for j in cal}.isdisjoint(j["prompt"] for j in held)
    assert {j["seed"] for j in cal}.isdisjoint(j["seed"] for j in held)


def test_all_graphs_keep_exact_prompt_seed_and_local_quantization(plan):
    for job in plan["jobs"]:
        graph = study.expected_graph(plan, job)
        assert graph["5"]["inputs"]["prompt"] == plan["prompt_text"][job["prompt"]]
        assert graph["6"]["inputs"]["noise_seed"] == job["seed"]
        assert graph["1"]["inputs"]["unet_name"].endswith("int8_convrot.safetensors")
        assert graph["2"]["inputs"]["clip_name"].endswith("nvfp4_awq.safetensors")
        assert graph["2"]["inputs"]["type"] == "minimax"
        assert graph["5"]["inputs"]["width"] == 640
        assert graph["5"]["inputs"]["height"] == 384
        assert graph["5"]["inputs"]["length"] == 124
        assert graph["8"]["inputs"]["steps"] == 20
        assert len([n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]) == (job["variant"] != "base")


@pytest.mark.parametrize("change", ["prompt", "seed", "profile", "raw_strength", "merge_scale", "export_hash", "loader"])
def test_rejects_changed_conditions_even_if_matrix_recomputed(plan, change):
    if change == "prompt":
        plan["prompt_text"][plan["jobs"][0]["prompt"]] += " changed"
    elif change == "seed":
        plan["jobs"][0]["seed"] += 1
    elif change == "profile":
        plan["profile"]["steps"] = 4
    else:
        key = "sully" if change == "raw_strength" else "sully_combat:ties"
        field = {"raw_strength": "strength", "merge_scale": "strength", "export_hash": "sha256", "loader": "api_name"}[change]
        plan["assets"][key][field] = .4 if "strength" == field else "changed"
        protocol = json.loads(Path(plan["protocol_path"]).read_text())
        plan["jobs"] = study.matrix(protocol, plan["assets"])
    with pytest.raises(ValueError):
        study.validate_plan(plan, verify_files=False)


def test_pin_asset_checks_full_hash_and_detects_change_during_hash(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    sha = study.digest(source)
    asset = study.pin_asset(source, sha, "source", 1.)
    assert asset["identity"] == study.identity(source)
    with pytest.raises(ValueError, match="Asset identity"):
        study.pin_asset(source, "wrong", "source", 1.)
    def changed(path):
        source.write_bytes(b"new contents")
        return sha
    monkeypatch.setattr(study, "digest", changed)
    with pytest.raises(ValueError, match="Asset identity"):
        study.pin_asset(source, sha, "source", 1.)


def test_install_uses_same_inode_and_never_overwrites(tmp_path, monkeypatch):
    install = tmp_path / "installed"
    install.mkdir()
    monkeypatch.setattr(study, "INSTALL", install)
    monkeypatch.setattr(study, "validate_plan", lambda *a, **k: None)
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    destination = install / "test.safetensors"
    plan = dict(assets={"test": dict(local_path=str(source), installed_path=str(destination))})
    study.install(plan)
    assert source.stat().st_ino == destination.stat().st_ino
    study.install(plan)  # Idempotent verified link, not a second copy.
    other = tmp_path / "other"
    other.write_bytes(b"other")
    plan["assets"]["test"]["local_path"] = str(other)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        study.install(plan)
    assert destination.read_bytes() == b"fixture"


def test_prepare_preserves_receipts_and_rejects_changed_case(plan):
    job = plan["jobs"][0]
    run, manifest = study.prepare_job(plan, job, "plan-sha")
    assert manifest["plan_sha256"] == "plan-sha"
    assert study.prepare_job(plan, job, "plan-sha")[1] == manifest
    with pytest.raises(ValueError, match="manifest"):
        study.prepare_job(plan, job, "other-plan")
    (run / "prompt_api.json").write_text('{}')
    with pytest.raises(ValueError, match="graph/prompt"):
        study.prepare_job(plan, job, "plan-sha")
    assert (run / "prompt_api.json").read_text() == '{}'


def test_run_requires_explicit_balanced_split(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda *a, **k: None)
    monkeypatch.setattr(study, "request", lambda *a, **k: pytest.fail("No network allowed"))
    with pytest.raises(ValueError, match="18 balanced"):
        study.run_batch(plan, "p", "heldout", 2026090831, 1)
    with pytest.raises(ValueError, match="1..18"):
        study.run_batch(plan, "p", "calibration", 2026090831, 72)


def test_unrelated_work_pauses_without_post(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda *a, **k: None)
    calls = []
    def request(base, path, body=None):
        calls.append(path)
        return dict(queue_running=[[0, "user-job"]], queue_pending=[])
    monkeypatch.setattr(study, "request", request)
    study.run_batch(plan, "p", "calibration", 2026090831, 1)
    assert calls == ["/queue"]
    assert not list(Path(plan["root"]).glob("*/submission*.json"))


def test_timeout_receipt_never_causes_double_submit(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda *a, **k: None)
    calls = []
    def request(base, path, body=None):
        calls.append(path)
        if path == "/prompt":
            raise TimeoutError("unknown submission receipt")
        if path == "/object_info/LoraLoaderModelOnly":
            return dict(LoraLoaderModelOnly=dict(input=dict(required=dict(lora_name=[
                [a["api_name"] for a in plan["assets"].values()]]))))
        return dict(queue_running=[], queue_pending=[]) if path == "/queue" else {}
    monkeypatch.setattr(study, "request", request)
    with pytest.raises(TimeoutError):
        study.run_batch(plan, "p", "calibration", 2026090831, 1)
    with pytest.raises(RuntimeError, match="Ambiguous prior submission"):
        study.run_batch(plan, "p", "calibration", 2026090831, 1)
    assert calls.count("/prompt") == 1


def test_resume_watches_same_handle_without_another_post(plan, monkeypatch):
    monkeypatch.setattr(study, "validate_plan", lambda *a, **k: None)
    job = plan["jobs"][0]
    run, _ = study.prepare_job(plan, job, "p")
    (run / "submission.json").write_text(json.dumps(dict(prompt_id="existing")))
    monkeypatch.setattr(study, "request", lambda *a, **k: pytest.fail("No new network submission"))
    def watch(command, **kwargs):
        assert command[2] == "watch"
        (run / "history.json").write_text(json.dumps(dict(status=dict(status_str="success"),
            prompt=[None, None, study.expected_graph(plan, job)])))
    monkeypatch.setattr(study.subprocess, "run", watch)
    study.run_batch(plan, "p", "calibration", 2026090831, 1)


def test_record_rounding_is_separate_from_compression(tmp_path):
    from scripts.h3_study_record import collect_merge_run
    (tmp_path / "manifest.json").write_text(json.dumps(dict(adapters=[])))
    (tmp_path / "export_check.json").write_text(json.dumps(dict(groups_checked=1,
        results=[dict(finite=True, relative_export_error=.0024, error_to_method_change=.005,
            relative_normal_storage_rounding_error=.0016, relative_error_after_storage_rounding=.0028)])))
    record = collect_merge_run(tmp_path)["precision"]
    assert record["max_relative_normal_storage_rounding_error"] == .0016
    assert record["max_relative_error_after_storage_rounding"] == .0028
    assert record["max_relative_export_error"] == .0024
