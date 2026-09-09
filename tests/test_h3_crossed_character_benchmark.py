"""Crossed-stage extension tests: no real models, network or execution."""
import copy
import json
from pathlib import Path

import pytest

from scripts import h3_character_benchmark as diagonal
from scripts import h3_crossed_character_benchmark as crossed


@pytest.fixture
def plan(tmp_path, monkeypatch):
    # Synthetic merge assets exercise graph/matrix safeguards, not a numerical
    # qualification claim. Real prepare checks the actual pair's audited export.
    parent = json.loads(crossed.PARENT.read_text())
    # Evidence records retain historical machine paths. Unit tests relocate
    # only a synthetic parent copy, never mutate those historical records or
    # require the original /media/p5 workspace to exist on a CI worker.
    parent["root"] = str(diagonal.ARTIFACTS)
    parent["protocol_path"] = str(diagonal.DATA / Path(parent["protocol_path"]).name)
    parent["evidence_records"] = {str(diagonal.DATA / Path(path).name): sha
        for path, sha in parent["evidence_records"].items()}
    parent_path = tmp_path / "synthetic-parent.json"
    parent_path.write_text(json.dumps(parent))
    monkeypatch.setattr(crossed, "PARENT", parent_path)
    monkeypatch.setattr(crossed, "PARENT_SHA", crossed.digest(parent_path))
    result = copy.deepcopy(parent)
    result["pairs"] = list(crossed.PAIRS)
    result["assets"] = {k: v for k, v in result["assets"].items() if ":" not in k}
    result["merge_evidence"] = {}
    for new, old in zip(crossed.PAIRS, diagonal.PAIRS):
        for variant, suffix in diagonal.SUFFIXES.items():
            key = f"{new}:{variant}"
            asset = copy.deepcopy(parent["assets"][f"{old}:{variant}"])
            name = f"{new.replace('_', '-')}-{suffix}.safetensors"
            asset.update(installed_path=str(diagonal.INSTALL / name),
                api_name=f"h3-autotuner-study-20260908/{name}")
            result["assets"][key] = asset
            result["merge_evidence"][key] = copy.deepcopy(parent["merge_evidence"][f"{old}:{variant}"])
    protocol = json.loads(Path(result["protocol_path"]).read_text())
    result["prompt_text"] = {f"{c}_{e}_{stage}": protocol["templates"][f"{e}_{split['prompt_suffix']}"].format(**protocol["characters"][c])
        for pair in crossed.PAIRS for c, e in [pair.split("_")] for stage, split in protocol["splits"].items()}
    result["jobs"] = crossed.engine().matrix(protocol, result["assets"])
    result["cases"] = [dict(pair=j["pair"], variant=j["variant"], stage=j["stage"],
        prompt=j["prompt"], seed=j["seed"], job_id=j["id"]) for j in result["jobs"]]
    return crossed.extension_fields(result)


def test_engine_is_private_and_does_not_change_the_frozen_diagonal_module():
    first, second = crossed.engine(), crossed.engine()
    assert first is not second and first is not diagonal
    assert first.PAIRS == crossed.PAIRS
    first.PAIRS = ()
    assert second.PAIRS == crossed.PAIRS
    assert diagonal.PAIRS == ("series30_cinema", "sully_combat")


def test_disjoint_complete_crossing_with_exact_profile_and_prompts(plan):
    crossed.validate_plan(plan, verify_files=False)
    parent = json.loads(crossed.PARENT.read_text())
    assert len(plan["jobs"]) == 72
    assert {j["id"] for j in parent["jobs"]}.isdisjoint(j["id"] for j in plan["jobs"])
    for pair in crossed.PAIRS:
        for seed in (2026090831, 2026090832, 2026090841, 2026090842):
            jobs = [j for j in plan["jobs"] if (j["pair"], j["seed"]) == (pair, seed)]
            assert len(jobs) == 9 and {j["variant"] for j in jobs} == set(diagonal.ARMS)
            assert len({j["prompt"] for j in jobs}) == 1
            for job in jobs:
                graph = diagonal.expected_graph(plan, job)
                assert graph["5"]["inputs"]["prompt"] == plan["prompt_text"][job["prompt"]]
                assert graph["5"]["inputs"]["width"] == 640
                assert graph["5"]["inputs"]["height"] == 384
                assert graph["5"]["inputs"]["length"] == 124
                assert graph["6"]["inputs"]["noise_seed"] == seed
                assert graph["1"]["inputs"]["unet_name"].endswith("int8_convrot.safetensors")
                assert graph["2"]["inputs"]["clip_name"].endswith("nvfp4_awq.safetensors")


@pytest.mark.parametrize("fault", ["phase", "parent", "helper", "prompt", "raw", "missing", "seed", "cases"])
def test_extension_rejects_changed_scope_before_execution(plan, fault):
    if fault == "phase":
        plan["execution_phase"] = "diagonal"
    elif fault == "parent":
        plan["parent_plan_sha256"] = "changed"
    elif fault == "helper":
        plan["harness_sha256"].pop(crossed.SELF.name)
    elif fault == "prompt":
        plan["prompt_text"][plan["jobs"][0]["prompt"]] += " changed"
    elif fault == "raw":
        plan["assets"]["sully"]["local_path"] = "wrong-control"
    elif fault == "missing":
        plan["jobs"].pop()
    elif fault == "cases":
        plan["cases"][0]["pair"] = "sully_combat"
    else:
        plan["jobs"][0]["seed"] += 1
    with pytest.raises(ValueError):
        crossed.validate_plan(plan, verify_files=False)


def test_prepare_never_replaces_existing_plan(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        crossed.prepare(path, [])
    assert path.read_text() == "original"


def test_extension_is_validated_before_the_first_plan_save(plan, tmp_path, monkeypatch):
    study = crossed.engine()
    observed = []
    def prepare_private(path, records):
        assert records == ["synthetic-record"]
        value = copy.deepcopy(plan)
        for key in ("execution_phase", "parent_plan_path", "parent_plan_sha256"):
            value.pop(key)
        value["harness_sha256"].pop(crossed.SELF.name)
        study.save_new(path, value)
    def validate_private(value, **kwargs):
        assert value["execution_phase"] == "crossed_pairs"
        assert value["harness_sha256"][crossed.SELF.name] == crossed.digest(crossed.SELF)
        assert not destination.exists()
        observed.append("validated before save")
    monkeypatch.setattr(study, "prepare", prepare_private)
    monkeypatch.setattr(crossed, "engine", lambda: study)
    monkeypatch.setattr(crossed, "validate_plan", validate_private)
    destination = tmp_path / "new-plan.json"
    crossed.prepare(destination, ["synthetic-record"])
    assert observed == ["validated before save"]
    assert json.loads(destination.read_text())["execution_phase"] == "crossed_pairs"
