import json

import pytest

from scripts import h3_character_similarity as diagonal
from scripts import h3_crossed_character_similarity as crossed


def test_private_engine_keeps_original_pairs_and_diagnostic_formula():
    metric = crossed.engine()
    assert metric.PAIRS == ("series30_combat", "sully_cinema")
    assert diagonal.PAIRS == ("series30_cinema", "sully_combat")
    assert metric.probe is diagonal.probe
    assert metric.SEEDS == diagonal.SEEDS
    rows = [dict(job=dict(pair="sully_cinema", seed=2026090831, variant=arm),
        median_head_maxcos=.7 if arm == "character_only" else .6,
        median_global_cosine=.5) for arm in diagonal.ARMS]
    assert metric.matched_deltas(rows) == diagonal.matched_deltas(rows)


def test_calibration_matrix_selects_both_crossed_pairs_not_diagonal_or_heldout():
    parent = crossed.parent_recipe()
    plan = dict(jobs=[])
    for job in parent["jobs"]:
        new = dict(job)
        new["pair"] = {"series30_cinema": "series30_combat", "sully_combat": "sully_cinema"}[job["pair"]]
        new["id"] += "-crossed-test"
        plan["jobs"].append(new)
    chosen = crossed.engine().calibration_jobs(plan)
    assert len(chosen) == 36
    assert {j["pair"] for j in chosen} == set(crossed.crossed.PAIRS)
    with pytest.raises(ValueError):
        diagonal.calibration_jobs(plan)


def test_diagonal_recipe_cannot_be_used_as_crossed_recipe():
    with pytest.raises(ValueError, match="crossed calibration"):
        crossed.validate_scope(json.loads(crossed.PARENT_RECIPE.read_text()))


@pytest.mark.parametrize("seed", [2026090841, 2026090842])
def test_heldout_is_rejected_before_reading_files_or_loading_models(tmp_path, seed):
    with pytest.raises(ValueError, match="calibration"):
        crossed.evaluate(tmp_path / "missing.json", seed, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_recipe_is_never_overwritten(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        crossed.prepare(path, tmp_path / "missing-plan.json")
    assert path.read_text() == "original"
