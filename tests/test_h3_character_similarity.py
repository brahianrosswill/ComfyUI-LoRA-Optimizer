import copy
import json
from pathlib import Path

import pytest

from scripts import h3_character_similarity as metric


def test_calibration_selection_keeps_both_complete_blocks_and_no_holdout():
    path = Path(__file__).parents[1] / "docs/research/data/2026-09-08-h3-character-benchmark-plan.json"
    plan = json.loads(path.read_text())
    jobs = metric.calibration_jobs(plan)
    assert len(jobs) == 36
    assert {j["seed"] for j in jobs} == set(metric.SEEDS)
    assert {j["stage"] for j in jobs} == {"calibration"}
    changed = copy.deepcopy(plan)
    changed["jobs"].remove(jobs[0])
    with pytest.raises(ValueError, match="36"):
        metric.calibration_jobs(changed)


def fixture_rows():
    return [dict(job=dict(pair="sully_combat", seed=31, variant=arm),
        median_head_maxcos=.8 if arm == "character_only" else .5 if arm == "additive" else .6,
        median_global_cosine=.7) for arm in metric.ARMS]


def test_matched_differences_keep_sign_without_percentage_normalization():
    out = {r["variant"]: r for r in metric.matched_deltas(fixture_rows())}
    assert out["character_only"]["delta_to_character_only"] == 0.
    assert out["additive"]["delta_to_additive"] == 0.
    assert out["ct_merge"]["delta_to_character_only"] == pytest.approx(-.2)
    assert out["ct_merge"]["delta_to_additive"] == pytest.approx(.1)
    assert set(out) == set(metric.ARMS)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "bool"])
def test_missing_invalid_and_duplicated_controls_are_not_silently_skipped(fault):
    rows = fixture_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["median_head_maxcos"] = float("nan") if fault == "nan" else True
    with pytest.raises(ValueError):
        metric.matched_deltas(rows)


def test_heldout_evaluation_is_rejected_before_reading_or_loading(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        metric.evaluate(tmp_path / "nonexistent-recipe.json", 2026090841, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
