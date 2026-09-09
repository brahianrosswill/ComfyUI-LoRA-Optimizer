"""Synthetic rating fixtures only; these are never written to study evidence."""
import pytest

from scripts.h3_av_results import summarize
from scripts.h3_av_review import DIMENSIONS


def fixture(variant, overall, *, pair="p", sha=None, stage="calibration"):
    return dict(pair=pair, variant=variant, stage=stage, prompt="synthetic", seed=1,
                job_id=pair + variant, blind_id=pair + variant, review_id="synthetic",
                original_sha256=sha or pair + variant, scores={d: overall for d in DIMENSIONS})


def test_ordinal_comparisons_preserve_ties_and_dimension_disagreement():
    rows = [fixture("additive", 2), fixture("winner", 3), fixture("np", 3), fixture("ct", 3)]
    rows[-1]["scores"]["audio"] = 1
    result = summarize(rows, rows, "calibration")
    comparisons = {(r["candidate"], r["baseline"]): r for r in result["paired_comparisons"]}
    assert comparisons["ct", "additive"]["overall_preference"] == "candidate"
    assert comparisons["ct", "winner"]["overall_preference"] == "tie"
    assert comparisons["ct", "winner"]["ordinal_grade_differences"]["audio"] == -2
    assert result["stage_fully_reviewed"]
    assert result["inferential_statistics"] is None and result["learned_ranking"] is None


def test_missing_ratings_are_not_imputed_as_failures():
    cases = [fixture("np", 3), fixture("additive", 2)]
    result = summarize(cases[:1], cases, "calibration")
    assert result["paired_comparisons"] == []
    assert result["groups"][0]["missing_variants"] == ["additive"]
    assert not result["stage_fully_reviewed"]


def test_reused_controls_are_counted_once_and_disagreement_is_retained():
    rows = [fixture("base", 2, pair="p", sha="shared"), fixture("base", 2, pair="q", sha="shared")]
    rows[1]["scores"]["temporal"] = 3
    result = summarize(rows, rows, "calibration")
    assert result["rating_entries"] == 2 and result["unique_videos"] == 1
    assert result["repeated_controls"][0]["grade_spread"]["overall"] == 0
    assert result["repeated_controls"][0]["grade_spread"]["temporal"] == 1


def test_duplicate_or_cross_split_ratings_are_rejected():
    row = fixture("ct", 3)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row, row], [row], "calibration")
    with pytest.raises(ValueError, match="mix"):
        summarize([row, fixture("np", 3, stage="heldout")], [row], "calibration")
    with pytest.raises(ValueError, match="No human"):
        summarize([], [row], "calibration")
