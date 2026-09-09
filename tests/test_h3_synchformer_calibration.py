import pytest

from scripts.h3_synchformer_calibration import pairwise


@pytest.mark.parametrize("human,model,relation", [
    ([1, 4], [.1, .9], "concordant"), ([4, 1], [.1, .9], "discordant"),
    ([1, 4], [.5, .5], "model_tied"), ([3, 3], [.1, .9], "human_tied"),
    ([3, 3], [.5, .5], "both_tied"), ([None, 3], [.1, .9], "missing"),
    ([1, 3], [None, .9], "missing")])
def test_all_comparison_outcomes_remain_explicit(human, model, relation):
    rows = [{"blind_id": str(i), "human_sync": h, "metric": m} for i, (h, m) in enumerate(zip(human, model))]
    report = pairwise(rows, "metric")
    assert report["counts"][relation] == 1
    if relation in ("human_tied", "both_tied", "missing"):
        assert report["tie_adjusted_descriptive_fraction"] is None
    elif relation == "model_tied":
        assert report["tie_adjusted_descriptive_fraction"] == .5


def test_seven_entries_have_21_comparisons():
    rows = [{"blind_id": str(i), "human_sync": i % 5, "metric": .5} for i in range(7)]
    report = pairwise(rows, "metric")
    assert len(report["comparisons"]) == 21
    assert sum(report["counts"].values()) == 21


def test_nonfinite_is_not_missing():
    with pytest.raises(ValueError):
        pairwise([{"blind_id": "a", "human_sync": 3, "metric": float("nan")},
                  {"blind_id": "b", "human_sync": 4, "metric": .1}], "metric")
