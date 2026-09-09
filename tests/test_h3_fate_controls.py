import copy

import pytest

from scripts.h3_fate_controls import CONDITIONS, controlled_audio, gates


@pytest.mark.parametrize("condition,n,direction", [
    ("delay750", 36000, 1), ("advance750", 36000, -1),
    ("delay125", 6000, 1), ("advance125", 6000, -1),
    ("delay80", 3840, 1), ("advance80", 3840, -1)])
def test_controls_shift_without_wrap_or_resampling(condition, n, direction):
    np = pytest.importorskip("numpy")
    x = np.arange(240000, dtype=np.float32)
    y = controlled_audio(x, condition)
    assert y.shape == x.shape and y.dtype == np.float32
    if direction == 1:
        assert np.array_equal(y[n:], x[:-n]) and np.count_nonzero(y[:n]) == 0
    else:
        assert np.array_equal(y[:-n], x[n:]) and np.count_nonzero(y[-n:]) == 0


def test_original_and_silence_do_not_mutate_input():
    np = pytest.importorskip("numpy")
    x = np.ones(240000, dtype=np.float32)
    y = controlled_audio(x, "original")
    y[0] = 0
    assert x[0] == 1
    assert not controlled_audio(x, "muted").any()


def test_tonal_control_matches_rms_is_finite_and_repeatable():
    np = pytest.importorskip("numpy")
    x = np.random.default_rng(42).normal(0, .1, 240000).astype(np.float32)
    a, b = [controlled_audio(x, "tone440") for _ in range(2)]
    assert len(a) == len(x) and np.isfinite(a).all() and np.array_equal(a, b)
    assert np.mean(a.astype(np.float64) ** 2) == pytest.approx(np.mean(x.astype(np.float64) ** 2), rel=1e-6)


@pytest.mark.parametrize("kind", ["short", "nonfinite", "wrong_rank", "wrong_condition", "wrong_rate", "silent_tone"])
def test_invalid_control_inputs_fail(kind):
    np = pytest.importorskip("numpy")
    x, condition, sr = np.ones(240000, dtype=np.float32), "original", 48000
    if kind == "short": x = x[:1]
    if kind == "nonfinite": x[0] = np.nan
    if kind == "wrong_rank": x = x.reshape(2, -1)
    if kind == "wrong_condition": condition = "invented"
    if kind == "wrong_rate": sr = 16000
    if kind == "silent_tone": x[:], condition = 0, "tone440"
    with pytest.raises(ValueError):
        controlled_audio(x, condition, sr)


def rows():
    result = []
    scope = [(c, k, 0) for c in ("A05", "A09") for k in CONDITIONS]
    scope += [("A05", c, 1) for c in ("original", "delay750")]
    for c, k, r in scope:
        score = .5 if k == "original" else .1
        result.append({"case_id": c, "condition": k, "repeat": r,
            "windows": [{"diagonal": [score] * 50, "score": score, "input_tensors": {"wave": k}} for _ in range(3)],
            "raw_score": score, "assessable": k != "muted", "reported_score": score if k != "muted" else None})
    return result


def test_passing_controls_never_qualify_for_quality_or_heldout():
    result = gates(rows())
    assert result["necessary_controls_pass"]
    assert not result["qualified_for_av_quality"] and not result["heldout_admitted"]


def test_bad_control_scores_block_ranking_sweep():
    values = rows()
    for row in values:
        if row["condition"] == "advance750" and row["case_id"] == "A05":
            row["raw_score"] = .6
            for window in row["windows"]:
                window.update(diagonal=[.6] * 50, score=.6)
    assert not gates(values)["necessary_controls_pass"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "nan", "short_vector", "wrong_window_score", "wrong_case_score"])
def test_corrupt_or_incomplete_records_fail(change):
    values = copy.deepcopy(rows())
    if change == "duplicate": values.append(values[0])
    if change == "missing": values.pop()
    if change == "nan": values[0]["windows"][0]["diagonal"][0] = float("nan")
    if change == "short_vector": values[0]["windows"][0]["diagonal"].pop()
    if change == "wrong_window_score": values[0]["windows"][0]["score"] = .2
    if change == "wrong_case_score": values[0]["raw_score"] = .2
    with pytest.raises(ValueError):
        gates(values)


def test_repeat_input_mismatch_blocks_even_with_equal_scores():
    values = rows()
    values[-1]["windows"][0]["input_tensors"] = {"wave": "changed"}
    assert not gates(values)["repeatability_gate"]
