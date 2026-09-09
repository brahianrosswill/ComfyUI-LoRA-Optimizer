"""Control admission and temporal bookkeeping, not perceptual validation."""
import copy
import math
import pytest

from scripts.h3_synchformer import control_gates, distribution, nearest_frame_map


def rows():
    result = []
    for case in ("A05", "A09"):
        for condition in ("original", "muted", "delay750"):
            logits = [0.] * 21
            logits[6 if condition == "delay750" else 10] = 5.
            silent = condition == "muted"
            result.append({"case_id": case, "condition": condition, "repeat": 0, "status": "ok",
                           "input": {"audio_peak": 0. if silent else .6},
                           "prediction": distribution(logits, silent)})
    result += [dict(copy.deepcopy(r), repeat=1) for r in result[:3]]
    return result


def test_necessary_controls_never_qualify_av_quality():
    gates = control_gates(rows())
    assert gates["necessary_controls_pass"]
    assert gates["modal_offset_delta_seconds"] == -.8
    assert not gates["qualified_for_av_quality"]
    assert not gates["qualified_for_heldout"]


def test_equal_delay_prediction_fails():
    values = rows()
    for i in (2, 8):
        values[i]["prediction"] = copy.deepcopy(values[0]["prediction"])
    gates = control_gates(values)
    assert not gates["direction_gate"] and not gates["ranking_gate"]
    assert not gates["necessary_controls_pass"]


def test_wrong_sign_cannot_pass():
    values = rows()
    for i in (2, 8):
        logits = [0.] * 21
        logits[14] = 5.
        values[i]["prediction"] = distribution(logits)
    assert not control_gates(values)["direction_gate"]


def test_silence_abstains_even_with_confident_zero_offset():
    logits = [0.] * 21
    logits[10] = 100.
    d = distribution(logits, silent=True)
    assert d["raw_near_zero_mass"] > .999
    assert not d["assessable"]
    assert d["reported_offset_seconds"] is None


@pytest.mark.parametrize("bad", [[0.] * 20, [float("nan")] * 21, [float("inf")] * 21, [True] * 21])
def test_invalid_logits_fail(bad):
    with pytest.raises(ValueError):
        distribution(bad)


@pytest.mark.parametrize("mutation", [lambda r: r.pop(), lambda r: r.append(r[0]),
                                     lambda r: r[0].update(case_id="heldout")])
def test_wrong_scope_fails(mutation):
    values = rows()
    mutation(values)
    with pytest.raises(ValueError):
        control_gates(values)


def test_inference_failure_cannot_pass():
    values = rows()
    values[0]["status"] = "error"
    assert not control_gates(values)["necessary_controls_pass"]


def test_accidentally_silent_original_cannot_pass():
    values = rows()
    values[0]["input"]["audio_peak"] = 0.
    assert not control_gates(values)["technical_gate"]


def test_prediction_tampering_fails():
    values = rows()
    values[0]["prediction"]["raw_near_zero_mass"] = 1.
    with pytest.raises(ValueError):
        control_gates(values)


def test_probability_change_fails_repeatability():
    values = rows()
    logits = list(values[-1]["prediction"]["logits"])
    logits[6] = 4.
    values[-1]["prediction"] = distribution(logits)
    assert not control_gates(values)["repeatability_gate"]


def test_real_geometry_mapping_has_no_accumulating_drift():
    pts = [i / 24 for i in range(124)]
    indices = nearest_frame_map(pts)
    assert len(indices) == 125
    assert indices[0] == 0 and indices[-1] == 119
    assert max(abs(pts[j] - i / 25) for i, j in enumerate(indices)) <= 1 / 48
    assert all(a <= b for a, b in zip(indices, indices[1:]))


@pytest.mark.parametrize("pts", [[0., 0.], [0., float("nan")], [.1, .2], [0., .5, .4], [0., 1.]])
def test_invalid_or_short_pts_rejected(pts):
    with pytest.raises(ValueError):
        nearest_frame_map(pts)


def test_nearest_mapping_uses_pts_not_assumed_source_fps():
    assert nearest_frame_map([0., .03, .09, .12], count=4, fps=25) == [0, 1, 2, 3]
    assert math.isclose(sum(distribution([0.] * 21)["probabilities"]), 1.)
