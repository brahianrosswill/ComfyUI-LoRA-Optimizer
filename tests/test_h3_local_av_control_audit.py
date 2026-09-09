"""Synthetic admission tests; never substitute these for real AV judgments."""
import copy
import pytest

from scripts.h3_local_av_control_audit import control_gates


def records():
    response = dict(audible_sound=True, speech_present=False, music_present=False,
                    **{k + "_score": 3 for k in ("action", "temporal", "appearance", "audio", "synchronization", "overall")})
    result = []
    for case in ("A", "B"):
        for condition in ("original", "muted", "delay750"):
            row = dict(case_id=case, condition=condition, repeat=0, status="valid_response", response=copy.deepcopy(response))
            if condition == "muted":
                row["response"]["audible_sound"] = False
            result.append(row)
    result.extend(dict(copy.deepcopy(r), repeat=1) for r in result[:2])
    return result


def test_equal_scores_for_delayed_audio_cannot_pass():
    report = control_gates(records(), "A")
    assert report["all_structured_responses_valid"]
    assert report["silence_boolean_gate"] and report["categorical_and_score_repeatability_gate"]
    assert not report["synchronization_direction_gate"]
    assert not report["necessary_numeric_gates_pass"]
    assert not report["qualified_for_quality_ranking"]


def test_necessary_controls_do_not_self_certify_quality():
    rows = records()
    rows[2]["response"]["synchronization_score"] = 1
    report = control_gates(rows, "A")
    assert report["necessary_numeric_gates_pass"]
    assert report["semantic_evidence_review_required"]
    assert not report["qualified_for_quality_ranking"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
    lambda r: r[0].update(case_id="heldout"),
])
def test_incomplete_duplicate_or_wrong_scope_fails(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        control_gates(rows, "A")


@pytest.mark.parametrize("mutation,gate", [
    (lambda r: r[1]["response"].update(audible_sound=True), "silence_boolean_gate"),
    (lambda r: r[2]["response"].update(synchronization_score=None), "synchronization_direction_gate"),
    (lambda r: r[-1]["response"].update(audio_score=0), "categorical_and_score_repeatability_gate"),
    (lambda r: r[0].update(status="invalid_response"), "all_structured_responses_valid"),
])
def test_missing_invalid_or_failed_controls_remain_failures(mutation, gate):
    rows = records()
    mutation(rows)
    assert not control_gates(rows, "A")[gate]
