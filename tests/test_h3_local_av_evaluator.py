"""Synthetic harness tests, not audiovisual quality evidence."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import h3_local_av_evaluator as judge
from scripts.h3_local_av_setup import save_new


def response():
    return dict(audible_sound=True, speech_present=False, music_present=False,
                sound_events=[dict(start_seconds=1.0, end_seconds=1.2, description="A dull impact")],
                visible_contact_count=None, scores={k: None for k in judge.DIMENSIONS},
                evidence={k: "Uncertain from sampled evidence" for k in judge.DIMENSIONS},
                uncertainty="Synthetic test fixture")


def test_nullable_observations_remain_missing():
    value = judge.validate_response(json.dumps(response()))
    assert all(v is None for v in value["scores"].values())
    assert value["visible_contact_count"] is None


def test_presentation_fence_only():
    raw = json.dumps(response())
    assert judge.validate_response("```json\n" + raw + "\n```") == response()
    with pytest.raises(ValueError):
        judge.validate_response(raw + "\nExtra explanation")


@pytest.mark.parametrize("score", [True, 1.5, -1, 5, "3", float("nan"), float("inf")])
def test_invalid_scores_are_not_clamped_or_coerced(score):
    value = response()
    value["scores"]["audio"] = score
    with pytest.raises(ValueError):
        judge.validate_response(json.dumps(value))


def test_duplicate_keys_rejected():
    raw = json.dumps(response())
    with pytest.raises(ValueError, match="Duplicate"):
        judge.validate_response(raw[:-1] + ', "audible_sound": false}')


@pytest.mark.parametrize("edit", [
    lambda v: v.update(audible_sound="yes"),
    lambda v: v.update(visible_contact_count=True),
    lambda v: v["scores"].pop("overall"),
    lambda v: v["evidence"].update(audio=""),
    lambda v: v["sound_events"][0].update(start_seconds=-1),
    lambda v: v["sound_events"][0].update(end_seconds=0.5),
    lambda v: v.update(model_name="leaked"),
])
def test_invalid_schema_is_retained_as_failure(edit):
    value = response()
    edit(value)
    with pytest.raises(ValueError):
        judge.validate_response(json.dumps(value))


def test_original_control_does_not_mutate_source():
    source = np.arange(32000, dtype=np.float32) / 32000
    original = source.copy()
    result = judge.audio_control(source, "original")
    np.testing.assert_array_equal(result, original)
    result[0] = -10
    np.testing.assert_array_equal(source, original)


def test_mute_is_exact_zero_and_delay_is_not_wrapped():
    source = np.arange(32000, dtype=np.float32) / 32000
    frozen = source.copy()
    assert np.count_nonzero(judge.audio_control(source, "muted")) == 0
    delayed = judge.audio_control(source, "delay750")
    assert len(delayed) == len(source)
    assert np.count_nonzero(delayed[:12000]) == 0
    np.testing.assert_array_equal(delayed[12000:], source[:-12000])
    np.testing.assert_array_equal(source, frozen)


@pytest.mark.parametrize("source,condition", [
    (np.ones((2, 3)), "original"), (np.array([]), "original"),
    (np.array([float("nan")]), "muted"), (np.ones(100), "delay750"),
    (np.ones(32000), "unknown"),
])
def test_bad_control_inputs_fail(source, condition):
    with pytest.raises(ValueError):
        judge.audio_control(source, condition)


def test_endpoint_rate_prevents_cumulative_timing_drift():
    indices = np.linspace(0, 123, 20).round().astype(int).tolist()
    effective = judge.effective_video_fps(indices, 24.0)
    assert (len(indices) - 1) / effective == pytest.approx(123 / 24)
    errors = [abs(i / 24 - n / effective) for n, i in enumerate(indices)]
    assert max(errors) < 1 / 24
    old_fps = len(indices) / 124 * 24
    assert abs(19 / old_fps - 123 / 24) > 0.2


@pytest.mark.parametrize("indices,fps", [([0], 24), ([0, 0, 2], 24), ([1, 2], 24),
                                        ([0, 1], 0), ([0, 1], float("nan"))])
def test_invalid_time_grids_fail(indices, fps):
    with pytest.raises(ValueError):
        judge.effective_video_fps(indices, fps)


def test_records_cannot_be_silently_replaced(tmp_path):
    path = tmp_path / "record.json"
    save_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        save_new(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_fixed_prompt_does_not_contain_methods_or_r1_labels():
    text = judge.SYSTEM + judge.INSTRUCTION
    for forbidden in ("NP-LoRA", "CT-Merging", "combat-cinema", "R1", "out of sync audio"):
        assert forbidden not in text


@pytest.mark.parametrize("queue", [
    {"queue_running": [1], "queue_pending": []},
    {"queue_running": [], "queue_pending": [1]},
    {"queue_pending": []}, {"queue_running": None, "queue_pending": []},
])
def test_existing_render_work_or_unknown_queue_stops_evaluator(monkeypatch, queue):
    monkeypatch.setattr(judge, "urlopen", lambda *a, **k: io.StringIO(json.dumps(queue)))
    with pytest.raises((ValueError, RuntimeError)):
        judge.check_generation_queue()


def test_verified_idle_queue_does_not_write_or_queue_work(monkeypatch):
    queue = {"queue_running": [], "queue_pending": []}
    monkeypatch.setattr(judge, "urlopen", lambda *a, **k: io.StringIO(json.dumps(queue)))
    assert judge.check_generation_queue() == queue


def test_stopping_ids_come_from_tokenizer_not_empty_generation_file():
    tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    assert judge.stopping_ids(tokenizer) == {"eos_token_id": 151645, "pad_token_id": 151643}


@pytest.mark.parametrize("bad", [None, True, -1, "151645"])
def test_missing_or_invalid_stop_token_rejected(bad):
    with pytest.raises(ValueError):
        judge.stopping_ids(SimpleNamespace(eos_token_id=bad, pad_token_id=151643))


def flat_response():
    return dict(observed_video="Synthetic example", observed_audio="Silence", sync_evidence="Unassessable",
                uncertainty="Synthetic test only", audible_sound=False, speech_present=False, music_present=False,
                **{k + "_score": None for k in judge.DIMENSIONS})


def test_flat_profile_preserves_unknown_scores_and_missing_audio():
    assert judge.validate_flat_response(json.dumps(flat_response())) == flat_response()
    assert judge.evaluation_profile("joint_flat")[4] is judge.validate_flat_response
    assert judge.evaluation_profile("legacy")[4] is judge.validate_response


@pytest.mark.parametrize("edit", [
    lambda v: v.update(audio_score=True), lambda v: v.update(synchronization_score=5),
    lambda v: v.update(appearance_score=float("nan")), lambda v: v.update(audible_sound="false"),
    lambda v: v.pop("overall_score"), lambda v: v.update(sync_evidence=""),
])
def test_invalid_flat_ratings_fail(edit):
    value = flat_response()
    edit(value)
    with pytest.raises(ValueError):
        judge.validate_flat_response(json.dumps(value))
