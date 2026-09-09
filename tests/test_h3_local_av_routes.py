"""Synthetic route-trace bookkeeping, not model-perception tests."""
from scripts.h3_local_av_routes import token_runs


def test_route_runs_preserve_audio_video_order_and_total():
    ids = [0, 1, 7, 7, 8, 8, 8, 7, 9]
    runs = token_runs(ids, audio_id=7, video_id=8)
    assert runs == [{"kind": "other", "length": 2}, {"kind": "audio", "length": 2},
                    {"kind": "video", "length": 3}, {"kind": "audio", "length": 1},
                    {"kind": "other", "length": 1}]
    assert sum(r["length"] for r in runs) == len(ids)


def test_empty_and_audio_only_runs():
    assert token_runs([], 7, 8) == []
    assert token_runs([7, 7], 7, 8) == [{"kind": "audio", "length": 2}]
