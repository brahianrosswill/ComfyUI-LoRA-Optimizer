import pytest

from scripts.h3_fate import base_mapping, validate_alignment


def test_strict_base_prefix_mapping_keeps_only_required_architecture():
    header = {"audio_model.audio_encoder.x": {"shape": [2, 3]},
              "video_model.video_head.y": {"shape": [4]},
              "text_model.unused": {"shape": [10]}}
    assert base_mapping(header, {"audio_encoder.x": [2, 3], "video_head.y": [4]}) == {
        "audio_encoder.x": "audio_model.audio_encoder.x", "video_head.y": "video_model.video_head.y"}


@pytest.mark.parametrize("header,expected", [
    ({}, {}), ({}, {"audio_encoder.x": [1]}),
    ({"audio_model.audio_encoder.x": {"shape": [2]}}, {"audio_encoder.x": [1]}),
    ({"audio_model.audio_encoder.x": {"shape": [1]}}, {"audio_encoder.y": [1]}),
    ({"wrong.audio_encoder.x": {"shape": [1]}}, {"audio_encoder.x": [1]})])
def test_missing_mismatched_empty_and_wrong_prefix_fail(header, expected):
    with pytest.raises(ValueError):
        base_mapping(header, expected)


def test_unpadded_alignment_is_supported():
    torch = pytest.importorskip("torch")
    validate_alignment(torch.zeros(1, 48, 4), torch.zeros(1, 50, 4), None, torch.ones(1, 50))


@pytest.mark.parametrize("vshape,ashape,vmask,amask", [
    ((1, 6, 4), (1, 3, 4), [[1, 1, 1, 0, 0, 0]], [[1, 1, 1]]),
    ((1, 3, 4), (1, 3, 4), [[1, 0, 0]], [[1, 1, 1]]),
    ((1, 3, 4), (1, 4, 4), None, [[1, 1, 1]]),
    ((2, 3, 4), (2, 4, 4), None, None),
    ((1, 3, 4), (1, 4, 5), None, None),
    ((1, 0, 4), (1, 4, 4), None, None),
    ((1, 3, 4), (1, 4, 4), None, [[1, 1, float('nan'), 1]])])
def test_alignment_edge_cases_fail_closed(vshape, ashape, vmask, amask):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        validate_alignment(torch.zeros(vshape), torch.zeros(ashape),
                           torch.tensor(vmask) if vmask is not None else None,
                           torch.tensor(amask) if amask is not None else None)
