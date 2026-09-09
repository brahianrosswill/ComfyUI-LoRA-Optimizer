import pytest

from scripts.h3_synchformer_coverage import covered_indices, coverage_fraction


def test_real_mapping_retains_115_unique_frames_not_224_independent_frames():
    mapping = [round(i * 24 / 25) for i in range(125)]
    ranges = [[2 + 8 * i, 18 + 8 * i] for i in range(14)]
    result = covered_indices(mapping, ranges, 124)
    assert len(result["target_slots"]) == 120
    assert result["source_frames"] == list(range(2, 117))
    assert result["excluded_source_frames"] == [0, 1, 117, 118, 119, 120, 121, 122, 123]


def test_frame_area_is_not_whole_image():
    assert coverage_fraction(426, 256, [101, 16, 224, 224]) == pytest.approx(.460093896713615)


@pytest.mark.parametrize("mapping,ranges,n", [
    ([], [[0, 1]], 2), ([0, 1], [], 2), ([0, -1], [[0, 2]], 2),
    ([0, 2], [[0, 2]], 2), ([1, 0], [[0, 2]], 2), ([0, 1], [[1, 1]], 2),
    ([0, 1], [[0, 3]], 2), ([0, True], [[0, 2]], 2)])
def test_invalid_temporal_mapping_fails(mapping, ranges, n):
    with pytest.raises(ValueError):
        covered_indices(mapping, ranges, n)


@pytest.mark.parametrize("crop", [[-1, 0, 2, 2], [0, 0, 0, 2], [0, 0, 427, 256], [0., 0, 2, 2]])
def test_invalid_crop_fails(crop):
    with pytest.raises(ValueError):
        coverage_fraction(426, 256, crop)
