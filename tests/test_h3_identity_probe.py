"""Similarity math tests, not validation against human judgments."""
import numpy as np
import pytest

from scripts import h3_identity_probe as probe


def test_reference_roi_uses_patch_centers_without_cropping_or_relocation():
    mask = probe.patch_mask([0, 0, .5, .5], 4).reshape(4, 4)
    assert mask[:2, :2].all()
    assert mask.sum() == 4
    assert probe.patch_mask(probe.ROIS["sully"], 32).sum() > 0
    with pytest.raises(ValueError, match="Invalid normalized"):
        probe.patch_mask([.8, 0, .2, 1], 32)
    with pytest.raises(ValueError, match="no patches"):
        probe.patch_mask([0, 0, .001, .001], 32)


def test_patch_similarity_ignores_output_location_not_reference_region():
    ref = np.eye(4, dtype=np.float32)
    mask = np.array([True, True, False, False])
    assert probe.similarities(ref, ref[::-1].copy(), mask) == pytest.approx(1.)
    different = np.repeat(ref[2:3], 4, axis=0)
    assert probe.similarities(ref, different, mask) == pytest.approx(0.)
    assert probe.similarities(ref, ref * 7., mask) == pytest.approx(1.)


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "zero", "shape"])
def test_bad_features_and_missing_regions_remain_explicit_failures(kind):
    ref = np.eye(4, dtype=np.float32)
    output = ref.copy()
    mask = np.ones(4, dtype=bool)
    if kind == "empty":
        mask[:] = False
    elif kind == "nonfinite":
        output[0, 0] = np.nan
    elif kind == "zero":
        output[:] = 0
    else:
        output = output[:3]
    with pytest.raises(ValueError):
        probe.similarities(ref, output, mask)


def test_sampling_and_preprocessor_are_explicit_not_library_defaults():
    assert probe.FRAMES == (0, 24, 48, 72, 96, 120)
    assert probe.PROCESSOR["size"] == dict(height=512, width=512)
    assert probe.PROCESSOR["resample"] == 2
    assert probe.PROCESSOR["image_mean"] == probe.PROCESSOR["image_std"] == [.5, .5, .5]
