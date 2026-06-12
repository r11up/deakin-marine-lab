"""Tests for image preprocessing utilities."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from furseal.data.balance import augment_image
from furseal.segmentation.morphology import morphological_mask


def _random_bgr_image(h=64, w=64, seed=0):
    return np.random.RandomState(seed).randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_augment_image_returns_four_variants():
    img = _random_bgr_image()
    variants = augment_image(img)
    assert len(variants) == 4
    for v in variants:
        assert v.dtype == np.uint8
        assert v.ndim == 3


def test_morphological_mask_is_binary_grayscale():
    img = _random_bgr_image()
    mask = morphological_mask(img)
    assert mask.ndim == 2
    assert mask.shape == img.shape[:2]
    # Thresholded to {0, maxval}
    assert set(np.unique(mask)).issubset({0, 100})


def test_preprocess_image_flattens_and_normalises(tmp_path):
    from furseal.data.resize import preprocess_image

    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), _random_bgr_image(32, 32))

    vec = preprocess_image(img_path, size=(4, 4))
    assert vec.shape == (16,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0
