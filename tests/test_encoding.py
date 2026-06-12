"""Tests for classical-to-quantum data encoding (no qiskit required)."""

import numpy as np
import pytest

from furseal.quantum.encoding import amplitude_encode


def test_amplitude_encode_unit_norm():
    vec = np.random.RandomState(0).rand(200)
    encoded = amplitude_encode(vec, max_dim=128)
    assert encoded.shape == (128,)
    assert np.isclose(np.linalg.norm(encoded), 1.0)


def test_amplitude_encode_short_vector_keeps_length():
    vec = np.ones(10)
    encoded = amplitude_encode(vec, max_dim=128)
    assert encoded.shape == (10,)
    assert np.isclose(np.linalg.norm(encoded), 1.0)


def test_amplitude_encode_zero_vector_raises():
    with pytest.raises(ValueError):
        amplitude_encode(np.zeros(16), max_dim=16)
