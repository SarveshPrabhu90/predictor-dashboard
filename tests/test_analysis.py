"""Unit tests for the analysis module (no TCP server required)."""

import numpy as np
import pytest

from app.analysis import (
    ground_truth,
    YIELD_W, YIELD_I, PURITY_W, PURITY_I,
    INPUT_NAMES, OUTPUT_NAMES, INPUT_RANGES,
    _fig_to_base64,
)


class TestGroundTruth:
    def test_shape(self):
        inp = np.random.rand(10, 3)
        out = ground_truth(inp)
        assert out.shape == (10, 2)

    def test_known_values(self):
        inp = np.array([[50.0, 5.0, 2.5]])
        out = ground_truth(inp)
        expected_y = 50 * 0.45 + 5 * 0.30 + 2.5 * 0.80 + 12
        expected_p = 50 * (-0.15) + 5 * 0.55 + 2.5 * 0.35 + 85
        assert out[0, 0] == pytest.approx(expected_y)
        assert out[0, 1] == pytest.approx(expected_p)

    def test_deterministic(self):
        inp = np.array([[30.0, 3.0, 1.0]])
        out1 = ground_truth(inp)
        out2 = ground_truth(inp)
        np.testing.assert_array_equal(out1, out2)

    def test_batch(self):
        inp = np.random.rand(100, 3) * [60, 9, 5] + [20, 1, 0.1]
        out = ground_truth(inp)
        assert out.shape == (100, 2)
        assert np.all(np.isfinite(out))


class TestConstants:
    def test_coefficient_lengths(self):
        assert len(YIELD_W) == 3
        assert len(PURITY_W) == 3

    def test_input_names(self):
        assert INPUT_NAMES == ["temperature", "flow_rate", "concentration"]

    def test_output_names(self):
        assert OUTPUT_NAMES == ["yield", "purity"]

    def test_input_ranges(self):
        assert len(INPUT_RANGES) == 3
        for lo, hi in INPUT_RANGES:
            assert lo < hi


class TestFigToBase64:
    def test_produces_string(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        encoded = _fig_to_base64(fig)
        assert isinstance(encoded, str)
        assert len(encoded) > 100  # should be a non-trivial PNG

    def test_valid_base64(self):
        import base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        encoded = _fig_to_base64(fig)
        decoded = base64.b64decode(encoded)
        assert decoded[:4] == b"\x89PNG"
