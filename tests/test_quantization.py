import numpy as np
import pytest

from wifi_emulator.quantization import normalize_iq, quantize_iq


def test_signed_iq_order_and_full_scale():
    result = quantize_iq([1, 1j, -1, -1j], digital_amplitude=1)
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result, [127, 0, 0, 127, -127, 0, 0, -127])


def test_complex_peak_normalization_and_rounding():
    source = np.array([3 + 4j, 0, -3 - 4j])
    normalized = normalize_iq(source)
    assert normalized.dtype == np.complex64
    assert np.max(np.abs(normalized)) == pytest.approx(0.6)
    np.testing.assert_allclose(normalized, [0.36 + 0.48j, 0, -0.36 - 0.48j])
    np.testing.assert_array_equal(source, [3 + 4j, 0, -3 - 4j])
    np.testing.assert_array_equal(quantize_iq(source), [46, 61, 0, 0, -46, -61])


def test_silence_empty_input_and_large_random_iq():
    np.testing.assert_array_equal(quantize_iq(np.zeros(10)), np.zeros(20, dtype=np.int8))
    assert quantize_iq([]).size == 0
    rng = np.random.default_rng(1)
    output = quantize_iq(1000 * (rng.normal(size=10000) + 1j * rng.normal(size=10000)), 1)
    assert output.min() >= -127
    assert output.max() <= 127
    assert output.size == 20000


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_finite_scales_remain_normalizable(scale):
    np.testing.assert_allclose(normalize_iq([scale * (1 + 1j)], 1), [(1 + 1j) / np.sqrt(2)])


@pytest.mark.parametrize("amplitude", [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_digital_amplitude(amplitude):
    with pytest.raises(ValueError):
        quantize_iq([1 + 1j], amplitude)


@pytest.mark.parametrize("iq", [[np.nan], [np.inf], [1 + np.inf * 1j], [[1, 2]], 1])
def test_invalid_iq(iq):
    with pytest.raises(ValueError):
        normalize_iq(iq)
