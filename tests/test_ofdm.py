import numpy as np
import pytest

from wifi_emulator.ofdm import (
    CYCLIC_PREFIX,
    DATA_CARRIERS,
    FFT_SIZE,
    OCCUPIED_CARRIERS,
    PILOT_CARRIERS,
    PILOT_VALUES,
    frequency_domain_symbols,
    generate_ofdm_symbols,
    qpsk_symbols,
    symbols_from_carriers,
)


def test_carrier_layout_dc_guards_and_pilots():
    bins = frequency_domain_symbols(5, np.random.default_rng(42))
    assert bins.shape == (5, 64)
    assert len(OCCUPIED_CARRIERS) == 52
    assert len(DATA_CARRIERS) == 48
    assert len(PILOT_CARRIERS) == 4
    active = np.asarray(OCCUPIED_CARRIERS) % FFT_SIZE
    guards = np.setdiff1d(np.arange(FFT_SIZE), active)
    assert np.all(bins[:, 0] == 0)
    assert np.all(bins[:, guards] == 0)
    assert np.all(np.count_nonzero(bins, axis=1) == 52)
    np.testing.assert_array_equal(
        bins[:, np.asarray(PILOT_CARRIERS) % FFT_SIZE], np.tile(PILOT_VALUES, (5, 1))
    )


def test_cyclic_prefix_and_frequency_recovery():
    bins = frequency_domain_symbols(7, np.random.default_rng(1))
    symbols = symbols_from_carriers(bins)
    assert symbols.shape == (7, FFT_SIZE + CYCLIC_PREFIX)
    assert symbols.dtype == np.complex64
    np.testing.assert_array_equal(symbols[:, :16], symbols[:, -16:])
    recovered = np.fft.fft(symbols[:, CYCLIC_PREFIX:], axis=-1) / np.sqrt(FFT_SIZE)
    np.testing.assert_allclose(recovered, bins, atol=3e-7)


@pytest.mark.parametrize("carrier", [-26, -5, 5, 26])
def test_signed_carriers_have_correct_frequency_orientation(carrier):
    bins = np.zeros(FFT_SIZE, dtype=np.complex64)
    bins[carrier % FFT_SIZE] = 1
    useful = symbols_from_carriers(bins)[CYCLIC_PREFIX:]
    expected = np.exp(2j * np.pi * carrier * np.arange(FFT_SIZE) / FFT_SIZE) / 8
    np.testing.assert_allclose(useful, expected, atol=1e-8)


def test_qpsk_bit_mapping_and_unit_average_power():
    class FixedBits:
        def integers(self, *args, **kwargs):
            return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)

    values = qpsk_symbols(4, FixedBits())
    expected = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    np.testing.assert_allclose(values, expected, atol=1e-7)
    random_values = qpsk_symbols(2048, np.random.default_rng(12))
    np.testing.assert_allclose(np.abs(random_values) ** 2, 1, atol=2e-7)
    assert len(np.unique(random_values)) == 4


def test_ofdm_generation_reproducible_and_empty_count_supported():
    first = generate_ofdm_symbols(20, np.random.default_rng(2))
    second = generate_ofdm_symbols(20, np.random.default_rng(2))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20 * 80,)
    assert generate_ofdm_symbols(0, np.random.default_rng(2)).size == 0


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_symbol_count(count):
    with pytest.raises(ValueError):
        frequency_domain_symbols(count, np.random.default_rng(1))
