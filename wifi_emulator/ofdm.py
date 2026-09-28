"""Simplified legacy Wi-Fi OFDM, with explicit signed carrier indexing.

Mathematical carrier k has frequency k * SAMPLE_RATE / FFT_SIZE. NumPy's
unshifted IFFT input stores it at index k % FFT_SIZE: positive carriers occupy
1..26, negative carriers occupy 38..63, and DC is index 0. No fftshift or
ifftshift is needed for modulation. fftshift is used only for plotting.
This is an energy-test waveform, not an interoperable IEEE 802.11 PHY.
"""

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 20_000_000
FFT_SIZE = 64
CYCLIC_PREFIX = 16
SYMBOL_SAMPLES = FFT_SIZE + CYCLIC_PREFIX
SYMBOL_DURATION_US = SYMBOL_SAMPLES / SAMPLE_RATE * 1e6
OCCUPIED_CARRIERS = tuple(range(-26, 0)) + tuple(range(1, 27))
PILOT_CARRIERS = (-21, -7, 7, 21)
DATA_CARRIERS = tuple(k for k in OCCUPIED_CARRIERS if k not in PILOT_CARRIERS)
PILOT_VALUES = (1, 1, 1, -1)


def qpsk_symbols(count: int, rng: np.random.Generator) -> NDArray[np.complex64]:
    """Map random bit pairs 00, 01, 11, 10 to NE, NW, SW, SE respectively."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("QPSK symbol count must be a nonnegative integer")
    bits = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
    values = ((1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0])) / np.sqrt(2)
    return values.astype(np.complex64)


def frequency_domain_symbols(
    count: int, rng: np.random.Generator
) -> NDArray[np.complex64]:
    """Return (count, 64) frequency bins in native NumPy IFFT order."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("OFDM symbol count must be a nonnegative integer")
    bins = np.zeros((count, FFT_SIZE), dtype=np.complex64)
    data_indices = np.asarray(DATA_CARRIERS) % FFT_SIZE
    pilot_indices = np.asarray(PILOT_CARRIERS) % FFT_SIZE
    bins[:, data_indices] = qpsk_symbols(count * len(DATA_CARRIERS), rng).reshape(
        count, len(DATA_CARRIERS)
    )
    bins[:, pilot_indices] = PILOT_VALUES
    return bins


def symbols_from_carriers(carriers: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
    """IFFT the last axis, scale to useful power, and prepend its last 16 samples.

The sqrt(64) scale makes the IFFT unitary; a fully populated 52-carrier symbol
therefore has useful-part mean power 52/64 before packet/burst normalization.
Leading axes are preserved, so a (n, 64) input yields (n, 80).
"""
    bins = np.asarray(carriers)
    if bins.ndim < 1 or bins.shape[-1] != FFT_SIZE:
        raise ValueError("The last carrier axis must contain exactly 64 FFT bins")
    if not np.all(np.isfinite(bins)):
        raise ValueError("Carrier values must be finite")
    useful = np.fft.ifft(bins, axis=-1) * np.sqrt(FFT_SIZE)
    return np.concatenate((useful[..., -CYCLIC_PREFIX:], useful), axis=-1).astype(
        np.complex64
    )


def generate_ofdm_symbols(count: int, rng: np.random.Generator) -> NDArray[np.complex64]:
    """Return concatenated, cyclic-prefixed random QPSK OFDM symbols."""
    return symbols_from_carriers(frequency_domain_symbols(count, rng)).reshape(-1)
