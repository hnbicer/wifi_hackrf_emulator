"""Peak-normalized complex IQ and HackRF signed, interleaved 8-bit samples."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_iq(iq: ArrayLike, digital_amplitude: float = 0.6) -> NDArray[np.complex64]:
    """Return a fresh vector with peak complex magnitude digital_amplitude.

Silence remains silence. Scaling by the largest component first avoids
overflow/underflow when a caller supplies unusually large/small finite IQ.
"""
    if not math.isfinite(digital_amplitude) or not 0 < digital_amplitude <= 1:
        raise ValueError("Digital amplitude must be finite and in (0, 1]")
    samples = np.asarray(iq, dtype=np.complex128)
    if samples.ndim != 1:
        raise ValueError("IQ must be a one-dimensional sample vector")
    if not np.all(np.isfinite(samples)):
        raise ValueError("IQ samples must be finite")
    if samples.size == 0:
        return np.empty(0, dtype=np.complex64)
    component_peak = max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))
    if component_peak == 0:
        return np.zeros(samples.size, dtype=np.complex64)
    scaled = samples / component_peak
    return (scaled * (digital_amplitude / np.max(np.abs(scaled)))).astype(np.complex64)


def quantize_iq(iq: ArrayLike, digital_amplitude: float = 0.6) -> NDArray[np.int8]:
    """Return I0,Q0,I1,Q1,... signed int8; never save samples to disk."""
    normalized = normalize_iq(iq, digital_amplitude)
    output = np.empty(normalized.size * 2, dtype=np.int8)
    output[0::2] = np.clip(np.rint(normalized.real * 127), -127, 127).astype(np.int8)
    output[1::2] = np.clip(np.rint(normalized.imag * 127), -127, 127).astype(np.int8)
    return output
