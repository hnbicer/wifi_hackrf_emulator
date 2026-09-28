"""In-memory packet envelopes with a replaceable, nonstandard preamble."""

import math

import numpy as np
from numpy.typing import NDArray

from .ofdm import (
    FFT_SIZE,
    OCCUPIED_CARRIERS,
    SAMPLE_RATE,
    SYMBOL_DURATION_US,
    SYMBOL_SAMPLES,
    generate_ofdm_symbols,
    symbols_from_carriers,
)

PREAMBLE_SYMBOLS = 4
PREAMBLE_DURATION_US = PREAMBLE_SYMBOLS * SYMBOL_DURATION_US


def packet_sample_count(duration_us: float) -> int:
    """Round a positive requested duration up to 4 us, with a 20 us minimum.

Every packet contains a 16 us preamble and at least one 4 us data symbol.
"""
    if not math.isfinite(duration_us) or duration_us <= 0:
        raise ValueError("Packet duration must be positive and finite")
    symbol_count = max(PREAMBLE_SYMBOLS + 1, math.ceil(duration_us / SYMBOL_DURATION_US))
    return symbol_count * SYMBOL_SAMPLES


def generate_preamble() -> NDArray[np.complex64]:
    """Four repeated BPSK-loaded OFDM symbols; not IEEE STF/LTF/SIGNAL.

Fixed signs provide a repeatable startup sequence while retaining the same
52 active bins as the data. A future compliant preamble can replace this
function without changing burst scheduling.
"""
    bins = np.zeros(FFT_SIZE, dtype=np.complex64)
    for index, carrier in enumerate(OCCUPIED_CARRIERS):
        bins[carrier % FFT_SIZE] = 1 if (index * index + 3 * index + 1) % 7 < 3 else -1
    return np.tile(symbols_from_carriers(bins), PREAMBLE_SYMBOLS)


def apply_boundary_ramp(
    iq: NDArray[np.complex64], ramp_samples: int
) -> NDArray[np.complex64]:
    """Apply a raised-cosine rise/fall in place, capped at half the packet.

The first and last samples become zero when ramping is enabled. Ramping
changes only the envelope, never packet length or the embedded gap timing.
"""
    if ramp_samples < 0:
        raise ValueError("Ramp sample count must be nonnegative")
    count = min(ramp_samples, iq.size // 2)
    if count:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, count)) ** 2
        iq[:count] *= ramp
        iq[-count:] *= ramp[::-1]
    return iq


class WifiLikePacketGenerator:
    """Generate QPSK packet-like activity with a short deterministic preamble."""

    def __init__(self, ramp_us: float = 3.0) -> None:
        if not math.isfinite(ramp_us) or ramp_us < 0:
            raise ValueError("Ramp duration must be nonnegative and finite")
        self.ramp_us = ramp_us
        self.ramp_samples = round(ramp_us * (SAMPLE_RATE / 1e6))

    def generate_packet(
        self, duration_us: float, rng: np.random.Generator
    ) -> NDArray[np.complex64]:
        """Return a new complex64 packet; its waveform exists only in memory."""
        sample_count = packet_sample_count(duration_us)
        data_symbols = sample_count // SYMBOL_SAMPLES - PREAMBLE_SYMBOLS
        iq = np.concatenate((generate_preamble(), generate_ofdm_symbols(data_symbols, rng)))
        return apply_boundary_ramp(iq, self.ramp_samples)
