"""Assemble reproducible packets and sample-timed silence, entirely in RAM."""

import math

import numpy as np

from .models import BurstWaveform, TrafficEvent
from .ofdm import SAMPLE_RATE
from .packet import WifiLikePacketGenerator


def generate_burst(event: TrafficEvent, ramp_us: float = 3.0) -> BurstWaveform:
    """Generate an event using its independent seed and rounded zero-sample gaps.

There must be exactly one gap between adjacent packets; no leading/trailing
gap is inserted. Python's nearest-integer rounding defines half-sample ties.
The event's idle_before_seconds belongs to the host scheduler, not this IQ.
"""
    count = len(event.packet_durations_us)
    if count == 0:
        raise ValueError("A burst must contain at least one packet")
    if len(event.inter_packet_gaps_us) != count - 1:
        raise ValueError("A burst must have exactly one gap between adjacent packets")
    gap_counts: list[int] = []
    for gap_us in event.inter_packet_gaps_us:
        if not math.isfinite(gap_us) or gap_us < 0:
            raise ValueError("Gap durations must be nonnegative and finite")
        gap_counts.append(round(gap_us * (SAMPLE_RATE / 1e6)))
    generator = WifiLikePacketGenerator(ramp_us=ramp_us)
    rng = np.random.default_rng(event.waveform_seed)
    pieces = []
    packet_counts = []
    for index, duration_us in enumerate(event.packet_durations_us):
        packet = generator.generate_packet(duration_us, rng)
        pieces.append(packet)
        packet_counts.append(packet.size)
        if index < len(gap_counts):
            pieces.append(np.zeros(gap_counts[index], dtype=np.complex64))
    return BurstWaveform(
        iq=np.concatenate(pieces),
        packet_sample_counts=packet_counts,
        gap_sample_counts=gap_counts,
        sample_rate=SAMPLE_RATE,
    )
