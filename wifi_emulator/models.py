"""Shared hardware-independent event and waveform records."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrafficEvent:
    event_id: int
    channel: int
    center_frequency_hz: int
    packet_durations_us: list[float]
    inter_packet_gaps_us: list[float]
    idle_before_seconds: float = 0.0
    requested_start_time: float | None = None
    waveform_seed: int = 0

    @property
    def requested_burst_duration_ms(self) -> float:
        return (sum(self.packet_durations_us) + sum(self.inter_packet_gaps_us)) / 1000


@dataclass(frozen=True)
class BurstWaveform:
    iq: NDArray[np.complex64]
    packet_sample_counts: list[int]
    gap_sample_counts: list[int]
    sample_rate: int = 20_000_000

    @property
    def actual_packet_durations_us(self) -> list[float]:
        return [n / self.sample_rate * 1e6 for n in self.packet_sample_counts]

    @property
    def actual_gap_durations_us(self) -> list[float]:
        return [n / self.sample_rate * 1e6 for n in self.gap_sample_counts]

    @property
    def duration_ms(self) -> float:
        return self.iq.size / self.sample_rate * 1000
