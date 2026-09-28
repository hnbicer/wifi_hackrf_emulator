"""Hardware-independent, reproducible sequential traffic scheduling."""

import math

import numpy as np

from .channels import channel_to_frequency
from .config import Config, feasible_packet_counts, timing_bounds
from .models import TrafficEvent


class TrafficScheduler:
    """Draw bursts conditional on their *actual* quantized duration limits.

    Choose uniformly among feasible packet counts, then uniformly choose a
    feasible total packet-symbol count and total gap-sample count. Split each
    total using bounded sequential integer draws and randomly permute parts.
    Packet requests are uniform within their chosen 4 us symbol bin; gaps use
    the 50 ns sample grid (clipped to configured endpoints if needed).

    These are conditional distributions, not independent uniform packet/gap
    draws. This construction avoids rejection loops, silently truncating
    packets, or violating burst limits when the allowed interval is narrow.
    Some configured packet counts can be infeasible and are never selected.

    Poisson mode adds exponential idle time between sequential bursts; it is
    not an absolute-time Poisson process while a single radio is occupied.
    """

    def __init__(self, config: Config):
        config.validate()
        # Copy mutable settings so later changes cannot alter a running schedule.
        self.config = Config(**config.to_dict())
        scheduling_seed, waveform_seed = np.random.SeedSequence(config.seed).spawn(2)
        self.rng = np.random.default_rng(scheduling_seed)
        self._waveform_rng = np.random.default_rng(waveform_seed)
        self._event_id = 0
        self._counts = feasible_packet_counts(config)
        self._bounds = timing_bounds(config)
        self._probabilities = None
        if config.channel_mode == "weighted":
            weights = np.array([config.channel_weights.get(ch, 0) for ch in config.channels], dtype=float)
            # Scaling first also handles valid weights whose unscaled sum overflows.
            weights /= weights.max()
            self._probabilities = weights / weights.sum()

    def _split(self, total: int, count: int, minimum: int, maximum: int) -> list[int]:
        values: list[int] = []
        for remaining in range(count - 1, -1, -1):
            low = max(minimum, total - remaining * maximum)
            high = min(maximum, total - remaining * minimum)
            value = int(self.rng.integers(low, high + 1)) if low < high else low
            values.append(value)
            total -= value
        self.rng.shuffle(values)
        return values

    def next_event(self) -> TrafficEvent:
        """Return the next plan; the runner supplies a wall-clock request time."""
        cfg = self.config
        channel = int(self.rng.choice(cfg.channels, p=self._probabilities))
        idle = (float(self.rng.exponential(1 / cfg.lambda_events_per_second))
                if cfg.traffic_mode == "poisson"
                else float(self.rng.uniform(cfg.idle_ms_min, cfg.idle_ms_max)) / 1000)
        count = int(self.rng.choice(self._counts))
        pmin, pmax, gmin, gmax, minimum, maximum = self._bounds
        low = max(count * pmin, math.ceil((minimum - (count - 1) * gmax) / 80))
        high = min(count * pmax, (maximum - (count - 1) * gmin) // 80)
        symbols_total = int(self.rng.integers(low, high + 1)) if low < high else low
        low_gap = max((count - 1) * gmin, minimum - 80 * symbols_total)
        high_gap = min((count - 1) * gmax, maximum - 80 * symbols_total)
        gap_total = int(self.rng.integers(low_gap, high_gap + 1)) if low_gap < high_gap else low_gap
        symbol_counts = self._split(symbols_total, count, pmin, pmax)
        gap_counts = self._split(gap_total, count - 1, gmin, gmax)
        packet_durations = []
        for symbols in symbol_counts:
            lower = cfg.packet_us_min if symbols == 5 else max(
                cfg.packet_us_min, float(np.nextafter((symbols - 1) * 4.0, math.inf))
            )
            upper = min(cfg.packet_us_max, symbols * 4.0)
            packet_durations.append(float(self.rng.uniform(lower, upper)) if lower < upper else lower)
        gaps = [max(cfg.gap_us_min, min(cfg.gap_us_max, samples / 20)) for samples in gap_counts]
        self._event_id += 1
        return TrafficEvent(
            event_id=self._event_id,
            channel=channel,
            center_frequency_hz=channel_to_frequency(channel),
            packet_durations_us=packet_durations,
            inter_packet_gaps_us=gaps,
            idle_before_seconds=idle,
            waveform_seed=int(self._waveform_rng.integers(0, 2**63, dtype=np.int64)),
        )
