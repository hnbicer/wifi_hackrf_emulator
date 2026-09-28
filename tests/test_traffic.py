from collections import Counter
from dataclasses import replace
import math

import numpy as np
import pytest

from wifi_emulator.config import Config
from wifi_emulator.traffic import TrafficScheduler


def samples_in(event):
    return (sum(max(5, math.ceil(duration / 4)) * 80 for duration in event.packet_durations_us)
            + sum(round(gap * 20) for gap in event.inter_packet_gaps_us))


def test_reproducibility():
    left = TrafficScheduler(Config(seed=12345))
    right = TrafficScheduler(Config(seed=12345))
    left_events = [left.next_event() for _ in range(50)]
    assert left_events == [right.next_event() for _ in range(50)]
    assert left_events[0].event_id == 1
    assert left_events[-1].event_id == 50
    assert len({event.waveform_seed for event in left_events}) == 50
    assert all(event.requested_start_time is None for event in left_events)
    other = TrafficScheduler(Config(seed=99))
    assert left_events != [other.next_event() for _ in range(50)]


def test_waveform_rng_does_not_perturb_schedule():
    left = TrafficScheduler(Config(seed=9))
    right = TrafficScheduler(Config(seed=9))
    for _ in range(10):
        one, two = left.next_event(), right.next_event()
        np.random.default_rng(one.waveform_seed).random(1000)
        assert one == two


@pytest.mark.parametrize("config", [
    Config(seed=123),
    Config(seed=3, packet_count_min=5, packet_count_max=6, packet_us_min=11,
           packet_us_max=101, gap_us_min=0.026, gap_us_max=0.074,
           burst_ms_min=0.3002, burst_ms_max=0.30025),
    Config(seed=4, packet_count_min=1, packet_count_max=1,
           burst_ms_min=0.2, burst_ms_max=0.2),
    Config(seed=5, packet_count_min=3, packet_count_max=3, packet_us_min=50.3,
           packet_us_max=50.3, gap_us_min=1.025, gap_us_max=1.025,
           burst_ms_min=0.158, burst_ms_max=0.159),
])
def test_hard_bounds_and_configured_ranges(config):
    scheduler = TrafficScheduler(config)
    for _ in range(150):
        event = scheduler.next_event()
        assert config.packet_count_min <= len(event.packet_durations_us) <= config.packet_count_max
        assert len(event.inter_packet_gaps_us) == len(event.packet_durations_us) - 1
        assert all(config.packet_us_min <= d <= config.packet_us_max for d in event.packet_durations_us)
        assert all(config.gap_us_min <= d <= config.gap_us_max for d in event.inter_packet_gaps_us)
        duration_ms = samples_in(event) / 20_000
        assert config.burst_ms_min - 1e-10 <= duration_ms <= config.burst_ms_max + 1e-10
        assert config.idle_ms_min / 1000 <= event.idle_before_seconds <= config.idle_ms_max / 1000


def test_weighted_channels():
    scheduler = TrafficScheduler(Config(seed=72, channels=[1, 6, 11, 13], channel_mode="weighted",
                                        channel_weights={1: 3, 6: 4, 11: 3}))
    counts = Counter(scheduler.next_event().channel for _ in range(6000))
    assert counts[13] == 0
    for channel, expected in ((1, 0.3), (6, 0.4), (11, 0.3)):
        assert abs(counts[channel] / 6000 - expected) < 0.035


def test_poisson_idle_mean():
    scheduler = TrafficScheduler(Config(seed=6, traffic_mode="poisson", lambda_events_per_second=20))
    samples = np.array([scheduler.next_event().idle_before_seconds for _ in range(3000)])
    assert samples.min() >= 0
    assert np.mean(samples) == pytest.approx(0.05, rel=0.08)


def test_infeasible_counts_are_excluded_without_violating_minimum():
    cfg = Config(seed=5, packet_count_min=3, packet_count_max=10, packet_us_min=100,
                 packet_us_max=100, gap_us_min=0, gap_us_max=0, burst_ms_min=0.3,
                 burst_ms_max=0.35)
    scheduler = TrafficScheduler(cfg)
    assert all(len(scheduler.next_event().packet_durations_us) == 3 for _ in range(30))
    with pytest.raises(ValueError, match="no packet count"):
        TrafficScheduler(replace(cfg, burst_ms_max=0.29, burst_ms_min=0.2))
