import numpy as np
import pytest

from wifi_emulator.burst import generate_burst
from wifi_emulator.models import TrafficEvent


def event(durations=None, gaps=None, seed=12):
    return TrafficEvent(
        event_id=1,
        channel=6,
        center_frequency_hz=2_437_000_000,
        packet_durations_us=[50, 101, 200] if durations is None else durations,
        inter_packet_gaps_us=[20.03, 120] if gaps is None else gaps,
        idle_before_seconds=1.0,
        waveform_seed=seed,
    )


def test_burst_sample_timing_and_zero_gaps():
    burst = generate_burst(event())
    assert burst.packet_sample_counts == [1040, 2080, 4000]
    assert burst.gap_sample_counts == [401, 2400]
    assert burst.iq.size == 1040 + 2080 + 4000 + 401 + 2400
    assert burst.iq.dtype == np.complex64
    np.testing.assert_array_equal(burst.iq[1040:1441], 0)
    np.testing.assert_array_equal(burst.iq[3521:5921], 0)
    assert burst.actual_packet_durations_us == pytest.approx([52, 104, 200])
    assert burst.actual_gap_durations_us == pytest.approx([20.05, 120])
    assert burst.duration_ms == pytest.approx(burst.iq.size / 20_000)


def test_burst_reproducibility_and_independent_seed():
    first = generate_burst(event())
    second = generate_burst(event())
    different = generate_burst(event(seed=13))
    np.testing.assert_array_equal(first.iq, second.iq)
    assert np.any(first.iq != different.iq)
    assert first.packet_sample_counts == different.packet_sample_counts


def test_single_packet_and_zero_length_gap():
    single = generate_burst(event([50], []))
    assert single.iq.size == 1040
    assert single.gap_sample_counts == []
    contiguous = generate_burst(event([50, 50], [0]))
    assert contiguous.iq.size == 2080
    assert contiguous.gap_sample_counts == [0]


@pytest.mark.parametrize(
    ("durations", "gaps"), [([], []), ([50], [20]), ([50, 50], []), ([50, 50], [-1]), ([50, 50], [np.nan])]
)
def test_invalid_burst(durations, gaps):
    with pytest.raises(ValueError):
        generate_burst(event(durations, gaps))
