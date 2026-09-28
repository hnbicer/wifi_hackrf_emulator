import numpy as np
import pytest

from wifi_emulator.packet import (
    PREAMBLE_SYMBOLS,
    WifiLikePacketGenerator,
    generate_preamble,
    packet_sample_count,
)


@pytest.mark.parametrize(
    ("duration_us", "samples"), [(1, 400), (16, 400), (20, 400), (50, 1040), (53.1, 1120), (100, 2000)]
)
def test_packet_symbol_rounding(duration_us, samples):
    packet = WifiLikePacketGenerator().generate_packet(duration_us, np.random.default_rng(3))
    assert packet_sample_count(duration_us) == samples
    assert packet.size == samples
    assert packet.dtype == np.complex64
    assert np.all(np.isfinite(packet))


@pytest.mark.parametrize("duration_us", [0, -1, np.nan, np.inf])
def test_invalid_packet_duration(duration_us):
    with pytest.raises(ValueError):
        packet_sample_count(duration_us)


def test_preamble_is_repeated_and_independent_of_data_rng():
    first = WifiLikePacketGenerator(ramp_us=0).generate_packet(100, np.random.default_rng(1))
    second = WifiLikePacketGenerator(ramp_us=0).generate_packet(100, np.random.default_rng(2))
    preamble = generate_preamble()
    np.testing.assert_array_equal(first[:320], preamble)
    np.testing.assert_array_equal(second[:320], preamble)
    for index in range(PREAMBLE_SYMBOLS):
        np.testing.assert_array_equal(preamble[index * 80 : (index + 1) * 80], preamble[:80])
    assert np.any(first[320:] != second[320:])


def test_ramp_preserves_length_middle_and_reduces_boundary_energy():
    ramped = WifiLikePacketGenerator(ramp_us=3).generate_packet(100, np.random.default_rng(1))
    plain = WifiLikePacketGenerator(ramp_us=0).generate_packet(100, np.random.default_rng(1))
    assert ramped.size == plain.size
    assert ramped[0] == ramped[-1] == 0
    np.testing.assert_array_equal(ramped[60:-60], plain[60:-60])
    assert np.sum(np.abs(ramped[:60]) ** 2) < np.sum(np.abs(plain[:60]) ** 2)
    assert np.sum(np.abs(ramped[-60:]) ** 2) < np.sum(np.abs(plain[-60:]) ** 2)


def test_long_ramp_is_capped_to_packet_length():
    packet = WifiLikePacketGenerator(ramp_us=1000).generate_packet(20, np.random.default_rng(1))
    assert packet.size == 400
    assert np.count_nonzero(packet) > 0
    assert packet[0] == packet[-1] == 0


@pytest.mark.parametrize("ramp_us", [-1, np.nan, np.inf])
def test_invalid_ramp(ramp_us):
    with pytest.raises(ValueError):
        WifiLikePacketGenerator(ramp_us=ramp_us)
