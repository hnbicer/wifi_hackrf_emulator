import pytest

from wifi_emulator.channels import channel_to_frequency, parse_channels


@pytest.mark.parametrize("channel,frequency", [(1, 2_412_000_000), (6, 2_437_000_000),
                                               (11, 2_462_000_000), (13, 2_472_000_000)])
def test_channel_frequencies(channel, frequency):
    assert channel_to_frequency(channel) == frequency


@pytest.mark.parametrize("channel", [0, 14, -1, 6.0, True, "6", None])
def test_invalid_channels(channel):
    with pytest.raises(ValueError):
        channel_to_frequency(channel)


def test_channel_selection():
    assert parse_channels("1-13") == list(range(1, 14))
    assert parse_channels("1,6,11") == [1, 6, 11]
    assert parse_channels(" 1 - 3, 6, 2 ") == [1, 2, 3, 6]


@pytest.mark.parametrize("selection", ["", "0", "14", "5-2", "1,", "1--2", "a", "1.0", "1 6"])
def test_invalid_channel_selection(selection):
    with pytest.raises(ValueError):
        parse_channels(selection)
