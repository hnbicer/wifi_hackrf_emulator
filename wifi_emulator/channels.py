"""Legacy 2.4 GHz Wi-Fi channel centers, including overlapping channels."""

import re


def channel_to_frequency(channel: int) -> int:
    """Return the center frequency in Hz for Wi-Fi channels 1 through 13."""
    if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 13:
        raise ValueError("Wi-Fi channel must be an integer between 1 and 13")
    return 2_412_000_000 + (channel - 1) * 5_000_000


def parse_channels(value: str) -> list[int]:
    """Parse comma-separated channels and inclusive ranges, preserving order.

    ``1-13``, ``1,6,11`` and ``1-3,6,11`` are accepted. Repeated channels
    are removed so a repeated argument cannot accidentally weight selection.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("channels must be a nonempty range or comma-separated list")
    channels: list[int] = []
    for part in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if match is None:
            raise ValueError(f"invalid channel selection: {part!r}")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        channel_to_frequency(first)
        channel_to_frequency(last)
        if first > last:
            raise ValueError("channel ranges must be increasing")
        for channel in range(first, last + 1):
            if channel not in channels:
                channels.append(channel)
    return channels
