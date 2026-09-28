"""Validated configuration and strict YAML loading.

``load_config`` loads values without cross-field validation so CLI overrides
can complete a configuration. Call ``Config.validate`` after all overrides.
"""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any

import yaml

from .channels import channel_to_frequency, parse_channels


@dataclass
class Config:
    duration_seconds: float = 60.0
    seed: int | None = None
    channels: list[int] = field(default_factory=lambda: list(range(1, 14)))
    sample_rate: int = 20_000_000
    tx_gain_db: int = 0
    rf_amp_enabled: bool = False
    digital_amplitude: float = 0.6
    fft_size: int = 64
    cyclic_prefix: int = 16
    modulation: str = "qpsk"
    ramp_us: float = 3.0
    traffic_mode: str = "random"
    lambda_events_per_second: float = 20.0
    idle_ms_min: float = 5.0
    idle_ms_max: float = 100.0
    packet_count_min: int = 1
    packet_count_max: int = 10
    packet_us_min: float = 50.0
    packet_us_max: float = 2000.0
    gap_us_min: float = 20.0
    gap_us_max: float = 500.0
    burst_ms_min: float = 0.2
    burst_ms_max: float = 10.0
    channel_mode: str = "uniform"
    channel_weights: dict[int, float] = field(default_factory=dict)
    output_dir: str = "results"
    transmit: bool = False
    verbose: bool = False
    transfer_timeout_seconds: float = 30.0

    def validate(self) -> None:
        """Reject invalid types, unsupported PHY settings and impossible bursts."""
        for name in ("sample_rate", "tx_gain_db", "fft_size", "cyclic_prefix",
                     "packet_count_min", "packet_count_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or null")
        for name in ("rf_amp_enabled", "transmit", "verbose"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if self.rf_amp_enabled:
            raise ValueError("rf_amp_enabled must remain false in this version")
        if self.sample_rate != 20_000_000 or self.fft_size != 64 or self.cyclic_prefix != 16:
            raise ValueError("this PHY requires sample_rate=20000000, fft_size=64, cyclic_prefix=16")
        if self.modulation != "qpsk":
            raise ValueError("modulation must be qpsk")
        if not 0 <= self.tx_gain_db <= 47:
            raise ValueError("tx_gain_db must be between 0 and 47")
        if not isinstance(self.channels, list) or not self.channels:
            raise ValueError("channels must be a nonempty list")
        for channel in self.channels:
            channel_to_frequency(channel)
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        numeric_fields = (
            "duration_seconds", "digital_amplitude", "ramp_us", "lambda_events_per_second",
            "idle_ms_min", "idle_ms_max", "packet_us_min", "packet_us_max", "gap_us_min",
            "gap_us_max", "burst_ms_min", "burst_ms_max", "transfer_timeout_seconds",
        )
        for name in numeric_fields:
            _finite_number(getattr(self, name), name)
        for name in ("lambda_events_per_second", "packet_us_min",
                     "burst_ms_min", "transfer_timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("duration_seconds", "idle_ms_min", "gap_us_min", "ramp_us"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.digital_amplitude <= 1:
            raise ValueError("digital_amplitude must be greater than 0 and at most 1")
        if self.packet_count_min < 1:
            raise ValueError("packet_count_min must be at least 1")
        for prefix in ("idle_ms", "packet_count", "packet_us", "gap_us", "burst_ms"):
            if getattr(self, prefix + "_min") > getattr(self, prefix + "_max"):
                raise ValueError(f"{prefix}_min must not exceed {prefix}_max")
        # Bound allocations for the in-memory subprocess backend. These permit
        # bursts much longer than the default 10 ms without unbounded RAM use.
        for name, limit in (("burst_ms_max", 1000), ("packet_count_max", 10_000),
                            ("packet_us_max", 1_000_000), ("gap_us_max", 1_000_000)):
            if getattr(self, name) > limit:
                raise ValueError(f"{name} must be at most {limit} for the in-memory backend")
        if self.traffic_mode not in ("random", "poisson"):
            raise ValueError("traffic_mode must be random or poisson")
        if self.channel_mode not in ("uniform", "weighted"):
            raise ValueError("channel_mode must be uniform or weighted")
        if not isinstance(self.channel_weights, dict):
            raise ValueError("channel_weights must be a mapping")
        for channel, weight in self.channel_weights.items():
            channel_to_frequency(channel)
            if channel not in self.channels:
                raise ValueError(f"weight for channel {channel} is outside selected channels")
            _finite_number(weight, f"channel_weights[{channel}]")
            if weight < 0:
                raise ValueError("channel weights must be nonnegative")
        if self.channel_mode == "weighted" and not any(self.channel_weights.values()):
            raise ValueError("weighted mode requires at least one positive channel weight")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path string")
        if not feasible_packet_counts(self):
            raise ValueError(
                "no packet count fits the burst duration limits after OFDM symbol "
                "and gap-sample rounding; adjust packet, gap, count or burst bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete serializable snapshot, including safe RF defaults."""
        return asdict(self)


def _finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def timing_bounds(config: Config) -> tuple[int, int, int, int, int, int]:
    """Packet lengths in 80-sample symbols; gap/burst limits in samples.

    A packet contains 16 us of preamble and at least one 4 us data symbol.
    Burst endpoints are inclusive, with sub-sample floating point noise removed.
    """
    return (
        max(5, math.ceil(config.packet_us_min / 4)),
        max(5, math.ceil(config.packet_us_max / 4)),
        round(config.gap_us_min * 20),
        round(config.gap_us_max * 20),
        math.ceil(config.burst_ms_min * 20_000 - 1e-8),
        math.floor(config.burst_ms_max * 20_000 + 1e-8),
    )


def feasible_packet_counts(config: Config) -> list[int]:
    """Counts having at least one exact, quantized realization inside the bounds."""
    pmin, pmax, gmin, gmax, minimum, maximum = timing_bounds(config)
    first = max(config.packet_count_min, math.ceil((minimum + gmax) / (80 * pmax + gmax)))
    last = min(config.packet_count_max, (maximum + gmin) // (80 * pmin + gmin))
    return [n for n in range(first, last + 1)
            if max(n * pmin, math.ceil((minimum - (n - 1) * gmax) / 80))
            <= min(n * pmax, (maximum - (n - 1) * gmin) // 80)]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject accidentally repeated configuration keys instead of hiding values."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise ValueError(f"duplicate YAML configuration key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise ValueError("configuration keys must be scalar values") from exc
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: str | Path) -> Config:
    """Load flat saved configs or the documented nested YAML schema.

    Unknown and duplicate keys are errors. Validation of values and combinations
    is deliberately deferred until CLI overrides have been applied.
    """
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    names = {item.name for item in fields(Config)}
    sections = {
        "experiment": {name: name for name in ("duration_seconds", "seed", "output_dir", "verbose", "transmit")},
        "hackrf": {name: name for name in ("sample_rate", "tx_gain_db", "rf_amp_enabled", "transfer_timeout_seconds")},
        "signal": {name: name for name in ("digital_amplitude", "fft_size", "cyclic_prefix", "modulation", "ramp_us")},
        "traffic": {name: name for name in ("lambda_events_per_second", "channels", "channel_mode", "channel_weights",
                                             "idle_ms_min", "idle_ms_max", "packet_count_min", "packet_count_max",
                                             "packet_us_min", "packet_us_max", "gap_us_min", "gap_us_max",
                                             "burst_ms_min", "burst_ms_max")},
    }
    sections["traffic"]["mode"] = "traffic_mode"
    ranges = {"idle_ms": "idle_ms", "packet_count": "packet_count", "packet_duration_us": "packet_us",
              "inter_packet_gap_us": "gap_us", "burst_duration_ms": "burst_ms"}
    values: dict[str, Any] = {}

    def assign(name: str, value: Any) -> None:
        if name in values:
            raise ValueError(f"configuration sets {name} more than once")
        values[name] = value

    for key, value in data.items():
        if key in ("program_version", "created_at", "rng", "timing_reference"):
            continue
        if key in names:
            assign(key, value)
        elif key in sections:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            for subkey, subvalue in value.items():
                if subkey in sections[key]:
                    assign(sections[key][subkey], subvalue)
                elif key == "traffic" and subkey in ranges:
                    if not isinstance(subvalue, dict):
                        raise ValueError(f"traffic.{subkey} must contain min/max values")
                    for endpoint, number in subvalue.items():
                        if endpoint not in ("min", "max"):
                            raise ValueError(f"unknown configuration key: traffic.{subkey}.{endpoint}")
                        assign(ranges[subkey] + "_" + endpoint, number)
                else:
                    raise ValueError(f"unknown configuration key: {key}.{subkey}")
        else:
            raise ValueError(f"unknown configuration key: {key}")
    if isinstance(values.get("channels"), str):
        values["channels"] = parse_channels(values["channels"])
    if isinstance(values.get("channel_weights"), dict):
        normalized: dict[int, Any] = {}
        for channel, weight in values["channel_weights"].items():
            if isinstance(channel, str) and channel.isdecimal():
                channel = int(channel)
            if channel in normalized:
                raise ValueError(f"duplicate channel weight: {channel}")
            normalized[channel] = weight
        values["channel_weights"] = normalized
    return Config(**values)
