from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from wifi_emulator.config import Config, load_config


def test_defaults_and_example():
    Config().validate()
    load_config(Path(__file__).parents[1] / "config.example.yaml").validate()
    assert Config().channels == list(range(1, 14))
    assert Config().transmit is False
    assert Config().rf_amp_enabled is False
    Config(duration_seconds=0).validate()


@pytest.mark.parametrize("changes", [
    {"sample_rate": 10_000_000}, {"rf_amp_enabled": True}, {"fft_size": 128},
    {"cyclic_prefix": 8}, {"modulation": "16qam"}, {"digital_amplitude": 0},
    {"digital_amplitude": 1.1}, {"digital_amplitude": float("nan")},
    {"duration_seconds": -1}, {"lambda_events_per_second": 0}, {"seed": -1},
    {"seed": True}, {"tx_gain_db": 48}, {"tx_gain_db": 2.5},
    {"packet_count_min": 0}, {"packet_count_min": True},
    {"packet_count_min": 11}, {"packet_us_min": 0}, {"gap_us_min": -1},
    {"idle_ms_min": 200}, {"transfer_timeout_seconds": 0}, {"channels": []},
    {"channels": [1, 1]}, {"channels": [14]}, {"transmit": "false"},
    {"channel_mode": "weighted"}, {"channel_mode": "weighted", "channel_weights": {1: 0}},
    {"channel_weights": {1: -1}}, {"channels": [1], "channel_weights": {6: 1}},
    {"channel_weights": {1: float("inf")}}, {"traffic_mode": "unknown"},
    {"burst_ms_max": 1000.1}, {"packet_count_max": 10_001},
    {"packet_us_max": 1_000_001}, {"gap_us_max": 1_000_001},
    {"packet_count_min": 3, "packet_count_max": 3, "packet_us_min": 1000,
     "packet_us_max": 1000, "burst_ms_max": 2},
    # Continuous-looking limits are impossible on the 4 us packet grid.
    {"packet_count_max": 1, "burst_ms_min": 0.201, "burst_ms_max": 0.203},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_mutable_defaults_are_independent():
    first, second = Config(), Config()
    first.channels.clear()
    first.channel_weights[1] = 3
    assert len(second.channels) == 13
    assert second.channel_weights == {}


def test_nested_and_flat_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""experiment:
  seed: 12
traffic:
  mode: poisson
  channels: '1,6,11'
  channel_mode: weighted
  channel_weights: {'1': 3, '6': 4, '11': 3}
  packet_count: {min: 2, max: 4}
  packet_duration_us: {min: 100, max: 700}
  inter_packet_gap_us: {min: 30, max: 300}
""", encoding="utf-8")
    config = load_config(path)
    config.validate()
    assert config.seed == 12
    assert config.packet_count_min == 2
    assert config.channel_weights == {1: 3, 6: 4, 11: 3}
    saved = config.to_dict()
    saved.update(program_version="0.1.0", created_at="now", rng={}, timing_reference="host")
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    assert load_config(path) == config


@pytest.mark.parametrize("text", ["unknown: 1", "traffic: {unknown: 1}", "signal: []",
                                     "[1,2]", "traffic: {packet_count: {bad: 1}}",
                                     "seed: 1\nseed: 2", "seed: 1\nexperiment: {seed: 2}",
                                     "channel_weights: {1: 1, '1': 2}", "traffic: ["])
def test_unknown_or_invalid_schema_is_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_validation_after_cli_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("packet_count_min: 11\n", encoding="utf-8")
    loaded = load_config(path)
    replace(loaded, packet_count_max=12).validate()
