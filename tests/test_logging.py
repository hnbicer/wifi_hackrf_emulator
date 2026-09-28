"""Verify durable records and sample-derived occupancy statistics."""

import csv
import json

import pytest

from wifi_emulator.config import Config
from wifi_emulator.logger import ExperimentLogger, FIELDS
from wifi_emulator.models import TrafficEvent


def event():
    return TrafficEvent(1, 1, 2_412_000_000, [99.5, 199.5], [299.5], waveform_seed=456)


def generated(record, *, transmitted):
    record.update(
        status="success" if transmitted else "dry_run",
        actual_packet_durations_us=[100.0, 200.0],
        actual_gap_durations_us=[300.0],
        packet_sample_counts=[2000, 4000],
        gap_sample_counts=[6000],
        sample_count=12000,
        actual_waveform_duration_ms=0.6,
        transmission_attempted=transmitted,
        hackrf_success=True if transmitted else None,
        host_elapsed_seconds=9.0 if transmitted else None,
    )
    return record


def test_requested_record_is_flushed_before_completion(tmp_path):
    logger = ExperimentLogger(Config(output_dir=str(tmp_path), seed=123, channels=[1, 6]))
    record = logger.requested(event())
    journal = [json.loads(line) for line in (logger.directory / "events.jsonl").read_text().splitlines()]
    assert len(journal) == 1
    assert journal[0]["status"] == "requested"
    assert journal[0]["sample_count"] is None
    assert journal[0]["hackrf_success"] is None
    assert not journal[0]["transmission_attempted"]
    assert set(record) == set(FIELDS)
    assert not any("filename" in name for name in FIELDS)
    assert list(csv.DictReader((logger.directory / "ground_truth.csv").open())) == []
    logger.completed(generated(record, transmitted=False))
    logger.finish(1.0, "completed")
    saved_config = json.loads((logger.directory / "config.json").read_text())
    assert saved_config["seed"] == 123
    assert saved_config["sample_rate"] == 20_000_000
    assert saved_config["rf_amp_enabled"] is False
    assert saved_config["program_version"]
    with (logger.directory / "ground_truth.csv").open(newline="") as stream:
        row, = csv.DictReader(stream)
    assert json.loads(row["packet_durations_us"]) == [99.5, 199.5]
    assert json.loads(row["actual_packet_durations_us"]) == [100.0, 200.0]
    assert json.loads(row["transmission_attempted"]) is False
    assert row["hackrf_success"] == ""
    journal = [json.loads(line) for line in (logger.directory / "events.jsonl").read_text().splitlines()]
    assert [row["status"] for row in journal] == ["requested", "dry_run"]
    assert {path.name for path in logger.directory.iterdir()} == {
        "ground_truth.csv", "events.jsonl", "config.json", "summary.json"
    }


@pytest.mark.parametrize("transmitted", [False, True])
def test_airtime_uses_packets_and_excludes_zero_gaps_and_process_time(tmp_path, transmitted):
    logger = ExperimentLogger(Config(output_dir=str(tmp_path), channels=[1, 6], transmit=transmitted, seed=1))
    logger.completed(generated(logger.requested(event()), transmitted=transmitted))
    summary = logger.finish(10.0, "completed")
    assert summary["requested_packet_airtime_seconds_per_channel"]["1"] == pytest.approx(0.000299)
    assert summary["generated_packet_airtime_seconds_per_channel"] == pytest.approx({"1": 0.0003, "6": 0.0})
    assert summary["generated_waveform_duration_seconds_per_channel"]["1"] == pytest.approx(0.0006)
    assert summary["successful_hackrf_transmissions"] == int(transmitted)
    assert summary["attempted_transmissions"] == int(transmitted)
    assert summary["failed_transmissions"] == 0
    assert summary["approximate_channel_duty_cycle"]["1"] == pytest.approx(0.00003 if transmitted else 0)
    if not transmitted:
        assert summary["simulated_channel_duty_cycle"]["1"] == pytest.approx(0.00003)
    assert summary["total_packet_count"] == 2
    assert summary["mean_packet_duration_us"] == 150
    assert summary["mean_burst_duration_ms"] == 0.6
    assert summary["bursts_per_channel"] == {"1": 1, "6": 0}
    assert json.loads((logger.directory / "summary.json").read_text()) == summary


def test_generation_failure_is_not_a_failed_transmission(tmp_path):
    logger = ExperimentLogger(Config(output_dir=str(tmp_path), transmit=True, channels=[1]))
    record = logger.requested(event())
    record.update(status="failed", error="waveform generation failed")
    logger.completed(record)
    summary = logger.finish(0.5, "failed", record["error"])
    assert summary["number_of_bursts"] == 1
    assert summary["generated_bursts"] == 0
    assert summary["event_status_counts"]["failed"] == 1
    assert summary["attempted_transmissions"] == summary["failed_transmissions"] == 0
    assert summary["total_packet_count"] == 0


def test_interrupted_attempt_counts_as_failed_transmission(tmp_path):
    logger = ExperimentLogger(Config(output_dir=str(tmp_path), transmit=True, channels=[1]))
    record = generated(logger.requested(event()), transmitted=True)
    record.update(status="interrupted", hackrf_success=False, error="KeyboardInterrupt")
    logger.completed(record)
    summary = logger.finish(0.5, "interrupted", "KeyboardInterrupt")
    assert summary["attempted_transmissions"] == summary["failed_transmissions"] == 1
    assert summary["successful_hackrf_transmissions"] == 0
    assert summary["successful_packet_airtime_seconds_per_channel"]["1"] == 0
