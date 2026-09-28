"""Exercise experiment state transitions using only in-memory/mock hardware."""

import csv
from dataclasses import replace
import json

import pytest

from wifi_emulator import hackrf, runner
from wifi_emulator.config import Config
from wifi_emulator.hackrf import TransmissionResult, TransmitterBackend
from wifi_emulator.models import TrafficEvent


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    return Config(output_dir=str(tmp_path), seed=123, channels=[1, 6], duration_seconds=60)


@pytest.fixture
def events():
    return [
        TrafficEvent(1, 1, 2_412_000_000, [52.0, 100.0], [50.0], waveform_seed=10),
        TrafficEvent(2, 6, 2_437_000_000, [200.0], [], waveform_seed=20),
    ]


def records(result):
    with (result.directory / "ground_truth.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    journal = [json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()]
    return csv_rows, journal


def result_record(success=True):
    return TransmissionResult(
        host_start_timestamp="2026-01-01T00:00:00+00:00",
        host_end_timestamp="2026-01-01T00:00:01+00:00",
        host_elapsed_seconds=1.0, return_code=0 if success else 2,
        success=success, error=None if success else "mock transfer failed",
        stderr="" if success else "USB transfer failed",
    )


class MockBackend(TransmitterBackend):
    def __init__(self, success=True):
        self.checks = 0
        self.calls = 0
        self.success = success
        self.last_result = None

    def check_available(self):
        self.checks += 1
        return "Mock device; no hardware access"

    def transmit(self, *args, **kwargs):
        self.calls += 1
        self.last_result = result_record(self.success)
        return self.last_result


def test_dry_run_never_accesses_backend_or_creates_iq_files(config, events, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run accessed hardware or created a temporary IQ file")

    backend = MockBackend()
    monkeypatch.setattr(backend, "check_available", forbidden)
    monkeypatch.setattr(backend, "transmit", forbidden)
    monkeypatch.setattr(runner, "HackRFTransferBackend", forbidden)
    monkeypatch.setattr(hackrf.subprocess, "Popen", forbidden)
    monkeypatch.setattr(hackrf.tempfile, "TemporaryDirectory", forbidden)
    result = runner.run_experiment(config, events=events, backend=backend)
    assert result.status == "completed"
    rows, journal = records(result)
    assert [row["status"] for row in rows] == ["dry_run", "dry_run"]
    assert [row["status"] for row in journal] == ["requested", "dry_run", "requested", "dry_run"]
    assert all(row["hackrf_success"] == "" for row in rows)
    assert all(row["transmission_attempted"] == "false" for row in rows)
    assert all(row["host_start_timestamp"] == "" for row in rows)
    assert result.summary["successful_hackrf_transmissions"] == result.summary["failed_transmissions"] == 0
    assert result.summary["dry_run_bursts"] == result.summary["generated_bursts"] == 2
    assert all(value == 0 for value in result.summary["approximate_channel_duty_cycle"].values())
    assert {path.name for path in result.directory.iterdir()} == {
        "config.json", "events.jsonl", "ground_truth.csv", "summary.json"
    }
    assert json.loads((result.directory / "summary.json").read_text()) == result.summary


def test_request_is_on_disk_before_waveform_generation(config, events, monkeypatch):
    original = runner.generate_burst

    def generate(event, **options):
        directory, = list(config_path.glob("experiment_*"))
        journal = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        assert journal[-1]["event_id"] == event.event_id
        assert journal[-1]["status"] == "requested"
        assert journal[-1]["sample_count"] is None
        return original(event, **options)

    from pathlib import Path
    config_path = Path(config.output_dir)
    monkeypatch.setattr(runner, "generate_burst", generate)
    runner.run_experiment(config, events=events)


def test_successful_mocked_transmission_logs_counts_and_warning(config, events, capsys):
    backend = MockBackend()
    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=backend)
    assert backend.checks == 1
    assert backend.calls == 2
    assert result.status == "completed"
    assert result.summary["attempted_transmissions"] == result.summary["successful_hackrf_transmissions"] == 2
    assert "RF TRANSMISSION ENABLED" in capsys.readouterr().out
    rows, _ = records(result)
    assert all(row["transmission_attempted"] == row["hackrf_success"] == "true" for row in rows)
    assert result.summary["successful_packet_airtime_seconds_per_channel"]["1"] == pytest.approx(0.000152)


def test_failed_result_stops_run_and_saves_summary(config, events):
    backend = MockBackend(success=False)
    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=backend)
    assert result.status == "failed"
    assert backend.calls == 1
    rows, journal = records(result)
    assert len(rows) == 1
    assert rows[0]["hackrf_stderr"] == "USB transfer failed"
    assert [row["status"] for row in journal] == ["requested", "failed"]
    assert result.summary["failed_transmissions"] == 1
    assert result.summary["successful_hackrf_transmissions"] == 0


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_generation_failure_after_success_does_not_copy_prior_result(config, events, monkeypatch, exception_type):
    original = runner.generate_burst

    def generate(event, **options):
        if event.event_id == 2:
            raise exception_type("generation stopped")
        return original(event, **options)

    backend = MockBackend()
    monkeypatch.setattr(runner, "generate_burst", generate)
    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=backend)
    expected = "failed" if exception_type is RuntimeError else "interrupted"
    assert result.status == expected
    rows, journal = records(result)
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == expected
    assert rows[1]["transmission_attempted"] == "false"
    assert rows[1]["host_start_timestamp"] == rows[1]["host_end_timestamp"] == ""
    assert rows[1]["hackrf_success"] == rows[1]["sample_count"] == ""
    assert len(journal) == 4
    assert backend.calls == 1
    assert result.summary["successful_hackrf_transmissions"] == 1
    assert result.summary["failed_transmissions"] == 0


def test_backend_result_reset_before_later_exception(config, events):
    class LaterFailure(MockBackend):
        def transmit(self, *args, **kwargs):
            assert self.last_result is None
            if self.calls:
                self.calls += 1
                raise RuntimeError("second backend invocation failed")
            return super().transmit(*args, **kwargs)

    backend = LaterFailure()
    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=backend)
    rows, _ = records(result)
    assert result.status == "failed"
    assert rows[1]["status"] == "failed"
    assert rows[1]["transmission_attempted"] == "true"
    assert rows[1]["hackrf_success"] == "false"
    assert rows[1]["host_start_timestamp"] == rows[1]["host_end_timestamp"] == ""
    assert result.summary["failed_transmissions"] == result.summary["successful_hackrf_transmissions"] == 1


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_backend_interrupt_preserves_cleanup_result_and_terminal_journal(config, events, exception_type):
    class InterruptedBackend(MockBackend):
        def transmit(self, *args, **kwargs):
            self.calls += 1
            self.last_result = result_record(False)
            raise exception_type("mock interrupt after child cleanup")

    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=InterruptedBackend())
    rows, journal = records(result)
    assert result.status == "interrupted"
    assert len(rows) == 1
    assert rows[0]["status"] == "interrupted"
    assert rows[0]["host_start_timestamp"] == result_record().host_start_timestamp
    assert rows[0]["hackrf_return_code"] == "2"
    assert [row["status"] for row in journal] == ["requested", "interrupted"]
    assert result.summary["attempted_transmissions"] == result.summary["failed_transmissions"] == 1


def test_preflight_failure_is_logged_without_transmission_attempt(config, events):
    class NoDevice(MockBackend):
        def check_available(self):
            raise RuntimeError("No device attached")

    backend = NoDevice()
    result = runner.run_experiment(replace(config, transmit=True), events=events, backend=backend)
    assert result.status == "failed"
    assert result.summary["number_of_bursts"] == result.summary["failed_transmissions"] == 0
    assert "No device attached" in result.summary["error"]
    assert backend.calls == 0
    assert records(result) == ([], [])


def test_interrupt_during_idle_preserves_request_without_rf_claim(config, events, monkeypatch):
    def interrupt(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner.time, "sleep", interrupt)
    result = runner.run_experiment(config, events=events)
    assert result.status == "interrupted"
    rows, journal = records(result)
    assert rows[0]["sample_count"] == rows[0]["hackrf_success"] == ""
    assert rows[0]["transmission_attempted"] == "false"
    assert [row["status"] for row in journal] == ["requested", "interrupted"]
    assert result.summary["generated_bursts"] == result.summary["failed_transmissions"] == 0


def test_unspecified_seed_is_saved_for_reproduction(config, events):
    result = runner.run_experiment(replace(config, seed=None), events=events[:1])
    saved = json.loads((result.directory / "config.json").read_text())
    assert isinstance(saved["seed"], int)
    assert saved["seed"] == result.summary["seed"]
