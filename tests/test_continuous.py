"""Continuous packet timing and menu behavior without any RF hardware access."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from wifi_emulator import continuous, hackrf
from wifi_emulator.continuous import ContinuousSettings, continuous_main, generate_loop, run_continuous
from wifi_emulator.hackrf import TransmissionResult
from wifi_emulator.runner import ExperimentResult


def forbidden(*args, **kwargs):
    pytest.fail("Unexpected waveform allocation, temporary IQ, or hardware access")


def saved_summary(result):
    summary = json.loads((result.directory / "continuous_summary.json").read_text(encoding="utf-8"))
    assert summary == result.summary
    assert summary["ended_at"] is not None
    assert summary["elapsed_seconds"] >= 0
    assert {path.name for path in result.directory.iterdir()} == {"config.json", "continuous_summary.json"}
    return summary


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(continuous.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(continuous.time, "sleep", clock.sleep)
    return clock


def host_result(*, success=True, error=None):
    return TransmissionResult("2026-10-01T12:00:00+00:00", "2026-10-01T12:00:01+00:00",
                              1.0, 0 if success else 1, success,
                              stdout="mock host output", error=error)


class FakeRepeater:
    def __init__(self, result=None, failure=None):
        self.result = result or host_result()
        self.failure = failure
        self.last_result = host_result(error="stale result")
        self.preflights = 0
        self.calls = []

    def check_available(self):
        self.preflights += 1
        return "mock HackRF available"

    def transmit_repeating(self, iq, frequency_hz, **kwargs):
        self.calls.append((iq.copy(), frequency_hz, kwargs))
        self.last_result = self.result
        if self.failure is not None:
            raise self.failure
        return self.result


@pytest.mark.parametrize("packet_us,gap_us,packet_samples,gap_samples", [
    (20.0, 0.0, 400, 0),
    (20.01, 0.025, 480, 0),
    (24.0, 0.075, 480, 2),
    (24.01, 0.125, 560, 2),
    (22.2, 0.175, 480, 4),
])
def test_loop_uses_symbol_rounded_packets_and_equal_internal_and_final_gaps(
        packet_us, gap_us, packet_samples, gap_samples):
    settings = ContinuousSettings(packet_us=packet_us, gap_us=gap_us, packet_count=3, seed=41)
    iq = generate_loop(settings)
    stride = packet_samples + gap_samples
    assert iq.dtype == np.complex64
    assert iq.size == settings.loop_sample_count == 3 * stride
    assert settings.gap_samples == gap_samples
    for start in range(0, iq.size, stride):
        assert np.any(iq[start:start + packet_samples] != 0)
        np.testing.assert_array_equal(iq[start + packet_samples:start + stride], 0)
    # The final gap also appears at the rewind seam of a repeated IQ file.
    doubled = np.tile(iq, 2)
    np.testing.assert_array_equal(doubled[iq.size - gap_samples:iq.size], 0)
    np.testing.assert_array_equal(doubled[iq.size:iq.size + packet_samples], iq[:packet_samples])
    info = settings.loop_description()
    assert info["actual_packet_us"] == pytest.approx(packet_samples / 20)
    assert info["actual_gap_us"] == pytest.approx(gap_samples / 20)
    assert info["loop_duration_ms"] == pytest.approx(3 * stride / 20_000)
    assert info["nominal_packets_per_second"] == pytest.approx(20_000_000 / stride)
    assert info["nominal_packet_duty_cycle"] == pytest.approx(packet_samples / stride)


def test_loop_seed_is_reproducible_and_changes_data():
    settings = ContinuousSettings(packet_us=28, gap_us=2, packet_count=2, seed=19)
    first = generate_loop(settings)
    np.testing.assert_array_equal(first, generate_loop(settings))
    assert not np.array_equal(first, generate_loop(replace(settings, seed=20)))


@pytest.mark.parametrize("channel", range(1, 14))
def test_all_requested_wifi_channels_are_accepted(channel):
    ContinuousSettings(channel=channel).validate()


@pytest.mark.parametrize("changes", [
    {"channel": 0}, {"channel": 14}, {"channel": True}, {"channel": 6.0},
    {"gain_db": -1}, {"gain_db": 48}, {"gain_db": True}, {"gain_db": 0.5},
    {"digital_amplitude": 0}, {"digital_amplitude": 1.01}, {"digital_amplitude": float("nan")},
    {"packet_us": 19.99}, {"packet_us": float("inf")},
    {"gap_us": -0.01}, {"gap_us": float("nan")},
    {"packet_count": 0}, {"packet_count": 10_001}, {"packet_count": 1.5},
    {"duration_seconds": -1}, {"duration_seconds": float("inf")},
    {"seed": -1}, {"seed": True}, {"output_dir": " "},
    {"rf_amp_enabled": 1}, {"rf_amp_enabled": "1"}, {"rf_amp_enabled": None},
    # Packet duration rounds to 1 second; the additional final gap exceeds the bound.
    {"packet_us": 999_999.99, "gap_us": 0.05, "packet_count": 1},
])
def test_invalid_settings_fail_before_allocation_logging_or_hardware(changes, tmp_path, monkeypatch):
    output = tmp_path / "uncreated"
    settings = replace(ContinuousSettings(output_dir=str(output)), **changes)
    monkeypatch.setattr(continuous, "generate_burst", forbidden)
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    with pytest.raises(ValueError):
        generate_loop(settings)
    with pytest.raises(ValueError):
        run_continuous(settings, transmit=True)
    assert not output.exists()


def test_loop_bound_includes_final_gap_and_accepts_exactly_one_second():
    settings = ContinuousSettings(packet_us=999_996, gap_us=4, packet_count=1)
    settings.validate()
    assert settings.loop_sample_count == 20_000_000
    with pytest.raises(ValueError, match="final gap"):
        replace(settings, gap_us=4.05).validate()


@pytest.mark.parametrize("rf_amp_enabled", [False, True])
def test_dry_run_completes_without_backend_or_temporary_iq(
        tmp_path, monkeypatch, fake_clock, rf_amp_enabled):
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    monkeypatch.setattr(hackrf.tempfile, "TemporaryDirectory", forbidden)
    settings = ContinuousSettings(channel=13, packet_us=20.1, gap_us=0.075,
                                  packet_count=2, duration_seconds=0.35, output_dir=str(tmp_path),
                                  rf_amp_enabled=rf_amp_enabled)
    result = run_continuous(settings, backend=object())
    summary = saved_summary(result)
    assert result.status == "completed"
    assert summary["transmit"] is False
    assert summary["rf_amp_enabled"] is rf_amp_enabled
    assert summary["waveform_generated"] is True
    assert summary["transmission_attempted"] is False
    assert summary["host_result"] is None
    assert summary["error"] is None
    assert summary["center_frequency_hz"] == 2_472_000_000
    assert fake_clock.sleeps == pytest.approx([0.2, 0.15])
    assert summary["elapsed_seconds"] == pytest.approx(0.35)
    snapshot = json.loads((result.directory / "config.json").read_text(encoding="utf-8"))
    assert snapshot["transmit"] is False
    assert snapshot["rf_amp_enabled"] is rf_amp_enabled
    assert snapshot["mode"] == "continuous"
    assert snapshot["packet_us"] == settings.packet_us
    assert "not measured" in summary["timing_reference"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_indefinite_dry_run_interrupt_is_persisted(tmp_path, monkeypatch, interrupt):
    def stop(seconds):
        assert seconds == 0.2
        raise interrupt()

    monkeypatch.setattr(continuous.time, "sleep", stop)
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    result = run_continuous(ContinuousSettings(packet_count=1, output_dir=str(tmp_path)))
    summary = saved_summary(result)
    assert result.status == "interrupted"
    assert summary["host_result"] is None
    assert summary["transmission_attempted"] is False


def test_preflight_failure_is_saved_before_waveform_generation(tmp_path, monkeypatch):
    class Unavailable(FakeRepeater):
        def check_available(self):
            assert self.last_result is None
            self.preflights += 1
            raise RuntimeError("No mock device")

    backend = Unavailable()
    monkeypatch.setattr(continuous, "generate_loop", forbidden)
    result = run_continuous(ContinuousSettings(output_dir=str(tmp_path)), transmit=True, backend=backend)
    summary = saved_summary(result)
    assert result.status == "failed"
    assert summary["error"] == "RuntimeError: No mock device"
    assert summary["waveform_generated"] is False
    assert summary["transmission_attempted"] is False
    assert summary["host_result"] is None
    assert backend.preflights == 1
    assert backend.calls == []


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("rf_amp_enabled", [False, True])
def test_repeat_result_and_requested_rf_controls_are_persisted(tmp_path, success, rf_amp_enabled):
    host = host_result(success=success, error=None if success else "mock early exit")
    backend = FakeRepeater(host)
    settings = ContinuousSettings(channel=1, gain_db=47 if rf_amp_enabled else 17,
                                  digital_amplitude=1 if rf_amp_enabled else 0.25,
                                  packet_us=24, gap_us=1, packet_count=2,
                                  duration_seconds=3, output_dir=str(tmp_path),
                                  rf_amp_enabled=rf_amp_enabled)
    result = run_continuous(settings, transmit=True, backend=backend)
    summary = saved_summary(result)
    assert result.status == ("completed" if success else "failed")
    assert summary["host_result"] == asdict(host)
    assert summary["error"] == host.error
    assert summary["transmission_attempted"] is True
    assert summary["waveform_generated"] is True
    assert summary["rf_amp_enabled"] is rf_amp_enabled
    snapshot = json.loads((result.directory / "config.json").read_text(encoding="utf-8"))
    assert snapshot["rf_amp_enabled"] is rf_amp_enabled
    assert snapshot["gain_db"] == settings.gain_db
    assert snapshot["digital_amplitude"] == settings.digital_amplitude
    assert backend.preflights == 1
    assert len(backend.calls) == 1
    iq, frequency, controls = backend.calls[0]
    np.testing.assert_array_equal(iq, generate_loop(settings))
    assert frequency == 2_412_000_000
    assert controls == {"gain_db": settings.gain_db, "digital_amplitude": settings.digital_amplitude,
                        "duration_seconds": 3, "rf_amp_enabled": rf_amp_enabled}


@pytest.mark.parametrize("failure,status", [(KeyboardInterrupt(), "interrupted"),
                                           (SystemExit(), "interrupted"),
                                           (OSError("mock I/O failure"), "failed")])
def test_interrupted_or_failed_repeat_keeps_last_host_result(tmp_path, failure, status):
    host = host_result(success=False, error="mock stopped")
    backend = FakeRepeater(host, failure=failure)
    result = run_continuous(ContinuousSettings(packet_count=1, output_dir=str(tmp_path)),
                            transmit=True, backend=backend)
    summary = saved_summary(result)
    assert result.status == status
    assert summary["host_result"] == asdict(host)
    assert summary["transmission_attempted"] is True
    if isinstance(failure, OSError):
        assert summary["error"] == "OSError: mock I/O failure"


def feed_menu(monkeypatch, responses):
    values = iter(responses)

    def answer(prompt):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", answer)


def test_menu_retains_invalid_atomic_edit_then_updates_channel_gain_and_restarts(
        tmp_path, monkeypatch, capsys):
    feed_menu(monkeypatch, ["5", "28", "200", "0", "1", "2", "13", "3", "17", "1", "1", "0"])
    calls = []

    def simulated_run(settings, **kwargs):
        calls.append((settings, kwargs))
        status = ["interrupted", "failed", "completed"][len(calls) - 1]
        return ExperimentResult(tmp_path / f"session{len(calls)}", status,
                                {"error": "mock device disconnected" if status == "failed" else None})

    monkeypatch.setattr(continuous, "run_continuous", simulated_run)
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    assert continuous_main(["--output-dir", str(tmp_path)]) == 0
    assert len(calls) == 3
    initial = calls[0][0]
    assert (initial.packet_us, initial.gap_us, initial.packet_count) == (1000, 100, 50)
    assert (initial.channel, initial.gain_db) == (6, 0)
    assert (calls[1][0].channel, calls[1][0].gain_db) == (13, 17)
    assert calls[1][0] == calls[2][0]
    assert all(controls == {"transmit": False, "verbose": False} for _, controls in calls)
    output = capsys.readouterr().out
    assert "Previous settings retained" in output
    assert "Session interrupted" in output
    assert "Session failed" in output
    assert "mock device disconnected" in output
    assert "Session completed" in output


def test_menu_cancels_partial_edit_and_keeps_blank_values(tmp_path, monkeypatch, capsys):
    feed_menu(monkeypatch, ["5", "28", KeyboardInterrupt(), "2", "", "1", "0"])
    calls = []

    def simulated_run(settings, **kwargs):
        calls.append((settings, kwargs))
        return ExperimentResult(tmp_path, "interrupted", {"error": None})

    monkeypatch.setattr(continuous, "run_continuous", simulated_run)
    assert continuous_main(["--channel", "11", "--transmit", "--verbose"]) == 0
    assert len(calls) == 1
    settings, controls = calls[0]
    assert settings.channel == 11
    assert (settings.packet_us, settings.gap_us, settings.packet_count) == (1000, 100, 50)
    assert controls == {"transmit": True, "verbose": True}
    assert "Edit cancelled. Previous settings retained" in capsys.readouterr().out


def test_menu_changes_rf_amp_and_retains_blank_or_invalid_values(tmp_path, monkeypatch, capsys):
    feed_menu(monkeypatch, ["8", "1", "1", "8", "", "1", "8", "2", "1", "8", "0", "1", "0"])
    calls = []

    def simulated_run(settings, **kwargs):
        calls.append((settings, kwargs))
        return ExperimentResult(tmp_path, "interrupted", {"error": None})

    monkeypatch.setattr(continuous, "run_continuous", simulated_run)
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    assert continuous_main(["--transmit", "--gain", "47", "--digital-amplitude", "1"]) == 0
    assert [settings.rf_amp_enabled for settings, _ in calls] == [True, True, True, False]
    assert all(settings.gain_db == 47 and settings.digital_amplitude == 1 for settings, _ in calls)
    assert all(controls == {"transmit": True, "verbose": False} for _, controls in calls)
    assert "Previous settings retained" in capsys.readouterr().out


def test_cli_rf_amp_flag_keeps_dry_run_without_transmit(tmp_path, monkeypatch, fake_clock, capsys):
    feed_menu(monkeypatch, ["1", "0"])
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    monkeypatch.setattr(hackrf.tempfile, "TemporaryDirectory", forbidden)
    assert continuous_main(["--rf-amp", "--gain", "47", "--digital-amplitude", "1",
                            "--duration", "0.35", "--packet-count", "1",
                            "--output-dir", str(tmp_path)]) == 0
    directory = next(tmp_path.iterdir())
    summary = json.loads((directory / "continuous_summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["rf_amp_enabled"] is True
    assert summary["transmit"] is False
    assert summary["transmission_attempted"] is False
    assert summary["host_result"] is None
    assert config["rf_amp_enabled"] is True
    assert config["gain_db"] == 47
    assert config["digital_amplitude"] == 1
    assert "AMP ON" in capsys.readouterr().out


@pytest.mark.parametrize("closing_input", [EOFError(), KeyboardInterrupt()])
def test_menu_closes_cleanly_on_terminal_interrupt(monkeypatch, closing_input, capsys):
    feed_menu(monkeypatch, ["bad menu choice", closing_input])
    monkeypatch.setattr(continuous, "run_continuous", forbidden)
    assert continuous_main([]) == 0
    output = capsys.readouterr().out
    assert "Choose a menu item" in output
    assert "Menu closed" in output


@pytest.mark.parametrize("arguments", [["--channel", "14"], ["--gain", "48"],
                                      ["--duration", "nan"], ["--packet-count", "0"]])
def test_cli_rejects_bad_settings_before_entering_menu(arguments, monkeypatch):
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(continuous, "generate_loop", forbidden)
    monkeypatch.setattr(continuous, "HackRFTransferBackend", forbidden)
    with pytest.raises(SystemExit) as exc:
        continuous_main(arguments)
    assert exc.value.code == 2


def test_script_help_works_from_other_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / "continuous_traffic.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    for option in ("--channel", "--gain", "--digital-amplitude", "--duration", "--transmit", "--rf-amp"):
        assert option in result.stdout
    assert list(tmp_path.iterdir()) == []
