"""Repeating subprocess lifecycle tests; no HackRF or RF access occurs."""

from datetime import datetime
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from wifi_emulator import hackrf


VALID_INFO = (
    "hackrf_info version: 2024.02.1\n"
    "libhackrf version: 2024.02.1 (0.9)\n"
    "Found HackRF\nFirmware Version: 2024.02.1 (API:1.08)\n"
)


class RepeatingProcess:
    """Model a repeat process, including output files and delayed shutdown."""

    def __init__(self, command, options, failure=None, exit_code=0, shutdown_failure=None):
        self.command = command
        self.options = options
        self.path = Path(command[command.index("-t") + 1])
        self.data = self.path.read_bytes()
        self.failure = failure
        self.exit_code = exit_code
        self.shutdown_failure = shutdown_failure
        self.returncode = None
        self.wait_timeouts = []
        self.terminated = False
        self.killed = False
        self.reaped = False
        # Only the tail should survive, even for substantial child output.
        options["stdout"].write(b"old output\n" * 2000 + f"repeated {self.path}".encode())
        options["stderr"].write(b"old diagnostics\n" * 2000 + b"final diagnostic")
        options["stdout"].flush()
        options["stderr"].flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        assert self.path.exists(), "IQ removed before child was reaped"
        assert not self.options["stdout"].closed
        assert not self.options["stderr"].closed
        self.wait_timeouts.append(timeout)
        if len(self.wait_timeouts) == 1:
            if self.failure is not None:
                raise self.failure
            self.returncode = self.exit_code
        elif len(self.wait_timeouts) == 2 and self.shutdown_failure is not None:
            raise self.shutdown_failure
        else:
            self.returncode = -9 if self.killed else -15
        self.reaped = True
        return self.returncode

    def terminate(self):
        assert self.path.exists()
        self.terminated = True

    def kill(self):
        assert self.path.exists()
        self.killed = True


@pytest.fixture
def iq():
    return np.array([1 + 1j, 0j, -1 - 1j, 0.5 + 0.25j], dtype=np.complex64)


def mock_repeat(monkeypatch, failure=None, exit_code=0, shutdown_failure=None, info_output=VALID_INFO):
    processes = []
    queries = []
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: f"mock-{name}")

    def query(command, **options):
        queries.append(command)
        return SimpleNamespace(returncode=0, stdout=info_output, stderr="")

    def popen(command, **options):
        process = RepeatingProcess(command, options, failure, exit_code, shutdown_failure)
        processes.append(process)
        return process

    monkeypatch.setattr(hackrf.subprocess, "run", query)
    monkeypatch.setattr(hackrf.subprocess, "Popen", popen)
    return processes, queries


def assert_cleaned(process):
    assert process.reaped
    assert not process.path.exists()
    assert not process.path.parent.exists()
    assert process.options["stdout"].closed
    assert process.options["stderr"].closed


def test_timed_repeat_flags_disk_output_and_cleanup(monkeypatch, iq):
    processes, queries = mock_repeat(monkeypatch, failure=subprocess.TimeoutExpired("mock-transfer", 60))
    backend = hackrf.HackRFTransferBackend(timeout_seconds=0.1)
    result = backend.transmit_repeating(iq, 2_437_000_000, gain_db=17, duration_seconds=60)
    process = processes[0]
    assert result.success
    assert result.error is None
    assert result.return_code == -15  # Intentional host shutdown retains child status.
    assert backend.last_result is result
    assert queries == [["mock-hackrf_info"]]
    assert process.wait_timeouts == [60.0, 2.0]
    assert process.terminated and not process.killed
    assert process.command[-1] == "-R"
    assert "-n" not in process.command
    flags = dict(zip(process.command[1:-1:2], process.command[2:-1:2]))
    assert flags["-f"] == "2437000000"
    assert flags["-s"] == flags["-b"] == "20000000"
    assert flags["-x"] == "17"
    assert flags["-a"] == flags["-p"] == "0"
    assert len(process.data) == 2 * iq.size
    assert process.path.suffix == ".cs8"
    assert process.options["shell"] is False
    assert process.options["stdin"] == subprocess.DEVNULL
    assert process.options["stdout"] is not subprocess.PIPE
    assert process.options["stderr"] is not subprocess.PIPE
    assert len(result.stdout) <= 8192 and len(result.stderr) <= 8192
    assert result.stdout.endswith("repeated <temporary IQ>")
    assert result.stderr.endswith("final diagnostic")
    assert str(process.path) not in result.stdout
    assert result.host_elapsed_seconds >= 0
    assert datetime.fromisoformat(result.host_end_timestamp) >= datetime.fromisoformat(result.host_start_timestamp)
    assert_cleaned(process)


@pytest.mark.parametrize("rf_amp_enabled", [False, True])
def test_repeat_explicit_amp_with_maximum_gain_and_amplitude(monkeypatch, rf_amp_enabled):
    processes, _ = mock_repeat(monkeypatch, failure=subprocess.TimeoutExpired("mock-transfer", 1))
    iq = np.array([1 + 0j, -1 + 0j, 0 + 1j, 0 - 1j], dtype=np.complex64)
    result = hackrf.HackRFTransferBackend().transmit_repeating(
        iq,
        2_412_000_000,
        gain_db=47,
        digital_amplitude=1.0,
        duration_seconds=1,
        rf_amp_enabled=rf_amp_enabled,
    )
    assert result.success
    process = processes[0]
    flags = dict(zip(process.command[1:-1:2], process.command[2:-1:2]))
    assert flags["-a"] == ("1" if rf_amp_enabled else "0")
    assert flags["-x"] == "47"
    assert flags["-p"] == "0"
    assert np.frombuffer(process.data, dtype=np.int8).tolist() == [127, 0, -127, 0, 0, 127, 0, -127]
    assert_cleaned(process)


@pytest.mark.parametrize("rf_amp_enabled", [0, 1, "true", "false", None, np.bool_(True)])
def test_nonboolean_amp_is_rejected_before_quantization_or_hardware(monkeypatch, iq, rf_amp_enabled):
    processes, queries = mock_repeat(monkeypatch)

    def forbidden_quantization(*args, **kwargs):
        pytest.fail("Invalid RF amplifier input must be rejected before quantization")

    monkeypatch.setattr(hackrf, "quantize_iq", forbidden_quantization)
    result = hackrf.HackRFTransferBackend().transmit_repeating(
        iq, 2_412_000_000, rf_amp_enabled=rf_amp_enabled,
    )
    assert not result.success
    assert "rf_amp_enabled must be a boolean" in result.error
    assert result.host_start_timestamp == result.host_end_timestamp == ""
    assert result.host_elapsed_seconds == 0
    assert processes == queries == []


@pytest.mark.parametrize("duration", [0, 3600])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_early_child_exit_is_failure_even_when_zero(monkeypatch, iq, duration, exit_code):
    processes, _ = mock_repeat(monkeypatch, exit_code=exit_code)
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000, duration_seconds=duration)
    assert not result.success
    assert result.return_code == exit_code
    assert f"exited unexpectedly with code {exit_code}" in result.error
    assert processes[0].wait_timeouts == [float(duration) if duration else None]
    assert not processes[0].terminated
    assert_cleaned(processes[0])


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_exit_between_duration_timeout_and_shutdown_is_failure(monkeypatch, iq, exit_code):
    processes, _ = mock_repeat(monkeypatch, failure=subprocess.TimeoutExpired("mock-transfer", 1))
    original_wait = RepeatingProcess.wait

    def exited_at_deadline(self, timeout=None):
        # wait timed out, but the following poll discovers the child exited
        # before the backend requested a stop. This is not a planned shutdown.
        self.returncode = exit_code
        self.reaped = True
        return original_wait(self, timeout)

    monkeypatch.setattr(RepeatingProcess, "wait", exited_at_deadline)
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000, duration_seconds=1)
    assert not result.success
    assert result.return_code == exit_code
    assert "exited unexpectedly" in result.error
    assert not processes[0].terminated
    assert processes[0].wait_timeouts == [1.0]
    assert_cleaned(processes[0])


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_indefinite_repeat_interrupt_propagates_after_cleanup(monkeypatch, iq, exception_type):
    processes, _ = mock_repeat(monkeypatch, failure=exception_type())
    backend = hackrf.HackRFTransferBackend(timeout_seconds=0.1)
    with pytest.raises(exception_type):
        backend.transmit_repeating(iq, 2_412_000_000)
    result = backend.last_result
    assert result is not None and not result.success
    assert exception_type.__name__ in result.error
    assert result.return_code == -15
    assert processes[0].wait_timeouts == [None, 2.0]
    assert processes[0].terminated
    assert_cleaned(processes[0])


@pytest.mark.parametrize("shutdown_failure", [subprocess.TimeoutExpired("mock-transfer", 2), KeyboardInterrupt()])
def test_shutdown_escalates_to_kill_and_reaps(monkeypatch, iq, shutdown_failure):
    processes, _ = mock_repeat(
        monkeypatch,
        failure=subprocess.TimeoutExpired("mock-transfer", 1),
        shutdown_failure=shutdown_failure,
    )
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000, duration_seconds=1)
    assert result.success
    assert result.return_code == -9
    assert processes[0].terminated and processes[0].killed
    assert processes[0].wait_timeouts == [1.0, 2.0, None]
    assert_cleaned(processes[0])


def test_wait_error_stops_child_before_removing_input(monkeypatch, iq):
    processes, _ = mock_repeat(monkeypatch, failure=OSError("wait failed"))
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000)
    assert not result.success
    assert "wait failed" in result.error
    assert result.return_code == -15
    assert processes[0].terminated
    assert_cleaned(processes[0])


def test_launch_error_closes_all_temporary_files(monkeypatch, iq):
    mock_repeat(monkeypatch)
    paths = []
    streams = []

    def fail_launch(command, **options):
        path = Path(command[command.index("-t") + 1])
        paths.append(path)
        streams.extend([options["stdout"], options["stderr"]])
        assert path.read_bytes()
        raise OSError(f"could not launch {path}")

    monkeypatch.setattr(hackrf.subprocess, "Popen", fail_launch)
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000)
    assert not result.success
    assert result.return_code is None
    assert "could not launch <temporary IQ>" in result.error
    assert not paths[0].exists() and not paths[0].parent.exists()
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), float("-inf"), True])
def test_invalid_duration_never_queries_or_launches(monkeypatch, iq, duration):
    processes, queries = mock_repeat(monkeypatch)
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000, duration_seconds=duration)
    assert not result.success
    assert "duration_seconds must be finite and nonnegative" in result.error
    assert result.host_start_timestamp == result.host_end_timestamp == ""
    assert result.host_elapsed_seconds == 0
    assert processes == queries == []


@pytest.mark.parametrize("setting", [{"gain_db": 48}, {"sample_rate": 10_000_000}, {"frequency_hz": -1}, {"digital_amplitude": 2}])
def test_repeat_retains_finite_validation(monkeypatch, iq, setting):
    processes, queries = mock_repeat(monkeypatch)
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, **{"frequency_hz": 2_412_000_000, **setting})
    assert not result.success
    assert result.host_start_timestamp == ""
    assert processes == queries == []


def test_repeat_requires_same_compatible_preflight(monkeypatch, iq):
    processes, queries = mock_repeat(monkeypatch, info_output=VALID_INFO.replace("2024.02.1", "2023.01.1"))
    result = hackrf.HackRFTransferBackend().transmit_repeating(iq, 2_412_000_000)
    assert not result.success
    assert "unsupported or unverified host version" in result.error
    assert queries == [["mock-hackrf_info"]]
    assert processes == []


def test_repeat_reuses_preflight_and_prints_only_final_tails(monkeypatch, iq, capsys):
    processes, queries = mock_repeat(monkeypatch, failure=subprocess.TimeoutExpired("mock-transfer", 1))
    backend = hackrf.HackRFTransferBackend(verbose=True)
    backend.check_available()
    for _ in range(2):
        assert backend.transmit_repeating(iq, 2_412_000_000, duration_seconds=1).success
    assert queries == [["mock-hackrf_info"]]
    assert len(processes) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert len(output.err) <= 4 * 8193
    assert "<temporary IQ>" in output.err
    assert all(str(process.path) not in output.err for process in processes)
    for process in processes:
        assert_cleaned(process)
