"""Subprocess and temporary-file contract tests; no HackRF access occurs."""

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
    "Found HackRF\nBoard ID Number: 2 (HackRF One)\n"
    "Firmware Version: 2024.02.1 (API:1.08)\n"
)


def mock_available(monkeypatch, output=VALID_INFO):
    calls = []
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: f"mock-{name}")

    def run(command, **options):
        calls.append(command)
        assert options["shell"] is False
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(hackrf.subprocess, "run", run)
    return calls


class FakeProcess:
    def __init__(self, command, options, failure=None, shutdown_timeout=False):
        self.command = command
        self.options = options
        self.path = Path(command[command.index("-t") + 1])
        self.data = self.path.read_bytes()
        self.returncode = None
        self.failure = failure
        self.shutdown_timeout = shutdown_timeout
        self.communications = 0
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        assert self.path.exists(), "IQ file removed before the child was reaped"
        self.communications += 1
        if self.communications == 1:
            if self.failure is not None:
                raise self.failure
            self.returncode = 0
        elif self.communications == 2 and self.shutdown_timeout:
            self.returncode = None
            raise subprocess.TimeoutExpired(self.command, timeout)
        return f"sent {self.path}", ""

    def terminate(self):
        assert self.path.exists()
        self.terminated = True
        self.returncode = -15

    def kill(self):
        assert self.path.exists()
        self.killed = True
        self.returncode = -9

    def wait(self):
        assert self.path.exists()
        self.waited = True
        return self.returncode


@pytest.fixture
def iq():
    return np.array([1 + 1j, 0j, -1 - 1j, 0.5 + 0.25j], dtype=np.complex64)


def mock_transfer(monkeypatch, failure=None, shutdown_timeout=False):
    processes = []
    mock_available(monkeypatch)

    def popen(command, **options):
        process = FakeProcess(command, options, failure, shutdown_timeout)
        processes.append(process)
        return process

    monkeypatch.setattr(hackrf.subprocess, "Popen", popen)
    return processes


def assert_cleaned(process):
    assert not process.path.exists()
    assert not process.path.parent.exists()


def test_success_command_quantization_timestamps_and_cleanup(monkeypatch, iq):
    processes = mock_transfer(monkeypatch)
    backend = hackrf.HackRFTransferBackend()
    result = backend.transmit(iq, 2_437_000_000)
    process = processes[0]
    assert result.success
    assert result.return_code == 0
    assert backend.last_result is result
    assert result.host_elapsed_seconds >= 0
    assert datetime.fromisoformat(result.host_end_timestamp) >= datetime.fromisoformat(result.host_start_timestamp)
    assert datetime.fromisoformat(result.host_start_timestamp).utcoffset().total_seconds() == 0
    assert isinstance(process.command, list)
    assert process.options["shell"] is False
    flags = dict(zip(process.command[1::2], process.command[2::2]))
    assert flags["-f"] == "2437000000"
    assert flags["-s"] == "20000000"
    assert flags["-n"] == str(iq.size)
    assert flags["-b"] == "20000000"
    assert flags["-a"] == flags["-p"] == flags["-x"] == "0"
    assert len(process.data) == 2 * iq.size
    assert process.path.suffix == ".cs8"
    assert str(process.path) not in result.stdout
    assert "<temporary IQ>" in result.stdout
    assert_cleaned(process)


def test_nonzero_exit_returns_failure_and_removes_file(monkeypatch, iq):
    processes = mock_transfer(monkeypatch)
    original = FakeProcess.communicate

    def failed(self, timeout=None):
        stdout, _ = original(self, timeout)
        self.returncode = 7
        return stdout, "USB transfer failed"

    monkeypatch.setattr(FakeProcess, "communicate", failed)
    result = hackrf.HackRFTransferBackend().transmit(iq, 2_412_000_000)
    assert not result.success
    assert result.return_code == 7
    assert result.error == "hackrf_transfer exited with code 7"
    assert result.stderr == "USB transfer failed"
    assert_cleaned(processes[0])


@pytest.mark.parametrize("failure", [OSError("pipe failure"), RuntimeError("unexpected error")])
def test_communication_exception_stops_child_and_cleans_up(monkeypatch, iq, failure):
    processes = mock_transfer(monkeypatch, failure=failure)
    result = hackrf.HackRFTransferBackend().transmit(iq, 2_412_000_000)
    assert not result.success
    assert type(failure).__name__ in result.error
    assert processes[0].terminated
    assert processes[0].returncode is not None
    assert_cleaned(processes[0])


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_is_reraised_after_cleanup_with_last_result(monkeypatch, iq, exception_type):
    processes = mock_transfer(monkeypatch, failure=exception_type())
    backend = hackrf.HackRFTransferBackend()
    with pytest.raises(exception_type):
        backend.transmit(iq, 2_412_000_000)
    assert backend.last_result is not None
    assert not backend.last_result.success
    assert exception_type.__name__ in backend.last_result.error
    assert processes[0].terminated
    assert_cleaned(processes[0])


def test_timeout_escalates_to_kill_then_removes_file(monkeypatch, iq):
    processes = mock_transfer(
        monkeypatch,
        failure=subprocess.TimeoutExpired("mock-hackrf_transfer", 0.1),
        shutdown_timeout=True,
    )
    backend = hackrf.HackRFTransferBackend(timeout_seconds=0.1)
    result = backend.transmit(iq, 2_412_000_000)
    assert not result.success
    assert result.error == "hackrf_transfer timed out after 0.1 seconds"
    assert processes[0].terminated
    assert processes[0].killed
    assert processes[0].waited
    assert result.return_code == -9
    assert_cleaned(processes[0])


def test_launch_exception_removes_closed_temporary_file(monkeypatch, iq):
    paths = []
    mock_available(monkeypatch)

    def fail_launch(command, **options):
        path = Path(command[command.index("-t") + 1])
        paths.append(path)
        # Opening the existing file is also possible on Windows because its
        # writer was closed before launching the child.
        with path.open("rb") as handle:
            assert len(handle.read()) == 2 * iq.size
        raise OSError("Cannot start process")

    monkeypatch.setattr(hackrf.subprocess, "Popen", fail_launch)
    result = hackrf.HackRFTransferBackend().transmit(iq, 2_412_000_000)
    assert not result.success
    assert result.return_code is None
    assert "Cannot start process" in result.error
    assert not paths[0].exists()
    assert not paths[0].parent.exists()


def test_invalid_settings_never_launch(monkeypatch, iq):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid inputs must not launch a transmitter")

    monkeypatch.setattr(hackrf.subprocess, "Popen", forbidden)
    backend = hackrf.HackRFTransferBackend()
    for setting in ({"gain_db": 48}, {"sample_rate": 10_000_000}, {"frequency_hz": -1}):
        args = {"frequency_hz": 2_412_000_000, **setting}
        result = backend.transmit(iq, **args)
        assert not result.success
        assert result.host_start_timestamp == result.host_end_timestamp == ""
        assert result.host_elapsed_seconds == 0


def test_check_available_queries_info_only(monkeypatch):
    calls = mock_available(monkeypatch)
    output = hackrf.check_hackrf()
    assert "HackRF One" in output
    assert calls == [["mock-hackrf_info"]]


@pytest.mark.parametrize("version", ["git-7047252", "2023.01.1", "2024.01.1", "unknown", "git-18b485e3", ""])
def test_check_available_rejects_legacy_or_unverified_host_builds(monkeypatch, version):
    output = VALID_INFO.replace("hackrf_info version: 2024.02.1", f"hackrf_info version: {version}")
    mock_available(monkeypatch, output)
    backend = hackrf.HackRFTransferBackend()
    with pytest.raises(RuntimeError) as failure:
        backend.check_available()
    message = str(failure.value)
    assert "unsupported or unverified host version" in message
    assert version in message
    assert "mock-hackrf_info" in message and "mock-hackrf_transfer" in message
    assert "final IQ buffer" in message
    assert "conda activate wifi-hackrf" in message
    assert "where.exe hackrf_transfer" in message
    assert "hackrf=2024.02.1" in message
    assert "Detected firmware: 2024.02.1" in message
    assert backend._transfer_executable is None


@pytest.mark.parametrize("version", ["2024.02.1", "2026.01.3", "v2024.02.1"])
def test_check_available_accepts_supported_releases(monkeypatch, version):
    mock_available(monkeypatch, VALID_INFO.replace("hackrf_info version: 2024.02.1", f"hackrf_info version: {version}"))
    backend = hackrf.HackRFTransferBackend()
    assert "Found HackRF" in backend.check_available()
    assert backend._transfer_executable == "mock-hackrf_transfer"


def test_check_available_rejects_old_loaded_library(monkeypatch):
    mock_available(monkeypatch, VALID_INFO.replace("libhackrf version: 2024.02.1 (0.9)", "libhackrf version: git-7047252 (0.6)"))
    with pytest.raises(RuntimeError, match="unsupported libhackrf 0.6"):
        hackrf.check_hackrf()


def test_check_available_rejects_mixed_installation_paths(monkeypatch, tmp_path):
    calls = mock_available(monkeypatch)
    info = tmp_path / "old" / "hackrf_info"
    transfer = tmp_path / "conda" / "hackrf_transfer"
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: str(info if name == "hackrf_info" else transfer))
    with pytest.raises(RuntimeError, match="different directories") as failure:
        hackrf.check_hackrf()
    assert str(info) in str(failure.value)
    assert str(transfer) in str(failure.value)
    assert calls == []


def test_direct_transmit_requires_compatible_preflight(monkeypatch, iq):
    processes = mock_transfer(monkeypatch)
    calls = mock_available(monkeypatch, VALID_INFO.replace("hackrf_info version: 2024.02.1", "hackrf_info version: git-7047252"))
    result = hackrf.HackRFTransferBackend().transmit(iq, 2_437_000_000)
    assert not result.success
    assert "git-7047252" in result.error
    assert result.return_code is None
    assert result.host_start_timestamp == ""
    assert calls == [["mock-hackrf_info"]]
    assert processes == []


def test_transmit_reuses_successful_preflight(monkeypatch, iq):
    processes = mock_transfer(monkeypatch)
    calls = mock_available(monkeypatch)
    backend = hackrf.HackRFTransferBackend()
    backend.check_available()
    assert backend.transmit(iq, 2_437_000_000).success
    assert backend.transmit(iq, 2_437_000_000).success
    assert calls == [["mock-hackrf_info"]]
    assert len(processes) == 2


def test_failed_recheck_clears_previously_verified_executable(monkeypatch, iq):
    processes = mock_transfer(monkeypatch)
    backend = hackrf.HackRFTransferBackend()
    backend.check_available()
    mock_available(monkeypatch, VALID_INFO.replace("hackrf_info version: 2024.02.1", "hackrf_info version: git-7047252"))
    with pytest.raises(RuntimeError):
        backend.check_available()
    assert not backend.transmit(iq, 2_437_000_000).success
    assert processes == []


@pytest.mark.parametrize("missing", ["hackrf_info", "hackrf_transfer"])
def test_check_available_requires_both_utilities(monkeypatch, missing):
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: None if name == missing else f"mock-{name}")
    with pytest.raises(RuntimeError, match=missing):
        hackrf.check_hackrf()


@pytest.mark.parametrize("returncode,stdout", [(0, "hackrf_info version: 1\nNo HackRF boards found"), (1, "USB access denied")])
def test_check_available_requires_detected_device(monkeypatch, returncode, stdout):
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: f"mock-{name}")
    monkeypatch.setattr(hackrf.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=""))
    with pytest.raises(RuntimeError):
        hackrf.check_hackrf()


def test_check_available_wraps_timeout(monkeypatch):
    monkeypatch.setattr(hackrf.shutil, "which", lambda name: f"mock-{name}")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("hackrf_info", 30)

    monkeypatch.setattr(hackrf.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Could not query HackRF devices"):
        hackrf.check_hackrf()
