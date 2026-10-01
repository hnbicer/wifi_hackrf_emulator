"""HackRF utility discovery and temporary-file transmission.

Timing here brackets the host subprocess, not the RF air interface. IQ files
live only in an operating-system temporary directory; the child is stopped and
reaped before that directory is removed, including on Ctrl+C.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from .quantization import quantize_iq


@dataclass(frozen=True)
class TransmissionResult:
    """A completed host invocation; success does not certify RF delivery."""

    host_start_timestamp: str
    host_end_timestamp: str
    host_elapsed_seconds: float
    return_code: int | None
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class TransmitterBackend(ABC):
    """Backend contract independent of traffic scheduling and waveforms."""

    last_result: TransmissionResult | None = None

    @abstractmethod
    def transmit(
        self,
        iq: np.ndarray,
        frequency_hz: int,
        sample_rate: int = 20_000_000,
        gain_db: int = 0,
        digital_amplitude: float = 0.6,
    ) -> TransmissionResult:
        """Transmit one burst; retain an interrupted attempt in last_result."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _host_tools_error(reason: str, info: str, transfer: str, output: str = "") -> RuntimeError:
    firmware = re.search(r"(?im)^[ \t]*Firmware Version:[ \t]*([^\r\n]+)", output)
    firmware_note = f" Detected firmware: {firmware.group(1).strip()}." if firmware else ""
    return RuntimeError(
        f"HackRF short-burst compatibility check failed: {reason}\n"
        f"hackrf_info: {info}\nhackrf_transfer: {transfer}\n"
        "This backend requires a numbered HackRF host release >= 2024.02.1. "
        "Legacy tools can discard the final IQ buffer while returning success. "
        "Use 'conda activate wifi-hackrf', then 'where.exe hackrf_transfer' and "
        "'where.exe hackrf_info' to check the selected installation. "
        "For firmware 2024.02.1, select the matching conda-forge hackrf=2024.02.1 "
        "host package (conda install -c conda-forge hackrf=2024.02.1). "
        "Keep both tools in the same installation; select host tools matching your firmware."
        f"{firmware_note} No firmware update is performed by this project."
    )


def _check_host_version(info: str, transfer: str, output: str) -> None:
    """Require a released toolchain with final-buffer delivery and TX flushing."""
    version_line = re.search(r"(?im)^[ \t]*hackrf_info version:[ \t]*([^\r\n]+)", output)
    version = version_line.group(1).strip() if version_line else "not reported"
    release = re.fullmatch(r"v?(\d{4})\.(\d{2})\.(\d+)", version)
    if release is None or tuple(map(int, release.groups())) < (2024, 2, 1):
        raise _host_tools_error(f"unsupported or unverified host version '{version}'.", info, transfer, output)
    library = re.search(r"(?im)^\s*libhackrf version:[^\r\n]*\((\d+)\.(\d+)(?:\.\d+)?\)", output)
    if library is not None and tuple(map(int, library.groups())) < (0, 9):
        raise _host_tools_error(
            f"host version '{version}' loaded unsupported libhackrf {library.group(1)}.{library.group(2)}; "
            "libhackrf >= 0.9 is required.", info, transfer, output,
        )


def _stop_and_reap(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Stop a live child and drain pipes before deleting its input file.

    A second interrupt during graceful shutdown escalates to kill. Waiting
    after kill is necessary on Windows, where an open file cannot be deleted.
    """
    try:
        process.terminate()
    except OSError:
        # The child may have exited between poll() and terminate().
        pass
    try:
        return process.communicate(timeout=2.0)
    except BaseException:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            if process.poll() is None:
                raise
        try:
            return process.communicate()
        finally:
            # communicate normally reaps, but wait also covers a read error.
            process.wait()


def _stop_repeating_and_reap(process: subprocess.Popen[str]) -> None:
    """Stop a child whose output goes to files, then release its IQ input."""
    try:
        process.terminate()
    except OSError:
        # It may have exited between the last poll and terminate.
        pass
    try:
        process.wait(timeout=2.0)
    except BaseException:
        # A second interrupt or an unresponsive child must not leave RF active.
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            if process.poll() is None:
                raise
        process.wait()


def _output_tail(stream, limit: int = 8192) -> str:
    """Read a bounded UTF-8 tail from a disk-backed child output stream."""
    stream.seek(0, os.SEEK_END)
    length = stream.tell()
    stream.seek(max(0, length - limit))
    return stream.read(limit).decode("utf-8", errors="replace")


class HackRFTransferBackend(TransmitterBackend):
    """Launch finite bursts or repeating sessions with RF amplification off by default."""

    def __init__(self, timeout_seconds: float = 30.0, verbose: bool = False):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self.verbose = verbose
        self.last_result: TransmissionResult | None = None
        self._transfer_executable: str | None = None

    def check_available(self) -> str:
        """Find both utilities and inspect attached devices without transmitting.

        Require one matching, released host installation supporting short files.
        Raise RuntimeError on incompatible tools, command failure, or no device.
        """
        self._transfer_executable = None
        info = shutil.which("hackrf_info")
        transfer = shutil.which("hackrf_transfer")
        missing = [name for name, path in (("hackrf_info", info), ("hackrf_transfer", transfer)) if path is None]
        if missing:
            raise RuntimeError(
                f"Missing HackRF utilities: {', '.join(missing)}. Install the HackRF host tools and add their directory to PATH."
            )
        if os.path.normcase(str(Path(info).resolve().parent)) != os.path.normcase(str(Path(transfer).resolve().parent)):
            raise _host_tools_error("utilities resolve to different directories; host version is unverified.", info, transfer)
        try:
            result = subprocess.run(
                [info],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not query HackRF devices: {exc}") from exc
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        if result.returncode != 0:
            raise RuntimeError(f"hackrf_info failed (exit {result.returncode}): {output[-8192:]}")
        if re.search(r"(?im)^\s*Found HackRF(?:\s|$)", output) is None:
            raise RuntimeError(f"No HackRF device detected. Check USB connection and drivers.\n{output[-8192:]}")
        _check_host_version(info, transfer, output)
        self._transfer_executable = transfer
        return output

    def transmit(
        self,
        iq: np.ndarray,
        frequency_hz: int,
        sample_rate: int = 20_000_000,
        gain_db: int = 0,
        digital_amplitude: float = 0.6,
    ) -> TransmissionResult:
        """Transmit a temporary signed-int8 IQ file and always remove it.

        Ordinary failures return a failed result. KeyboardInterrupt and
        SystemExit propagate after cleanup, with last_result available for the
        caller to log. The RF amplifier and antenna-port power are always off.
        """
        # Blank timestamps distinguish validation/quantization failures from
        # attempts to launch a hardware process.
        start_timestamp = ""
        start_clock: float | None = None
        end_timestamp: str | None = None
        elapsed: float | None = None
        process: subprocess.Popen[str] | None = None
        temp_path: Path | None = None
        stdout = stderr = ""
        error: str | None = None
        return_code: int | None = None
        try:
            if isinstance(frequency_hz, bool) or int(frequency_hz) != frequency_hz or not 0 <= frequency_hz <= 7_250_000_000:
                raise ValueError("frequency_hz must be an integer from 0 to 7250000000")
            if sample_rate != 20_000_000:
                raise ValueError("The Wi-Fi-like backend requires a 20000000 sample/s rate")
            if isinstance(gain_db, bool) or int(gain_db) != gain_db or not 0 <= gain_db <= 47:
                raise ValueError("gain_db must be an integer from 0 to 47")
            interleaved = quantize_iq(iq, digital_amplitude)
            if interleaved.size == 0:
                raise ValueError("Cannot transmit an empty waveform")
            if self._transfer_executable is None:
                self.check_available()
            executable = self._transfer_executable
            with tempfile.TemporaryDirectory(prefix="wifi_hackrf_") as temporary_directory:
                temp_path = Path(temporary_directory) / "burst.cs8"
                # write_bytes closes the file before Popen opens it on Windows.
                temp_path.write_bytes(interleaved.tobytes())
                command = [
                    executable,
                    "-t", str(temp_path),
                    "-f", str(int(frequency_hz)),
                    "-s", str(sample_rate),
                    "-x", str(int(gain_db)),
                    "-a", "0",
                    "-p", "0",
                    "-b", "20000000",
                    "-n", str(interleaved.size // 2),
                ]
                start_timestamp = _utc_now()
                start_clock = time.perf_counter()
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        errors="replace",
                        shell=False,
                    )
                    stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                finally:
                    if process is not None:
                        if process.poll() is None:
                            stdout, stderr = _stop_and_reap(process)
                        return_code = process.returncode
                    end_timestamp = _utc_now()
                    elapsed = time.perf_counter() - start_clock
                if return_code != 0:
                    error = f"hackrf_transfer exited with code {return_code}"
        except BaseException as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                error = f"hackrf_transfer timed out after {self.timeout_seconds:g} seconds"
            else:
                error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            if not isinstance(exc, Exception):
                raise
        finally:
            # Never retain a temporary waveform filename, even if a tool prints
            # it or embeds its argument list into an exception message.
            def concise(value: str | bytes | None) -> str:
                result = _text(value)
                if temp_path is not None:
                    result = result.replace(str(temp_path), "<temporary IQ>")
                    result = result.replace(str(temp_path.parent), "<temporary directory>")
                return result[-8192:]

            self.last_result = TransmissionResult(
                host_start_timestamp=start_timestamp,
                host_end_timestamp=end_timestamp or (_utc_now() if start_timestamp else ""),
                host_elapsed_seconds=(elapsed if elapsed is not None else
                                      time.perf_counter() - start_clock if start_clock is not None else 0.0),
                return_code=return_code,
                success=error is None and return_code == 0,
                stdout=concise(stdout),
                stderr=concise(stderr),
                error=concise(error) if error is not None else None,
            )
            if self.verbose:
                if self.last_result.stdout:
                    print(self.last_result.stdout, file=sys.stderr)
                if self.last_result.stderr:
                    print(self.last_result.stderr, file=sys.stderr)
        return self.last_result

    def transmit_repeating(
        self,
        iq: np.ndarray,
        frequency_hz: int,
        sample_rate: int = 20_000_000,
        gain_db: int = 0,
        digital_amplitude: float = 0.6,
        duration_seconds: float = 0,
        rf_amp_enabled: bool = False,
    ) -> TransmissionResult:
        """Repeat a temporary IQ file until interrupted or a duration elapses.

        A zero duration waits indefinitely. A positive duration stops the child
        and returns host success when shutdown completes, preserving its actual
        exit code. An early child exit is a failure, including exit code zero.
        Interrupts propagate after cleanup with last_result available. The
        finite-burst timeout only applies to the device preflight in this mode.

        Output is captured on disk while running and only its final 8192-byte
        tails are retained in memory. RF amplification is disabled by default
        and can be explicitly enabled for this repeating session. Antenna-port
        power remains disabled.
        """
        start_timestamp = ""
        start_clock: float | None = None
        end_timestamp: str | None = None
        elapsed: float | None = None
        process: subprocess.Popen[str] | None = None
        temp_path: Path | None = None
        stdout = stderr = ""
        error: str | None = None
        return_code: int | None = None
        duration_elapsed = False
        stop_requested = False
        try:
            if not isinstance(rf_amp_enabled, bool):
                raise ValueError("rf_amp_enabled must be a boolean")
            if isinstance(frequency_hz, bool) or int(frequency_hz) != frequency_hz or not 0 <= frequency_hz <= 7_250_000_000:
                raise ValueError("frequency_hz must be an integer from 0 to 7250000000")
            if sample_rate != 20_000_000:
                raise ValueError("The Wi-Fi-like backend requires a 20000000 sample/s rate")
            if isinstance(gain_db, bool) or int(gain_db) != gain_db or not 0 <= gain_db <= 47:
                raise ValueError("gain_db must be an integer from 0 to 47")
            if isinstance(duration_seconds, bool) or not math.isfinite(duration_seconds) or duration_seconds < 0:
                raise ValueError("duration_seconds must be finite and nonnegative; 0 means until interrupted")
            interleaved = quantize_iq(iq, digital_amplitude)
            if interleaved.size == 0:
                raise ValueError("Cannot transmit an empty waveform")
            if self._transfer_executable is None:
                self.check_available()
            executable = self._transfer_executable
            with tempfile.TemporaryDirectory(prefix="wifi_hackrf_repeat_") as temporary_directory:
                temp_path = Path(temporary_directory) / "repeat.cs8"
                temp_path.write_bytes(interleaved.tobytes())
                command = [
                    executable,
                    "-t", str(temp_path),
                    "-f", str(int(frequency_hz)),
                    "-s", str(sample_rate),
                    "-x", str(int(gain_db)),
                    "-a", str(int(rf_amp_enabled)),
                    "-p", "0",
                    "-b", "20000000",
                    "-R",
                ]
                # A live repeat can produce output for hours. File handles avoid
                # pipe backpressure and communicate() accumulating it in RAM.
                with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory) as stdout_file, \
                        tempfile.TemporaryFile(mode="w+b", dir=temporary_directory) as stderr_file:
                    start_timestamp = _utc_now()
                    start_clock = time.perf_counter()
                    try:
                        process = subprocess.Popen(
                            command,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            text=True,
                            errors="replace",
                            shell=False,
                        )
                        try:
                            process.wait(timeout=float(duration_seconds) if duration_seconds else None)
                        except subprocess.TimeoutExpired:
                            duration_elapsed = True
                    finally:
                        try:
                            if process is not None:
                                if process.poll() is None:
                                    stop_requested = duration_elapsed
                                    _stop_repeating_and_reap(process)
                                return_code = process.returncode
                        finally:
                            end_timestamp = _utc_now()
                            elapsed = time.perf_counter() - start_clock
                            stdout = _output_tail(stdout_file)
                            stderr = _output_tail(stderr_file)
                if not (duration_elapsed and stop_requested):
                    error = f"hackrf_transfer exited unexpectedly with code {return_code} while repeating"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            if not isinstance(exc, Exception):
                raise
        finally:
            def concise(value: str | bytes | None) -> str:
                result = _text(value)
                if temp_path is not None:
                    result = result.replace(str(temp_path), "<temporary IQ>")
                    result = result.replace(str(temp_path.parent), "<temporary directory>")
                return result[-8192:]

            self.last_result = TransmissionResult(
                host_start_timestamp=start_timestamp,
                host_end_timestamp=end_timestamp or (_utc_now() if start_timestamp else ""),
                host_elapsed_seconds=(elapsed if elapsed is not None else
                                      time.perf_counter() - start_clock if start_clock is not None else 0.0),
                return_code=return_code,
                success=error is None and duration_elapsed and stop_requested and return_code is not None,
                stdout=concise(stdout),
                stderr=concise(stderr),
                error=concise(error) if error is not None else None,
            )
            if self.verbose:
                if self.last_result.stdout:
                    print(self.last_result.stdout, file=sys.stderr)
                if self.last_result.stderr:
                    print(self.last_result.stderr, file=sys.stderr)
        return self.last_result


def check_hackrf() -> str:
    """Convenience probe for scripts; no RF transmission occurs."""
    return HackRFTransferBackend().check_available()
