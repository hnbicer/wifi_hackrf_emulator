"""Sequential execution independent of the traffic and transmitter implementations."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .burst import generate_burst
from .config import Config
from .hackrf import HackRFTransferBackend, TransmitterBackend, TransmissionResult
from .logger import ExperimentLogger
from .models import TrafficEvent
from .traffic import TrafficScheduler


@dataclass(frozen=True)
class ExperimentResult:
    directory: Path
    status: str
    summary: dict


@contextmanager
def graceful_interrupts() -> Iterator[None]:
    """Allow normal stack unwinding on SIGTERM and Windows Ctrl+Break too."""
    previous = {}
    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Signal {signum}")
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                previous[sig] = signal.signal(sig, interrupt)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_experiment(config: Config, *, events: Iterable[TrafficEvent] | None = None,
                   backend: TransmitterBackend | None = None) -> ExperimentResult:
    """Run until duration expires or events finish; persist partial runs on errors."""
    config.validate()
    if config.seed is None:
        config = replace(config, seed=int(np.random.SeedSequence().entropy))
    logger = ExperimentLogger(config)
    if config.transmit:
        print("========================================\nRF TRANSMISSION ENABLED\n========================================", flush=True)
    print(f"{'TRANSMIT' if config.transmit else 'DRY RUN'} | seed={config.seed} | {logger.directory}", flush=True)
    started = time.perf_counter()
    deadline = started + config.duration_seconds if config.duration_seconds else float("inf")
    status, error = "completed", None
    current: dict | None = None
    transmitter = None
    try:
        with graceful_interrupts():
            if config.transmit:
                transmitter = backend or HackRFTransferBackend(
                    timeout_seconds=config.transfer_timeout_seconds, verbose=config.verbose
                )
                check = getattr(transmitter, "check_available", None)
                if check is not None:
                    print(check(), flush=True)
            scheduler = TrafficScheduler(config)
            iterator = iter(events) if events is not None else None
            while time.perf_counter() < deadline:
                if iterator is not None:
                    try:
                        event = next(iterator)
                    except StopIteration:
                        break
                else:
                    event = scheduler.next_event()
                target = time.perf_counter() + event.idle_before_seconds
                if target >= deadline:
                    time.sleep(max(0, deadline - time.perf_counter()))
                    break
                event = replace(event, requested_start_time=time.time() + event.idle_before_seconds)
                current = logger.requested(event)
                time.sleep(max(0, target - time.perf_counter()))
                burst = generate_burst(event, ramp_us=config.ramp_us)
                current.update(
                    actual_packet_durations_us=burst.actual_packet_durations_us,
                    actual_gap_durations_us=burst.actual_gap_durations_us,
                    packet_sample_counts=burst.packet_sample_counts,
                    gap_sample_counts=burst.gap_sample_counts,
                    actual_waveform_duration_ms=burst.duration_ms, sample_count=int(burst.iq.size),
                    scheduling_lateness_seconds=max(0, time.perf_counter() - target),
                )
                if config.transmit:
                    assert transmitter is not None
                    # Clear the previous event before calling a backend that may
                    # fail or be interrupted before producing a fresh result.
                    transmitter.last_result = None
                    current.update(transmission_attempted=True, hackrf_success=False)
                    result = transmitter.transmit(
                        burst.iq, event.center_frequency_hz, sample_rate=config.sample_rate,
                        gain_db=config.tx_gain_db, digital_amplitude=config.digital_amplitude,
                    )
                    _record_result(current, result)
                    current["status"] = "success" if result.success else "failed"
                else:
                    # Model the embedded burst duration without touching a hardware backend.
                    time.sleep(burst.duration_ms / 1000)
                    current["status"] = "dry_run"
                logger.completed(current)
                completed = current
                current = None
                print(f"[{event.event_id:04d}] CH{event.channel:<2} {event.center_frequency_hz / 1e6:.0f} MHz "
                      f"packets={len(event.packet_durations_us)} burst={burst.duration_ms:.3f} ms "
                      f"{completed['status']}", flush=True)
                failed = completed["status"] == "failed"
                if failed:
                    status, error = "failed", completed["error"] or "hackrf_transfer failed"
                if failed:
                    break
    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        error = str(exc) or type(exc).__name__
        if current is not None:
            if current["transmission_attempted"]:
                result = getattr(transmitter, "last_result", None)
                if result is not None:
                    _record_result(current, result)
            current.update(status=status, error=error)
            logger.completed(current)
    finally:
        summary = logger.finish(time.perf_counter() - started, status, error)
    return ExperimentResult(logger.directory, status, summary)


def _record_result(record: dict, result: TransmissionResult) -> None:
    """Copy compact metadata and bounded, backend-sanitized error diagnostics."""
    record.update(
        host_start_timestamp=result.host_start_timestamp,
        host_end_timestamp=result.host_end_timestamp,
        host_elapsed_seconds=result.host_elapsed_seconds,
        hackrf_return_code=result.return_code,
        hackrf_success=result.success,
        hackrf_stderr=result.stderr[-2048:] if not result.success else None,
        error=result.error,
    )
