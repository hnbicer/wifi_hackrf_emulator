"""Interactive fixed-channel packet trains with one repeating host invocation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from . import __version__
from .burst import generate_burst
from .channels import channel_to_frequency
from .hackrf import HackRFTransferBackend
from .models import TrafficEvent
from .ofdm import SAMPLE_RATE
from .packet import packet_sample_count
from .runner import ExperimentResult, graceful_interrupts


@dataclass(frozen=True)
class ContinuousSettings:
    channel: int = 6
    gain_db: int = 0
    digital_amplitude: float = 0.6
    packet_us: float = 1000.0
    gap_us: float = 100.0
    packet_count: int = 50
    duration_seconds: float = 0.0
    seed: int = 0
    output_dir: str = "results"
    rf_amp_enabled: bool = False

    def validate(self) -> None:
        channel_to_frequency(self.channel)
        if not isinstance(self.rf_amp_enabled, bool):
            raise ValueError("rf_amp_enabled must be true or false")
        for name, low, high in (("gain_db", 0, 47), ("packet_count", 1, 10_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("digital_amplitude", "packet_us", "gap_us", "duration_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.digital_amplitude <= 1:
            raise ValueError("digital_amplitude must be greater than 0 and at most 1")
        if not 20 <= self.packet_us <= 1_000_000:
            raise ValueError("packet_us must be between 20 and 1000000")
        if not 0 <= self.gap_us <= 1_000_000:
            raise ValueError("gap_us must be between 0 and 1000000")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be nonnegative; 0 runs until Ctrl+C")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")
        if self.loop_sample_count > SAMPLE_RATE:
            raise ValueError("The packet train including its final gap must be at most 1000 ms; reduce packet length, gap, or count")

    @property
    def gap_samples(self) -> int:
        return round(self.gap_us * (SAMPLE_RATE / 1e6))

    @property
    def loop_sample_count(self) -> int:
        return self.packet_count * (packet_sample_count(self.packet_us) + self.gap_samples)

    def loop_description(self) -> dict:
        packet_samples = packet_sample_count(self.packet_us)
        return {
            "sample_rate": SAMPLE_RATE,
            "sample_count": self.loop_sample_count,
            "packets_per_loop": self.packet_count,
            "actual_packet_us": packet_samples / SAMPLE_RATE * 1e6,
            "actual_gap_us": self.gap_samples / SAMPLE_RATE * 1e6,
            "loop_duration_ms": self.loop_sample_count / SAMPLE_RATE * 1000,
            "nominal_packets_per_second": SAMPLE_RATE / (packet_samples + self.gap_samples),
            "nominal_packet_duty_cycle": packet_samples / (packet_samples + self.gap_samples),
        }


def generate_loop(settings: ContinuousSettings) -> np.ndarray:
    """Generate a bounded train and a last-to-first gap, without saving IQ."""
    settings.validate()
    event = TrafficEvent(
        1, settings.channel, channel_to_frequency(settings.channel),
        [settings.packet_us] * settings.packet_count,
        [settings.gap_us] * (settings.packet_count - 1), waveform_seed=settings.seed,
    )
    burst = generate_burst(event)
    # generate_burst has only internal gaps. The trailing gap completes the
    # final packet's spacing when hackrf_transfer rewinds to the first packet.
    return np.concatenate((burst.iq, np.zeros(settings.gap_samples, dtype=np.complex64)))


def run_continuous(settings: ContinuousSettings, *, transmit: bool = False,
                   verbose: bool = False, backend: HackRFTransferBackend | None = None) -> ExperimentResult:
    """Keep the train running until duration/interrupt; preserve session metadata."""
    settings.validate()
    directory = Path(settings.output_dir) / datetime.now().strftime("continuous_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    snapshot = {**asdict(settings), "transmit": transmit, "mode": "continuous",
                "program_version": __version__}
    (directory / "config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "running", "transmit": transmit, "channel": settings.channel,
        "center_frequency_hz": channel_to_frequency(settings.channel),
        "rf_amp_enabled": settings.rf_amp_enabled,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None, "elapsed_seconds": None, "error": None,
        "loop": settings.loop_description(), "waveform_generated": False,
        "transmission_attempted": False, "host_result": None,
        "timing_reference": "Host timing and nominal loop values; RF delivery and packet totals are not measured.",
    }
    summary_path = directory / "continuous_summary.json"

    def save_summary() -> None:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save_summary()
    started = time.perf_counter()
    transmitter = None
    print(f"{'RF TRANSMISSION ENABLED' if transmit else 'DRY RUN (no RF output)'} | Logs: {directory}", flush=True)
    try:
        with graceful_interrupts():
            if transmit:
                transmitter = backend or HackRFTransferBackend(verbose=verbose)
                transmitter.last_result = None
                print(transmitter.check_available(), flush=True)
            iq = generate_loop(settings)
            summary["waveform_generated"] = True
            if transmit:
                summary["transmission_attempted"] = True
                save_summary()
                print("Starting continuous packet train. Ctrl+C stops and returns to the menu.", flush=True)
                result = transmitter.transmit_repeating(
                    iq, channel_to_frequency(settings.channel), gain_db=settings.gain_db,
                    digital_amplitude=settings.digital_amplitude, duration_seconds=settings.duration_seconds,
                    rf_amp_enabled=settings.rf_amp_enabled,
                )
                summary["host_result"] = asdict(result)
                summary["status"] = "completed" if result.success else "failed"
                summary["error"] = result.error
            else:
                save_summary()
                print("Simulating continuous timing. Ctrl+C stops and returns to the menu.", flush=True)
                active_start = time.perf_counter()
                while True:
                    if settings.duration_seconds:
                        remaining = settings.duration_seconds - (time.perf_counter() - active_start)
                        if remaining <= 0:
                            break
                        time.sleep(min(0.2, remaining))
                    else:
                        time.sleep(0.2)
                summary["status"] = "completed"
    except (KeyboardInterrupt, SystemExit):
        summary["status"] = "interrupted"
        if transmitter is not None and summary["transmission_attempted"] and transmitter.last_result is not None:
            summary["host_result"] = asdict(transmitter.last_result)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if transmitter is not None and summary["transmission_attempted"] and transmitter.last_result is not None:
            summary["host_result"] = asdict(transmitter.last_result)
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        save_summary()
    return ExperimentResult(directory, summary["status"], summary)


def _show_settings(settings: ContinuousSettings, transmit: bool) -> None:
    loop = settings.loop_description()
    duration = f"{settings.duration_seconds:g} seconds" if settings.duration_seconds else "until Ctrl+C"
    print(f"\nContinuous Wi-Fi-like OFDM | {'RF lab mode' if transmit else 'dry run'}")
    print(f"Channel {settings.channel} ({channel_to_frequency(settings.channel) / 1e6:g} MHz), "
          f"TX gain {settings.gain_db} dB, digital amplitude {settings.digital_amplitude:g}, "
          f"RF AMP {'ON' if settings.rf_amp_enabled else 'OFF'}")
    print(f"Packets {settings.packet_us:g} us (actual {loop['actual_packet_us']:g}), "
          f"gaps {settings.gap_us:g} us (actual {loop['actual_gap_us']:g}), {settings.packet_count} packets/loop")
    print(f"Loop {loop['loop_duration_ms']:g} ms, nominal packet duty {loop['nominal_packet_duty_cycle']:.1%}, "
          f"run {duration}, seed {settings.seed}")
    print("1 Start\n2 Channel\n3 TX gain\n4 Digital amplitude\n5 Packet timing\n6 Run duration\n7 Seed\n8 RF amplifier\n0 Exit")


def _amp_setting(raw: str) -> bool:
    if raw not in {"0", "1"}:
        raise ValueError("RF amplifier must be 0 (off) or 1 (on)")
    return raw == "1"


def _edit_settings(settings: ContinuousSettings, choice: str) -> ContinuousSettings:
    fields = {
        "2": [("channel", "Channel (1-13)", int)],
        "3": [("gain_db", "TX gain in dB (0-47)", int)],
        "4": [("digital_amplitude", "Digital amplitude (>0 to 1)", float)],
        "5": [("packet_us", "Packet duration in us (20 or more)", float),
              ("gap_us", "Gap in us, including loop boundary (0 or more)", float),
              ("packet_count", "Packets per loop (1-10000, whole loop at most 1000 ms)", int)],
        "6": [("duration_seconds", "Run seconds (0 = until Ctrl+C)", float)],
        "7": [("seed", "Waveform seed (nonnegative integer)", int)],
        "8": [("rf_amp_enabled", "RF amplifier (0=off, 1=on)", _amp_setting)],
    }
    changes = {}
    for name, label, convert in fields[choice]:
        current = getattr(settings, name)
        displayed = int(current) if isinstance(current, bool) else current
        raw = input(f"{label} [{displayed}]: ").strip()
        if raw:
            changes[name] = convert(raw)
    candidate = replace(settings, **changes)
    candidate.validate()
    return candidate


def continuous_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Menu for continuous fixed-channel Wi-Fi-like packet trains; dry run by default.")
    parser.add_argument("--channel", type=int, default=6)
    parser.add_argument("--gain", dest="gain_db", type=int, default=0)
    parser.add_argument("--digital-amplitude", type=float, default=0.6)
    parser.add_argument("--rf-amp", dest="rf_amp_enabled", action="store_true",
                        help="Enable the RF amplifier during transmission (default: off)")
    parser.add_argument("--packet-us", type=float, default=1000.0)
    parser.add_argument("--gap-us", type=float, default=100.0)
    parser.add_argument("--packet-count", type=int, default=50)
    parser.add_argument("--duration", dest="duration_seconds", type=float, default=0,
                        help="Run seconds; 0 runs until Ctrl+C (default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--transmit", action="store_true", help="Enable actual RF for a shielded or conducted lab setup")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = ContinuousSettings(**{name: getattr(args, name) for name in ContinuousSettings.__dataclass_fields__})
        settings.validate()
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print("20 MHz sample rate and filter; simplified Wi-Fi-like energy, not IEEE 802.11 frames.")
    if args.transmit:
        print("RF lab mode: use a shielded or conducted setup with suitable attenuation. Start with gain 0.")
    else:
        print("Hardware output requires launching with --transmit.")
    while True:
        try:
            _show_settings(settings, args.transmit)
            choice = input("Select: ").strip()
            if choice == "0":
                return 0
            if choice == "1":
                try:
                    result = run_continuous(settings, transmit=args.transmit, verbose=args.verbose)
                except OSError as exc:
                    print(f"Cannot write experiment output: {exc}")
                    continue
                print(f"Session {result.status}. Logs: {result.directory}")
                if result.summary["error"]:
                    print(f"Reason: {result.summary['error']}")
            elif choice in {"2", "3", "4", "5", "6", "7", "8"}:
                try:
                    settings = _edit_settings(settings, choice)
                except (ValueError, TypeError) as exc:
                    print(f"Invalid settings: {exc}. Previous settings retained.")
                except KeyboardInterrupt:
                    print("\nEdit cancelled. Previous settings retained.")
            else:
                print("Choose a menu item from 0 to 8.")
        except (EOFError, KeyboardInterrupt):
            print("\nMenu closed.")
            return 0
