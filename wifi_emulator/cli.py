"""Command-line entry points. Only an explicit --transmit enables RF output."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .channels import channel_to_frequency, parse_channels
from .config import Config, load_config

if TYPE_CHECKING:
    from .runner import ExperimentResult


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="YAML or JSON configuration; CLI values override it")
    parser.add_argument("--channels", type=parse_channels, default=None)
    parser.add_argument("--duration", dest="duration_seconds", type=float, default=None,
                        help="Experiment seconds; 0 runs until Ctrl+C")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gain", dest="tx_gain_db", type=int, default=None)
    parser.add_argument("--digital-amplitude", type=float, default=None)
    parser.add_argument("--traffic-mode", choices=["random", "poisson"], default=None)
    parser.add_argument("--lambda-events", dest="lambda_events_per_second", type=float, default=None)
    parser.add_argument("--channel-mode", choices=["uniform", "weighted"], default=None)
    parser.add_argument("--channel-weights", help="CH:weight pairs, e.g. 1:0.3,6:0.4,11:0.3")
    for flag in ("idle-ms-min", "idle-ms-max", "packet-us-min", "packet-us-max",
                 "gap-us-min", "gap-us-max", "burst-ms-min", "burst-ms-max", "ramp-us"):
        parser.add_argument(f"--{flag}", type=float, default=None)
    for flag in ("packet-count-min", "packet-count-max"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    parser.add_argument("--transfer-timeout", dest="transfer_timeout_seconds", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--transmit", action="store_true", help="Enable actual RF transmission (default: dry run)")
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    changes = {name: getattr(args, name) for name in cfg.__dataclass_fields__
               if hasattr(args, name) and getattr(args, name) is not None}
    # A saved transmitting configuration must not silently authorize a later RF run.
    changes["transmit"] = args.transmit
    if args.channel_weights is not None:
        try:
            pairs = [entry.split(":") for entry in args.channel_weights.split(",")]
            weights = {int(ch): float(value) for ch, value in pairs}
            if len(weights) != len(pairs):
                raise ValueError("duplicate channel")
        except (ValueError, TypeError) as exc:
            raise ValueError("--channel-weights must contain unique CH:weight pairs") from exc
        changes["channel_weights"] = weights
        if args.channel_mode is None:
            changes["channel_mode"] = "weighted"
    return replace(cfg, **changes)


def _report(result: "ExperimentResult") -> int:
    print(f"Experiment {result.status}. Logs: {result.directory}")
    if result.summary.get("error"):
        print(f"Reason: {result.summary['error']}")
    return 130 if result.status == "interrupted" else (1 if result.status == "failed" else 0)


def random_main(argv: list[str] | None = None) -> int:
    from .runner import run_experiment
    parser = _parser("Generate sequential randomized packetized OFDM traffic; dry run by default.")
    args = parser.parse_args(argv)
    try:
        cfg = _config(args)
        cfg.validate()
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    try:
        return _report(run_experiment(cfg))
    except OSError as exc:
        parser.error(f"Cannot write experiment output: {exc}")


def single_main(argv: list[str] | None = None) -> int:
    from .models import TrafficEvent
    from .runner import run_experiment
    parser = _parser("Generate one deterministic packetized burst; dry run by default.")
    parser.add_argument("--channel", type=int, default=6)
    parser.add_argument("--duration-ms", type=float, default=5.0, help="Requested whole burst length")
    args = parser.parse_args(argv)
    try:
        cfg = _config(args)
        frequency = channel_to_frequency(args.channel)
        if not math.isfinite(args.duration_ms) or not 0.02 <= args.duration_ms <= 500:
            raise ValueError("--duration-ms must be between 0.02 and 500")
        count = max(1, math.ceil(args.duration_ms / 1.5))
        gap_us = 100.0
        packet_us = (args.duration_ms * 1000 - (count - 1) * gap_us) / count
        cfg = replace(cfg, channels=[args.channel], channel_mode="uniform", channel_weights={},
                      seed=cfg.seed if cfg.seed is not None else 0, duration_seconds=0,
                      packet_count_min=count, packet_count_max=count,
                      packet_us_min=packet_us, packet_us_max=packet_us,
                      gap_us_min=gap_us, gap_us_max=gap_us,
                      burst_ms_min=0.02, burst_ms_max=max(10.0, args.duration_ms + count * 0.004))
        cfg.validate()
        waveform_seed = int(np.random.default_rng(cfg.seed).integers(0, 2**63))
        event = TrafficEvent(1, args.channel, frequency, [packet_us] * count,
                             [gap_us] * (count - 1), waveform_seed=waveform_seed)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    try:
        return _report(run_experiment(cfg, events=[event]))
    except OSError as exc:
        parser.error(f"Cannot write experiment output: {exc}")


def inspect_main(argv: list[str] | None = None) -> int:
    from .analysis import plot_waveform
    from .burst import generate_burst
    from .quantization import normalize_iq
    from .traffic import TrafficScheduler
    parser = _parser("Inspect an in-memory OFDM burst. No RF output or IQ files.")
    parser.add_argument("--packet-count", type=int, default=None)
    parser.add_argument("--save-plot", type=Path, help="Explicitly save a diagnostic image, never IQ")
    parser.add_argument("--no-show", action="store_true", help="Render without opening a plot window")
    args = parser.parse_args(argv)
    if args.transmit:
        parser.error("waveform inspection cannot transmit")
    try:
        cfg = _config(args)
        if args.packet_count is not None:
            cfg = replace(cfg, packet_count_min=args.packet_count, packet_count_max=args.packet_count)
        cfg = replace(cfg, seed=cfg.seed if cfg.seed is not None else 123)
        cfg.validate()
        event = TrafficScheduler(cfg).next_event()
        burst = generate_burst(event, ramp_us=cfg.ramp_us)
        iq = normalize_iq(burst.iq, cfg.digital_amplitude)
        print(f"Packets: {len(burst.packet_sample_counts)}")
        print(f"Requested packet durations (us): {event.packet_durations_us}")
        print(f"Actual packet durations (us): {burst.actual_packet_durations_us}")
        print(f"Inter-packet gaps (us): {burst.actual_gap_durations_us}")
        print(f"Burst duration: {burst.duration_ms:.4f} ms | Samples: {burst.iq.size}")
        print(f"Peak amplitude: {np.max(np.abs(iq)):.6f} | RMS: {np.sqrt(np.mean(np.abs(iq)**2)):.6f}")
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot_waveform(burst, digital_amplitude=cfg.digital_amplitude)
        if args.save_plot:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_plot, dpi=150)
            print(f"Diagnostic image: {args.save_plot}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


def check_main(argv: list[str] | None = None) -> int:
    from .hackrf import check_hackrf
    parser = argparse.ArgumentParser(description="Check HackRF utilities and device; does not transmit.")
    parser.parse_args(argv)
    try:
        print(check_hackrf())
    except (RuntimeError, OSError) as exc:
        print(f"HackRF check failed: {exc}")
        return 1
    return 0
