"""Durable requested/completed event journal and host-side ground truth."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .models import TrafficEvent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


FIELDS = [
    "event_id", "status", "requested_timestamp", "host_start_timestamp",
    "host_end_timestamp", "host_elapsed_seconds", "scheduling_lateness_seconds",
    "channel", "center_frequency_hz", "requested_burst_duration_ms",
    "actual_waveform_duration_ms", "number_of_packets", "packet_durations_us",
    "actual_packet_durations_us", "inter_packet_gaps_us", "actual_gap_durations_us",
    "packet_sample_counts", "gap_sample_counts", "sample_count", "sample_rate",
    "modulation", "fft_size", "cyclic_prefix_length", "digital_amplitude",
    "tx_vga_gain_db", "rf_amp_enabled", "random_seed", "waveform_seed",
    "hackrf_return_code", "hackrf_success", "hackrf_stderr", "traffic_mode", "idle_before_seconds",
    "transmit_requested", "transmission_attempted", "error",
]


class ExperimentLogger:
    """Flush requests before work; CSV has terminal rows, JSONL has both states."""

    def __init__(self, config: Config):
        self.config = config
        root = Path(config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.directory = root / f"experiment_{stamp}"
        self.directory.mkdir()
        saved = config.to_dict()
        saved["program_version"] = __version__
        saved["timing_reference"] = "host timestamps and waveform samples; not RF timestamps"
        saved["created_at"] = utc_now()
        saved["rng"] = "numpy.default_rng; independent per-event waveform_seed"
        (self.directory / "config.json").write_text(
            json.dumps(saved, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._csv = (self.directory / "ground_truth.csv").open("w", newline="", encoding="utf-8")
        self._journal = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=FIELDS)
        self._writer.writeheader()
        self._flush(self._csv)
        self.started_at = utc_now()
        self.counts: Counter[str] = Counter()
        self.by_channel: Counter[str] = Counter()
        self.requested_airtime: dict[str, float] = defaultdict(float)
        self.generated_airtime: dict[str, float] = defaultdict(float)
        self.successful_airtime: dict[str, float] = defaultdict(float)
        self.generated_span: dict[str, float] = defaultdict(float)
        self.packet_count = 0
        self.packet_us = 0.0
        self.burst_ms = 0.0
        self.transmission_attempts = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0

    @staticmethod
    def _flush(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())

    def _journal_record(self, record: dict[str, Any]) -> None:
        self._journal.write(json.dumps(record, allow_nan=False) + "\n")
        self._flush(self._journal)

    def requested(self, event: TrafficEvent) -> dict[str, Any]:
        cfg = self.config
        record: dict[str, Any] = dict.fromkeys(FIELDS)
        record.update(
            event_id=event.event_id, status="requested",
            requested_timestamp=(datetime.fromtimestamp(event.requested_start_time, timezone.utc).isoformat()
                                 if event.requested_start_time is not None else utc_now()),
            channel=event.channel, center_frequency_hz=event.center_frequency_hz,
            requested_burst_duration_ms=event.requested_burst_duration_ms,
            number_of_packets=len(event.packet_durations_us),
            packet_durations_us=event.packet_durations_us,
            inter_packet_gaps_us=event.inter_packet_gaps_us,
            sample_rate=cfg.sample_rate, modulation=cfg.modulation,
            fft_size=cfg.fft_size, cyclic_prefix_length=cfg.cyclic_prefix,
            digital_amplitude=cfg.digital_amplitude, tx_vga_gain_db=cfg.tx_gain_db,
            rf_amp_enabled=False, random_seed=cfg.seed, waveform_seed=event.waveform_seed,
            traffic_mode=cfg.traffic_mode, idle_before_seconds=event.idle_before_seconds,
            transmit_requested=cfg.transmit, transmission_attempted=False,
        )
        self._journal_record(record)
        self.counts["requested"] += 1
        key = str(event.channel)
        self.by_channel[key] += 1
        self.requested_airtime[key] += sum(event.packet_durations_us) / 1e6
        return record

    def completed(self, record: dict[str, Any]) -> None:
        self._journal_record(record)
        self._writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict, bool)) else v
                               for k, v in record.items()})
        self._flush(self._csv)
        self.counts[record["status"]] += 1
        if record["transmission_attempted"]:
            self.transmission_attempts += 1
            if record["hackrf_success"] is True:
                self.successful_transmissions += 1
            else:
                self.failed_transmissions += 1
        if record["sample_count"] is not None:
            self.counts["generated"] += 1
            key = str(record["channel"])
            packet_us = sum(record["actual_packet_durations_us"])
            self.packet_count += record["number_of_packets"]
            self.packet_us += packet_us
            self.burst_ms += record["actual_waveform_duration_ms"]
            self.generated_airtime[key] += packet_us / 1e6
            self.generated_span[key] += record["actual_waveform_duration_ms"] / 1000
            if record["transmission_attempted"] and record["hackrf_success"] is True:
                self.successful_airtime[key] += packet_us / 1e6

    def finish(self, elapsed: float, status: str, error: str | None = None) -> dict[str, Any]:
        generated = self.counts["generated"]
        channels = [str(c) for c in self.config.channels]
        def complete(values: dict[str, float] | Counter[str]) -> dict[str, float]:
            return {ch: values.get(ch, 0) for ch in channels}
        summary = {
            "status": status, "error": error, "started_at": self.started_at,
            "ended_at": utc_now(), "experiment_duration_seconds": elapsed,
            "number_of_bursts": self.counts["requested"], "generated_bursts": generated,
            "attempted_transmissions": self.transmission_attempts,
            "successful_hackrf_transmissions": self.successful_transmissions,
            "failed_transmissions": self.failed_transmissions,
            "dry_run_bursts": self.counts["dry_run"], "event_status_counts": dict(self.counts),
            "bursts_per_channel": complete(self.by_channel),
            "requested_packet_airtime_seconds_per_channel": complete(self.requested_airtime),
            "generated_packet_airtime_seconds_per_channel": complete(self.generated_airtime),
            "generated_waveform_duration_seconds_per_channel": complete(self.generated_span),
            "successful_packet_airtime_seconds_per_channel": complete(self.successful_airtime),
            "approximate_channel_duty_cycle": {
                ch: self.successful_airtime.get(ch, 0) / elapsed if elapsed > 0 else 0 for ch in channels
            },
            "simulated_channel_duty_cycle": {
                ch: self.generated_airtime.get(ch, 0) / elapsed if elapsed > 0 else 0 for ch in channels
            } if not self.config.transmit else None,
            "total_packet_count": self.packet_count,
            "mean_packets_per_burst": self.packet_count / generated if generated else 0,
            "mean_packet_duration_us": self.packet_us / self.packet_count if self.packet_count else 0,
            "mean_burst_duration_ms": self.burst_ms / generated if generated else 0,
            "seed": self.config.seed, "transmit": self.config.transmit,
            "airtime_note": "Packet airtime excludes zero gaps and process overhead. Successful means process exit 0, not measured RF delivery. Duty is attributed to the selected center channel; overlapping channels are not inferred.",
            "transmission_count_note": "Attempts count backend calls; failed attempts include interrupted calls. Failures before a backend call are event failures, not failed transmissions.",
        }
        try:
            (self.directory / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        finally:
            self._csv.close()
            self._journal.close()
        return summary
