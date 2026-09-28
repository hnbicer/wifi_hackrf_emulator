"""User-facing CLI semantics and executable script smoke checks."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from wifi_emulator.cli import random_main, single_main, inspect_main


def test_single_deterministic_dry_run(tmp_path):
    for output in (tmp_path / "a", tmp_path / "b"):
        assert single_main(["--duration-ms", "5", "--channel", "6", "--output-dir", str(output)]) == 0
    records = []
    for output in (tmp_path / "a", tmp_path / "b"):
        directory = next(output.iterdir())
        with (directory / "ground_truth.csv").open(newline="", encoding="utf-8") as stream:
            record = next(csv.DictReader(stream))
        records.append(record)
        assert record["status"] == "dry_run"
        assert record["center_frequency_hz"] == "2437000000"
        assert record["hackrf_success"] == ""
        assert record["host_start_timestamp"] == ""
        assert 5 <= float(record["actual_waveform_duration_ms"]) < 5.02
        assert {p.suffix for p in directory.iterdir()} <= {".json", ".csv", ".jsonl"}
    for key in ("waveform_seed", "sample_count", "packet_durations_us", "inter_packet_gaps_us"):
        assert records[0][key] == records[1][key]


def test_yaml_cannot_enable_rf_without_transmit_switch(tmp_path, monkeypatch):
    import wifi_emulator.runner as runner
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted hardware access")
    monkeypatch.setattr(runner, "HackRFTransferBackend", forbidden)
    config = tmp_path / "run.yaml"
    config.write_text("transmit: true\nseed: 9\nduration_seconds: 0.01\n", encoding="utf-8")
    assert random_main(["--config", str(config), "--output-dir", str(tmp_path / "results")]) == 0
    saved = json.loads(next((tmp_path / "results").glob("*/config.json")).read_text())
    # to_dict may use nested sections; durable summary unambiguously records run mode.
    summary = json.loads(next((tmp_path / "results").glob("*/summary.json")).read_text())
    assert summary["transmit"] is False
    assert saved["program_version"] == "0.1.0"


def test_saved_single_config_replays_with_cli_overrides_in_dry_run(tmp_path, monkeypatch):
    import wifi_emulator.runner as runner

    def forbidden(*args, **kwargs):
        pytest.fail("Replaying a saved configuration attempted hardware access")

    monkeypatch.setattr(runner, "HackRFTransferBackend", forbidden)
    source = tmp_path / "single"
    replay = tmp_path / "replay"
    assert single_main(["--channel", "6", "--duration-ms", "1", "--seed", "123",
                        "--output-dir", str(source)]) == 0
    snapshot = next(source.glob("*/config.json"))
    original = json.loads(snapshot.read_text(encoding="utf-8"))
    assert original["duration_seconds"] == 0
    assert "program_version" in original and "timing_reference" in original

    assert random_main(["--config", str(snapshot), "--duration", ".01",
                        "--idle-ms-min", "0", "--idle-ms-max", "0",
                        "--output-dir", str(replay)]) == 0
    directory = next(replay.iterdir())
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert config["duration_seconds"] == 0.01
    assert config["output_dir"] == str(replay)
    for key in ("seed", "channels", "packet_count_min", "packet_count_max",
                "packet_us_min", "packet_us_max"):
        assert config[key] == original[key]
    assert config["transmit"] is False
    assert summary["status"] == "completed"
    assert summary["transmit"] is False
    assert summary["attempted_transmissions"] == 0
    assert summary["dry_run_bursts"] > 0
    with (directory / "ground_truth.csv").open(newline="", encoding="utf-8") as stream:
        assert all(row["status"] == "dry_run" for row in csv.DictReader(stream))


@pytest.mark.parametrize("arguments", [["--channel", "14"], ["--duration-ms", "nan"],
                                      ["--duration-ms", "0"], ["--gain", "48"]])
def test_single_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        single_main(arguments)
    assert exc.value.code == 2


def test_inspection_has_no_implicit_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert inspect_main(["--seed", "123", "--packet-count", "3", "--no-show"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_inspection_rejects_transmission():
    with pytest.raises(SystemExit):
        inspect_main(["--transmit", "--no-show"])


@pytest.mark.parametrize("main", [random_main, single_main])
def test_output_path_error_is_clean(main, tmp_path, capsys):
    blocked = tmp_path / "file.txt"
    blocked.write_text("existing user file", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(blocked)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Cannot write experiment output" in error
    assert "Traceback" not in error
    assert blocked.read_text() == "existing user file"


@pytest.mark.parametrize("script", ["check_hackrf", "inspect_waveform", "transmit_single", "random_traffic"])
def test_scripts_help_from_other_directory(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / f"{script}.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
