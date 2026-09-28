"""Generate one deterministic burst, optionally transmit it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wifi_emulator.cli import single_main

if __name__ == "__main__":
    raise SystemExit(single_main())
