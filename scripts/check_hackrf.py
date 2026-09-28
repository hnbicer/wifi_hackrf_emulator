"""Read HackRF device information without transmitting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wifi_emulator.cli import check_main

if __name__ == "__main__":
    raise SystemExit(check_main())
