"""Run from the project root, with or without an editable installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wifi_emulator.cli import random_main

if __name__ == "__main__":
    raise SystemExit(random_main())
