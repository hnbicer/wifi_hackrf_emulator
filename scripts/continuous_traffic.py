"""Menu for continuous fixed-channel Wi-Fi-like packet trains."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wifi_emulator.continuous import continuous_main

if __name__ == "__main__":
    raise SystemExit(continuous_main())
