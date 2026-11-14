#!/usr/bin/env python3
"""Run the independent phase allocation experiment directly from this checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/nero_core'))
from nero_core.two_second.phase_timing_test import main

if __name__ == '__main__':
    raise SystemExit(main())
