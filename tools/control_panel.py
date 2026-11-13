#!/usr/bin/env python3
"""Start the workspace control panel without rebuilding the hardware installation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/nero_core'))
from nero_core.control_panel.server import main

if __name__ == '__main__':
    raise SystemExit(main())
