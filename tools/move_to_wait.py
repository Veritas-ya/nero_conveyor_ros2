#!/usr/bin/env python3
"""Repository entry; the default validates configuration without connecting CAN."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/nero_core'))
from nero_core.move_to_wait import main

if __name__ == '__main__':
    raise SystemExit(main())
