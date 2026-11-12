#!/usr/bin/env python3
"""Offline A/B/C planning replay; does not start camera or robot processes."""
from nero_core.two_second.paper_replay import main

if __name__ == '__main__':
    raise SystemExit(main())
