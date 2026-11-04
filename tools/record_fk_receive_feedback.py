#!/usr/bin/env python3
"""Record robot feedback passively; this script never sends motion or enable commands."""
import argparse
from pathlib import Path
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
from nero_core.two_second.config import CellConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--hz', type=float, default=50.)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.seconds, args.hz]).all() or args.seconds <= 0 or not 0 < args.hz <= 100:
        parser.error('seconds must be positive and hz must be in (0, 100]')
    if args.output.suffix != '.npz':
        parser.error('output must end in .npz')
    if args.output.exists():
        parser.error('output already exists')
    config = CellConfig.load(args.config)
    hw = Hardware(config.raw['hardware']['can_channel'], config.raw['hardware']['firmware'], False,
                  config.raw['hardware']['joint_receive_offset_s'])
    trace = ExecutionTrace()
    begun = time.monotonic()
    print('只读采集：不发送机械臂控制或使能指令', flush=True)
    try:
        hw.start()
        begun = time.monotonic()
        slot = 0
        while time.monotonic()-begun < args.seconds:
            deadline = begun+slot/args.hz
            delay = deadline-time.monotonic()
            if delay > 0: time.sleep(delay)
            start = time.monotonic()
            state = hw.snapshot()
            trace.record_feedback(state, start, time.monotonic(), stage='passive_receive_timing')
            slot = max(slot+1, int((time.monotonic()-begun)*args.hz)+1)
    except KeyboardInterrupt:
        print('采集已停止，保存已有数据', flush=True)
    finally:
        hw.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **trace.arrays())
        print(f'已保存 {len(trace.feedback_rows)} 条反馈至 {args.output}', flush=True)


if __name__ == '__main__':
    main()
