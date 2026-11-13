"""Explicit arm-enable operation; never opens/closes or zeros the gripper."""
import argparse
import json
from pathlib import Path
import signal
from threading import Event
import time


def enable_arm(config, stopped, *, factory=None, clock=time.monotonic, sleep=time.sleep):
    if stopped.is_set():
        raise RuntimeError('使能操作已取消，未连接设备')
    if factory is None:
        from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
        factory = Hardware
    hardware = config.raw['hardware']
    device = factory(hardware['can_channel'], hardware['firmware'], True,
                     hardware['joint_receive_offset_s'])
    try:
        device.start()  # acquires the same motion lease as the pick executor
        deadline = clock()+5.
        while not stopped.is_set() and clock() < deadline:
            enabled = device.robot.enable()
            if enabled:
                # Validate real fresh driver/pose/jaw status; a bool alone is
                # not a grab-ready claim. Does not send a motion or jaw target.
                state = device.snapshot(wait_timeout_s=min(1., max(0., deadline-clock())))
                return {'enabled': True, 'status': 'ready', 'joints_rad': state['joints'],
                        'width_m': state['width_m'], 'force_n': state['force_n'],
                        'verified_monotonic_s': clock(), 'motion_commands': 0,
                        'gripper_commands': 0}
            sleep(.05)
        if stopped.is_set():
            raise RuntimeError('使能操作已取消')
        raise RuntimeError('5 秒内未确认七轴使能，请检查设备供电与反馈')
    finally:
        device.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    stopped = Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stopped.set())
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.diagnostics import json_value
    try:
        cfg = CellConfig.load(args.config)
        result = enable_arm(cfg, stopped)
        print('[使能] 七轴已使能，真实反馈检查通过；未发送运动或夹爪开合指令', flush=True)
        code = 0
    except Exception as error:
        result = {'enabled': False, 'status': 'failed', 'error': str(error),
                  'note': '若已发出使能请求，退出不会自动失能；未发送运动或夹爪目标'}
        print('[使能检查未通过]', error, flush=True)
        code = 1
    (args.output/'panel_status.json').write_text(json.dumps(json_value(result), ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
