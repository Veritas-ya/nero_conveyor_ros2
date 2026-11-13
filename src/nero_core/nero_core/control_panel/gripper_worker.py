"""Explicit manual opening with real feedback; sends no arm motion targets."""
import argparse
import json
from pathlib import Path
import signal
from threading import Event
import time

import numpy as np


class GripperOpenError(RuntimeError):
    def __init__(self, message, *, command_attempted=False):
        super().__init__(message)
        self.command_attempted = command_attempted


def open_gripper(config, stopped, *, factory=None, clock=time.monotonic, sleep=time.sleep):
    if stopped.is_set():
        raise GripperOpenError('张爪操作已取消，未连接设备')
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    from nero_core.two_second.feedback_gates import OpenConfirmation
    from nero_core.two_second.streaming import fresh_feedback
    from nero_core.two_second.wait_precision import JointStillness, STILL_RANGE_RAD
    if factory is None:
        from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
        factory = Hardware
    hardware = config.raw['hardware']
    width = float(config.settings.open_width_m)
    if not np.isfinite(width) or not 0 < width <= .1:
        raise GripperOpenError('配置张爪宽度无效')
    device = factory(hardware['can_channel'], hardware['firmware'], True,
                     hardware['joint_receive_offset_s'])
    command_attempted = False
    began = clock()
    deadline = began+min(5., config.settings.move_timeout_s)
    previous = {}

    def measured():
        if stopped.is_set():
            raise RuntimeError('张爪操作已取消，请以实际夹爪反馈为准')
        if clock() >= deadline:
            raise RuntimeError('张爪检查超时，未确认机械臂停稳或夹爪到达目标开口')
        state = device.snapshot(wait_timeout_s=min(.2, max(0., deadline-clock())))
        if stopped.is_set() or clock() >= deadline:
            raise RuntimeError('张爪操作已取消或反馈等待超时')
        fresh_feedback(state, clock(), previous)
        q = np.asarray(state['joints'], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('张爪前关节反馈无效')
        if not np.isfinite([state['width_m'], state['force_n']]).all():
            raise RuntimeError('夹爪开口或受力反馈无效')
        return state

    try:
        device.start()  # Acquires the original cross-process motion lease.
        stillness = JointStillness()
        while True:
            state = measured()
            if stillness.observe(state['joints'], mapped_stamp(state, 'joint')):
                break
            sleep(.01)
        if stopped.is_set() or clock() >= deadline:
            raise RuntimeError('张爪操作已取消或超时，未发送张爪目标')
        anchor = np.asarray(state['joints'], dtype=float).copy()
        command_attempted = True
        device.grip(width, 1.)  # Same opening width and 1 N request as release_at_rest.
        opening = OpenConfirmation(width, not_before=clock())
        while True:
            state = measured()
            if np.max(abs(np.asarray(state['joints'])-anchor)) > STILL_RANGE_RAD:
                raise RuntimeError('张爪期间关节位置改变，未继续操作')
            if opening.observe(state['width_m'], mapped_stamp(state, 'gripper')):
                return {'status': 'gripper_opened', 'opened': True,
                    'requested_width_m': width, 'width_m': float(state['width_m']),
                    'force_n': float(state['force_n']), 'gripper_commands': 1,
                    'motion_commands': 0, 'verified_monotonic_s': clock(),
                    'elapsed_s': clock()-began}
            sleep(.01)
    except Exception as error:
        raise GripperOpenError(str(error), command_attempted=command_attempted) from error
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
    try:
        result = open_gripper(CellConfig.load(args.config), stopped)
        print(f'[张爪] 已确认打开至 {result["width_m"]*1000:.1f} mm；未发送机械臂运动目标', flush=True)
        code = 0
    except Exception as error:
        attempted = bool(getattr(error, 'command_attempted', False))
        result = {'status': 'failed', 'opened': False, 'error': str(error),
            'gripper_command_attempted': attempted, 'motion_commands': 0,
            'note': '已尝试发送张爪目标，请以真实反馈为准' if attempted else '未尝试发送张爪目标'}
        print('[张爪未确认]', error, flush=True)
        code = 1
    (args.output/'panel_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
