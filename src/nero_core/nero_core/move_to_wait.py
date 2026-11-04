"""Return to configured wait joints with manufacturer move_j; check-only by default."""
import argparse
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import signal
from threading import Event
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from .two_second.diagnostics import RunJournal, json_value
from .two_second.console import ConsoleProgress, short

RETURN_WAIT_TIMEOUT_S = 15.


def validate_wait_joints(joints, model=None):
    """Read only joint bounds; this is not scene/path certification."""
    if np.shape(joints) != (7,) or not np.isfinite(joints).all():
        raise ValueError('standby_joint_rad 须为7个有限弧度值')
    if model is None:
        from mujoco_conveyor_sorting.runtime.single_pick import RobotModel
        model = RobotModel()
    if np.any(joints < model.command_lower) or np.any(joints > model.command_upper):
        raise ValueError('等待关节超出当前机器人模型的规划关节范围')


def default_config():
    name = 'two_second.hardware.legacy_observed.yaml'
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('nero_bringup'))/'config'/name
    except (ImportError, LookupError, OSError):
        for parent in Path(__file__).resolve().parents:
            candidate = parent/'src/nero_bringup/config'/name
            if candidate.is_file():
                return candidate
    raise ValueError('未找到现场配置，请显式提供 --config')


@dataclass(frozen=True)
class WaitConfig:
    source: Path
    source_hash: str
    joints: np.ndarray
    channel: str
    firmware: str
    speed_percent: int
    force_threshold: float

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        raw = yaml.safe_load(content)
        if not isinstance(raw, dict) or raw.get('joint_names') != [f'joint{i}' for i in range(1, 8)]:
            raise ValueError('配置须按 joint1..joint7 声明关节顺序')
        joints = np.asarray(raw.get('standby_joint_rad'), float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError('standby_joint_rad 须为7个有限弧度值')
        hardware = raw.get('hardware', {})
        if not isinstance(hardware, dict):
            raise ValueError('hardware 须为配置映射')
        channel, firmware = hardware.get('can_channel'), hardware.get('firmware')
        if not isinstance(channel, str) or not channel or '/' in channel or '\x00' in channel:
            raise ValueError('CAN通道名称无效')
        if not isinstance(firmware, str) or not firmware:
            raise ValueError('须明确厂家固件版本')
        settings = raw.get('pick_settings', {})
        if not isinstance(settings, dict):
            raise ValueError('pick_settings 须为配置映射')
        speed = settings.get('speed_percent', 100)
        if type(speed) is not int or not 1 <= speed <= 100:
            raise ValueError('配置速度须为1..100的整数百分比')
        force = float(settings.get('min_force_n', .2))
        if not np.isfinite(force) or force <= 0:
            raise ValueError('夹爪受力门槛须为正有限数')
        # Read only model joint bounds. Camera, TCP and bin calibration are
        # irrelevant to this joint-only command; no scene certificate is made.
        validate_wait_joints(joints)
        return cls(path, sha256(content).hexdigest(), joints.copy(), channel, firmware, speed, force)


class MoveCancelled(RuntimeError):
    pass


def make_report(config, speed_percent=None):
    return {'status': 'configuration_valid', 'config': str(config.source),
        'config_sha256': config.source_hash, 'target_joints_rad': config.joints.tolist(),
        'target_joints_deg': np.rad2deg(config.joints).tolist(), 'can_channel': config.channel,
        'firmware': config.firmware,
        'speed_percent': config.speed_percent if speed_percent is None else speed_percent,
        'joint_command_api': 'move_j', 'trajectory_source': 'manufacturer_internal_smoothing',
        'scene_collision_certified': False, 'automatic_enable': False, 'gripper_commands': 0,
        'joint_command_attempts': 0, 'joint_target_commands': 0, 'speed_commands': 0,
        'hardware_connections': 0, 'firmware_limits_modified': False}


def execute_return_wait(device, config, report, emit, cancelled, *,
                        timeout_s=RETURN_WAIT_TIMEOUT_S, clock=time.monotonic, sleep=time.sleep):
    """Shared by the standalone CLI and pick startup, on an already owned CAN."""
    try:
        device.configure_joint_command_mode('j')
        run_move(device, config, report, emit, cancelled,
                 timeout_s=timeout_s, clock=clock, sleep=sleep)
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, (MoveCancelled, KeyboardInterrupt)) else 'stopped',
                      error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', {}))
        if report['joint_command_attempts']:
            try:
                device.stop()
                report['electronic_stop_sent'] = True
            except BaseException as stop_error:
                report['stop_error'] = str(stop_error)
        raise


def run_move(device, config, report, emit, cancelled, *, timeout_s, clock, sleep):
    """One joint target, original measured-time arrival gate; no effector writes."""
    from .two_second.feedback_gates import RestArrival
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    previous_stamp = previous_epoch = None

    def read(phase):
        nonlocal previous_stamp, previous_epoch
        if cancelled.is_set():
            raise MoveCancelled('用户取消回等待位')
        state = device.snapshot()
        stamp = mapped_stamp(state, 'joint')
        now = clock()
        q = np.asarray(state['joints'], float)
        if (q.shape != (7,) or not np.isfinite(q).all() or stamp is None or
                not np.isfinite(stamp) or not 0 <= now-stamp <= .2):
            raise RuntimeError('关节反馈无效或超过原0.2秒新鲜度要求')
        evidence = state.get('feedback_time', {})
        epoch = tuple((evidence.get(key) or {}).get('clock_epoch')
                      for key in ('joint_12', 'joint_34', 'joint_56', 'joint_7'))
        if (previous_stamp is not None and stamp < previous_stamp or
                previous_epoch is not None and epoch != previous_epoch):
            raise RuntimeError('回等待位期间反馈时钟重置或倒退')
        previous_stamp, previous_epoch = stamp, epoch
        emit('feedback', phase=phase, state=state)
        report['last_state'] = state
        force = float(state['force_n'])
        if not np.isfinite(force) or force >= config.force_threshold:
            raise RuntimeError('夹爪存在受力，请先检查工件或限位；未自动操作夹爪')
        return state, q, stamp

    initial, anchor, stamp = read('preflight')
    report['initial_state'] = initial
    ready = RestArrival(clock())
    until = clock()+3.  # original startup feedback allowance
    while not ready.observe(anchor, np.asarray(initial['joints']), stamp):
        if clock() >= until:
            raise RuntimeError('当前关节未稳定，未发送回位命令')
        sleep(.02)
        state, anchor, stamp = read('preflight')
    if np.max(abs(anchor-config.joints)) <= .005:
        report.update(status='already_at_wait', max_abs_joint_error_rad=float(np.max(abs(anchor-config.joints))))
        return
    if cancelled.is_set():
        raise MoveCancelled('发送前已取消')
    # Force J even if the grasp controller config selects JS. Manufacturer J
    # plans/smooths a single destination; JS must never receive a position jump.
    device.robot.set_speed_percent(report['speed_percent'])
    report['speed_commands'] = 1
    if cancelled.is_set():
        raise MoveCancelled('发送关节目标前已取消')
    report['joint_command_attempts'] = 1
    report['actual_send_monotonic_s'] = clock()
    device.move_joints(config.joints.tolist())
    report['joint_target_commands'] = 1
    emit('target_sent', speed_percent=report['speed_percent'], target_joints_deg=np.rad2deg(config.joints))
    arrival = RestArrival(report['actual_send_monotonic_s'])
    until = clock()+timeout_s
    while True:
        state, q, stamp = read('moving')
        error = float(np.max(abs(q-config.joints)))
        report['max_abs_joint_error_rad'] = error
        if arrival.observe(q, config.joints, stamp):
            report.update(status='wait_reached', move_wall_s=clock()-report['actual_send_monotonic_s'])
            return
        if clock() >= until:
            raise TimeoutError('等待位未在时限内达到原0.005 rad、稳定0.1秒条件')
        sleep(.02)


def main(argv=None, *, hardware_factory=None, clock=time.monotonic, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--speed-percent', type=int, help='1..100，默认读取现场配置')
    parser.add_argument('--timeout-s', type=float, default=RETURN_WAIT_TIMEOUT_S)
    parser.add_argument('--output', type=Path, help='执行日志的新目录')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='连接CAN并移动；不自动使能或操作夹爪')
    mode.add_argument('--check-config', action='store_true', help='只检查配置（默认）')
    args = parser.parse_args(argv)
    try:
        config = WaitConfig.load(args.config if args.config is not None else default_config())
        speed = config.speed_percent if args.speed_percent is None else args.speed_percent
        if not 1 <= speed <= 100 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
            raise ValueError('速度须为1..100，超时须为正有限秒数')
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        if args.execute:
            print(f'[停止] 配置检查未通过：{short(error)}', flush=True)
        else:
            print(json.dumps({'status': 'configuration_invalid', 'error': str(error), 'joint_target_commands': 0}, ensure_ascii=False))
        return 2
    report = make_report(config, speed)
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False))
        return 0
    output = args.output or Path('runs/manual_return_wait')/(
        datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    try:
        output.mkdir(parents=True, exist_ok=False)
        console = ConsoleProgress()
        journal = RunJournal(output, clock=clock, observer=console.observe)
    except OSError as error:
        print(f'[停止] 无法创建运行日志：{short(error)}', flush=True)
        return 2
    device, cancelled = None, Event()
    handlers = {number: signal.signal(number, lambda *_: cancelled.set()) for number in (signal.SIGINT, signal.SIGTERM)}
    begun = clock()
    try:
        journal.emit('return_wait_started', **report)
        print(f'[连接] 机械臂 {config.channel}', flush=True)
        if hardware_factory is None:
            from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
            hardware_factory = Hardware
        device = hardware_factory(config.channel, config.firmware, True)
        device.start()
        report['hardware_connections'] = 1
        execute_return_wait(device, config, report, journal.emit, cancelled,
                 timeout_s=args.timeout_s, clock=clock, sleep=sleep)
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, (MoveCancelled, KeyboardInterrupt)) else 'stopped',
                      error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', {}))
    finally:
        if device is not None:
            try:
                device.close()
            except BaseException as error:
                report['disconnect_error'] = str(error)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        report.update(total_wall_s=clock()-begun, output=str(output.resolve()))
        journal.emit('return_wait_finished', **report)
        report['journal'] = journal.close()
        temporary = output/'result.json.tmp'
        temporary.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2)+'\n')
        temporary.replace(output/'result.json')
    if report['status'] in ('wait_reached', 'already_at_wait'):
        print('[完成] 等待位已确认', flush=True)
    else:
        print(f'[停止] 回等待位：{short(report.get("error", report["status"]))}', flush=True)
    print(f'[日志] {output.resolve()}', flush=True)
    return 0 if report['status'] in ('wait_reached', 'already_at_wait') else 1


if __name__ == '__main__':
    raise SystemExit(main())
