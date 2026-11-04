"""Pick startup recovery, before the camera or any frozen target exists."""
import time
import numpy as np

from ..move_to_wait import (WaitConfig, MoveCancelled, RETURN_WAIT_TIMEOUT_S,
                            make_report, validate_wait_joints, execute_return_wait)
from .execution import validate_start_state
from .feedback_gates import JOINT_ARRIVAL_TOLERANCE_RAD


def return_wait_policy(config):
    return {'enabled': True, 'joint_command_api': 'move_j',
        'speed_percent': config.settings.speed_percent, 'timeout_s': RETURN_WAIT_TIMEOUT_S,
        'arrival_tolerance_rad': JOINT_ARRIVAL_TOLERANCE_RAD,
        'restore_joint_command_mode': config.raw['hardware']['joint_command_mode'],
        'scene_collision_certified': False}


def prepare_start_state(config, device, initial_state, *, model, cancelled, emit, report,
                        clock=time.monotonic, sleep=time.sleep):
    """Reuse CLI movement/stop logic without another connection or signal owner."""
    hardware = config.raw['hardware']
    wait = WaitConfig(config.source, config.source_sha256, config.standby.copy(),
        hardware['can_channel'], hardware['firmware'], config.settings.speed_percent,
        config.settings.min_force_n)
    report.update(make_report(wait), status='checking', connection_reused=True,
                  initial_state=initial_state)
    begin = clock()

    def movement_event(event, **data):
        emit('startup_return_wait_'+event, **data)

    try:
        if cancelled.is_set():
            raise MoveCancelled('启动回位前已取消')
        # Joint-only recovery must not carry a possibly held workpiece or
        # silently open a jaw. Keep the pick's original width/force checks.
        validate_start_state(config, initial_state, require_wait=False)
        error = float(np.max(abs(np.asarray(initial_state['joints'])-wait.joints)))
        report['initial_max_abs_joint_error_rad'] = error
        if error <= JOINT_ARRIVAL_TOLERANCE_RAD:
            report.update(status='already_at_wait', last_state=initial_state,
                          max_abs_joint_error_rad=error)
            movement_event('skipped', reason='already_at_wait', max_abs_joint_error_rad=error)
            return initial_state
        validate_wait_joints(wait.joints, model)
        movement_event('started', **report)
        execute_return_wait(device, wait, report, movement_event, cancelled,
                            clock=clock, sleep=sleep)
        # J is only for recovery. Streaming must regain its configured JS/J
        # selection locally, with no movement target sent on the mode change.
        device.configure_joint_command_mode(hardware['joint_command_mode'])
        report['restored_joint_command_mode'] = hardware['joint_command_mode']
        if cancelled.is_set():
            raise MoveCancelled('回位后已取消，不再开启相机或抓取')
        state = device.snapshot()
        validate_start_state(config, state)
        report['last_state'] = state
        movement_event('passed', **report)
        return state
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, (MoveCancelled, KeyboardInterrupt)) else 'stopped',
            error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', {}))
        movement_event('failed', **report)
        raise
    finally:
        report['total_wall_s'] = clock()-begin
