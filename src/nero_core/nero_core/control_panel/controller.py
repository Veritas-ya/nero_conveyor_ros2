"""Allowlisted process ownership for a single local operator panel."""
from collections import deque
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import RLock, Thread
import time
from zoneinfo import ZoneInfo


PRESETS = {
    'single': ('固定位置单帧', 'fixed_freeze_pick', 'two_second.hardware.legacy_observed.yaml'),
    'coarse': ('单帧 · 先换抓姿', 'fixed_freeze_coarse_first_pick', 'two_second.hardware.legacy_observed.yaml'),
    'tracked': ('多帧匹配', 'main', 'two_second.hardware.tracked_freeze.yaml'),
}


class PanelError(ValueError):
    pass


def motion_owner_busy():
    """Inspect the existing abstract-socket lease without acquiring it or CAN."""
    address = '@nero7-motion-v1-'+hashlib.sha256(b'socketcan:can0').hexdigest()
    try:
        return any(line.split()[-1:] == [address] for line in Path('/proc/net/unix').read_text().splitlines())
    except OSError:
        return None  # Actual device workers still enforce the original lease.


def tail_events(path, limit=80):
    """Bounded diagnostic read; never scan the whole continuous-run journal."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size-131072))
            lines = stream.read().splitlines()
        if size > 131072:
            lines = lines[1:]
        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeError):
                continue
        return rows
    except OSError:
        return []


class CanInterface:
    def __init__(self, runner=None):
        self.runner = runner or subprocess.run
        self.ip = shutil.which('ip') or '/usr/sbin/ip'

    def status(self):
        try:
            result = self.runner([self.ip, '-j', '-details', 'link', 'show', 'can0'],
                capture_output=True, text=True, timeout=3, check=False)
            if result.returncode:
                return {'present': False, 'up': False, 'bitrate': None,
                        'message': '未找到 can0，请检查 USB-CAN 连接'}
            value = json.loads(result.stdout)[0]
            info = value.get('linkinfo', {})
            data = info.get('info_data', {})
            bitrate = data.get('bittiming', {}).get('bitrate')
            is_can = info.get('info_kind') == 'can'
            up = 'UP' in value.get('flags', []) and is_can
            bus_state = data.get('state')
            usable = up and bitrate == 1000000 and bus_state != 'BUS-OFF'
            return {'present': True, 'up': up, 'ready': usable, 'is_can': is_can,
                    'bitrate': bitrate, 'bus_state': bus_state,
                    'message': ('can0 · 1 Mbit/s' if usable else
                                'CAN BUS-OFF，请检查总线' if bus_state == 'BUS-OFF' else 'can0 尚未就绪')}
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
            return {'present': False, 'up': False, 'bitrate': None,
                    'message': f'无法读取 can0：{error}'}

    def start(self):
        from mujoco_conveyor_sorting.runtime.motion_lease import MotionLease
        # Starting/reconfiguring CAN cannot interrupt a cooperative SDK owner.
        with MotionLease('can0', owner='control-panel-can-setup'):
            state = self.status()
            if state.get('ready'):
                return state
            if not state['present']:
                raise PanelError(state['message'])
            if not state.get('is_can'):
                raise PanelError('can0 不是 CAN 接口，未修改设备')
            prefix = [] if os.geteuid() == 0 else ['sudo', '-n']
            commands = []
            if state['up']:
                commands.append(['link', 'set', 'can0', 'down'])
            commands.extend([['link', 'set', 'can0', 'type', 'can', 'bitrate', '1000000'],
                             ['link', 'set', 'can0', 'up']])
            for command in commands:
                result = self.runner([*prefix, self.ip, *command], capture_output=True,
                    text=True, timeout=5, check=False)
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    raise PanelError('启动 can0 失败：'+detail+
                        '。若提示权限不足，请在同一终端执行 sudo -v 后重新启动界面；界面不保存密码。')
            state = self.status()
            if not state.get('ready'):
                raise PanelError('CAN 设置命令已结束，但接口未通过 1 Mbit/s 状态检查')
            return state


class PanelController:
    def __init__(self, workspace, *, demo=False, can=None, popen=None, owner_busy=None):
        self.root = Path(workspace).resolve()
        self.demo = demo
        self.can = can or CanInterface()
        self.popen = popen or subprocess.Popen
        self.owner_busy = owner_busy or motion_owner_busy
        self.lock = RLock()
        self.process = None
        self.thread = None
        self.logs = deque(maxlen=160)
        self.job = None
        self.error = None
        self.arm_result = None
        self.gripper_result = None
        self.can_state = {'present': False, 'up': False, 'message': '正在读取 can0'}
        self.sequence = 0
        self._last_can_check = -math.inf
        self._presets = None
        self.external_owner = False

    def presets(self):
        if self._presets is None:
            from nero_core.two_second.config import CellConfig
            records = []
            for key, (label, module, name) in PRESETS.items():
                path = self.root/'src/nero_bringup/config'/name
                cfg = CellConfig.load(path)
                records.append({'id': key, 'label': label, 'config': str(path),
                    'maximum_speed_cm_s': cfg.raw['maximum_measured_speed_m_s']*100,
                    'freeze_upstream_cm': None if key == 'tracked' else cfg.raw['freeze_position_upstream_m']*100,
                    'freeze_tolerance_cm': None if key == 'tracked' else cfg.raw['freeze_position_tolerance_m']*100,
                    'command_advance_ms': cfg.raw['command_advance_s']*1000,
                    'approach_s': cfg.raw['synchronized_join_candidates_s'][0],
                    'follow_s': cfg.settings.full_plan_close_window_s,
                    'open_width_mm': cfg.settings.open_width_m*1000,
                    'brake_s': cfg.raw['brake_duration_s'], 'module': module})
            self._presets = records
        return [dict(r) for r in self._presets]

    def validate(self, options):
        if not isinstance(options, dict) or set(options)-{'preset', 'belt_speed_cm_s', 'continuous'}:
            raise PanelError('运行参数包含未知字段')
        preset = options.get('preset', 'single')
        selected = next((p for p in self.presets() if p['id'] == preset), None)
        if selected is None:
            raise PanelError('请选择已有的抓取算法')
        speed = options.get('belt_speed_cm_s')
        if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or
                not math.isfinite(speed) or not 0 < speed <= selected['maximum_speed_cm_s']):
            raise PanelError(f'实测带速必须大于 0，且不超过 {selected["maximum_speed_cm_s"]:g} cm/s')
        continuous = options.get('continuous', False)
        if not isinstance(continuous, bool):
            raise PanelError('连续抓取参数必须为布尔值')
        return {**selected, 'belt_speed_cm_s': float(speed), 'continuous': continuous}

    def _log(self, message):
        with self.lock:
            self.sequence += 1
            self.logs.append({'id': self.sequence, 'time': datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S'),
                              'text': str(message)[:2000]})

    def _idle(self):
        if self.job and self.job['active']:
            raise PanelError('已有任务正在运行，请先停止并等待设备资源释放')

    def operator_options(self):
        """Restore form values only; this never resumes a device operation."""
        if self.demo:
            return None
        try:
            values = json.loads((self.root/'runs/control_panel/operator_options.json').read_text())
            validated = self.validate(values)
            return {'preset': validated['id'], 'belt_speed_cm_s': validated['belt_speed_cm_s'],
                    'continuous': validated['continuous']}
        except (OSError, ValueError, TypeError):
            return None

    def _output(self, kind):
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        path = self.root/'runs/control_panel'/f'{kind}_{stamp}'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def command(self, kind, options, output):
        package = 'nero_core.control_panel.'
        if kind == 'enable':
            return [sys.executable, '-u', '-m', package+'device_worker',
                    '--config', options['config'], '--output', str(output)]
        if kind == 'open_gripper':
            return [sys.executable, '-u', '-m', package+'gripper_worker',
                    '--config', options['config'], '--output', str(output)]
        taught = self.root/'taught/session_20261104_223416'
        if kind == 'camera':
            return [sys.executable, '-u', '-m', package+'camera_worker',
                    '--config', options['config'], '--output', str(output),
                    '--belt-speed-cm-s', str(options['belt_speed_cm_s']),
                    '--preset', options['id'], '--workspace', str(self.root)]
        return [sys.executable, '-u', '-m', 'nero_core.two_second.'+options['module'],
                '--config', options['config'], '--belt-speed-cm-s', str(options['belt_speed_cm_s']),
                '--taught-transfer', str(taught/'transfer_trimmed_start025_20261105.json'),
                '--taught-return', str(taught/'planned_return/return_generated.json'),
                '--continuous' if options['continuous'] else '--count',
                *([] if options['continuous'] else ['1']),
                '--execute', '--web-preview', '--output', str(output)]

    def start(self, action, options):
        if action == 'observe':
            self.observe()
            return
        if action not in ('can', 'enable', 'open_gripper', 'camera', 'pick'):
            raise PanelError('未知操作')
        with self.lock:
            self._idle()
            if not self.demo and self.owner_busy():
                raise PanelError('另一个程序正在控制机械臂。可“查看已有运行”；开始新任务前请在原终端停止原程序。')
            validated = self.validate(options)
            if action == 'camera' and validated['id'] == 'tracked':
                raise PanelError('相机规划预览使用单帧入口，请选择固定单帧或先换抓姿')
            if action in ('enable', 'open_gripper', 'pick') and not self.demo:
                state = self.can.status()
                if not state.get('ready'):
                    raise PanelError(state['message']+'；请先启动 can0')
            output = self._output(action)
            if not self.demo:
                values = {'preset': validated['id'], 'belt_speed_cm_s': validated['belt_speed_cm_s'],
                          'continuous': validated['continuous']}
                try:
                    temporary = self.root/'runs/control_panel/operator_options.tmp'
                    temporary.write_text(json.dumps(values))
                    temporary.replace(temporary.with_suffix('.json'))
                except OSError as error:
                    self._log('界面参数保存失败：'+str(error))
            self.error = None
            self.job = {'kind': action, 'active': True, 'stopping': False, 'run': str(output),
                'started_monotonic_s': time.monotonic(), 'options': validated, 'exit_code': None,
                'demo': self.demo}
            self._log({'can': '启动 can0', 'enable': '请求机械臂使能',
                       'open_gripper': '请求打开夹爪',
                       'camera': '启动仅相机点云与模型规划', 'pick': '启动真机抓取'}[action])
            if self.demo:
                if action == 'can':
                    self.can_state = {'present': True, 'up': True, 'ready': True,
                        'bitrate': 1000000, 'message': '演示 can0'}
                    self.job.update(active=False, exit_code=0)
                elif action == 'enable':
                    self.arm_result = {'enabled': True, 'demo': True}
                    self.job.update(active=False, exit_code=0)
                elif action == 'open_gripper':
                    self.gripper_result = {'opened': True, 'demo': True,
                        'width_m': validated['open_width_mm']/1000}
                    self.job.update(active=False, exit_code=0)
                return
            if action == 'can':
                self.thread = Thread(target=self._can_job, daemon=True)
                self.thread.start()
                return
            env = dict(os.environ)
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONPATH'] = str(self.root/'src/nero_core')+os.pathsep+env.get('PYTHONPATH', '')
            try:
                self.process = self.popen(self.command(action, validated, output),
                    cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True)
            except OSError as error:
                self.job.update(active=False, exit_code=-1)
                self.error = str(error)
                raise PanelError('启动进程失败：'+str(error)) from error
            self.job['pid'] = self.process.pid
            self.thread = Thread(target=self._read_output, args=(self.process, output), daemon=True)
            self.thread.start()

    def _can_job(self):
        with self.lock:
            job = self.job
        error = None
        state = dict(self.can_state)
        try:
            state = self.can.start()
        except Exception as failure:
            error = str(failure)
        finally:
            run = Path(job['run'])
            try:
                run.mkdir(parents=True, exist_ok=True)
                (run/'panel_status.json').write_text(json.dumps({'can': state,
                    'error': error, 'exit_code': 1 if error else 0}, ensure_ascii=False, indent=2))
            except OSError as failure:
                error = (error+'；' if error else '')+'记录写入失败：'+str(failure)
            with self.lock:
                self.can_state = state
                self.error = error
                job.update(active=False, exit_code=1 if error else 0)
                self._log(error or 'can0 已就绪，1 Mbit/s')

    def _read_output(self, process, output):
        try:
            with output.with_suffix('.console.log').open('w') as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self._log(line.rstrip())
        except Exception as error:
            self._log('输出日志读取异常：'+str(error))

    def refresh(self):
        with self.lock:
            if not self.demo and time.monotonic()-self._last_can_check >= 2.:
                self._last_can_check = time.monotonic()
                self.can_state = self.can.status()
                self.external_owner = bool(self.owner_busy() and not
                    (self.process is not None and self.job and self.job['active'] and self.job['kind'] in ('pick', 'enable', 'open_gripper')))
            if self.process is not None:
                code = self.process.poll()
                if code is not None:
                    self.job.update(active=False, stopping=False, exit_code=code)
                    self.process = None
                    result = self.worker_status()
                    if self.job['kind'] == 'enable':
                        self.arm_result = result
                    elif self.job['kind'] == 'open_gripper':
                        self.gripper_result = result
                    if code and not self.job.get('stop_requested'):
                        self.error = result.get('error') or '运行退出，请查看运行日志和记录目录'
                    self._log(f'任务已结束（退出码 {code}），记录已保留')

    def worker_status(self):
        if self.job:
            for name in ('panel_status.json', 'summary.json'):
                try:
                    return json.loads((Path(self.job['run'])/name).read_text())
                except (OSError, ValueError):
                    continue
        return {}

    def stop(self):
        with self.lock:
            if not self.job or not self.job['active']:
                return
            if self.job['kind'] == 'can':
                raise PanelError('CAN 设置操作正在结束，请稍候')
            self.job.update(stopping=True, stop_requested=True)
            if self.job['kind'] == 'observe':
                self.job.update(active=False, stopping=False, exit_code=0)
                self._log('已结束只读查看；原终端的设备程序保持原状态')
                return
            if self.demo:
                self.job.update(active=False, stopping=False, exit_code=0)
            elif self.process is not None and self.process.poll() is None:
                # Existing handlers cancel the current task and release owners.
                # Do not kill a sender or switch a moving arm's CAN interface.
                try:
                    os.killpg(self.process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            self._log('已请求停止，等待原程序结束并释放相机与控制权')

    def observe(self, run=None):
        with self.lock:
            self._idle()
            if run is None:
                candidates = list((self.root/'runs').glob('*/preview/latest.npz'))
                candidates += list((self.root/'runs/control_panel').glob('*/preview/latest.npz'))
                candidates = [p for p in candidates if p.is_file()]
                if not candidates:
                    raise PanelError('没有找到共享画面；原程序需使用 --preview 或 --web-preview 启动')
                run = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.parent
            run = Path(run).resolve()
            if not run.is_relative_to(self.root/'runs') or not (run/'preview/latest.npz').is_file():
                raise PanelError('只能查看本工作区 runs 内已有的共享预览记录')
            records = tail_events(run/'events.jsonl')
            progress = next((r.get('data', {}) for r in reversed(records)
                             if r.get('event') == 'runtime_progress'), {})
            policy = (progress.get('vision') or {}).get('velocity_policy') or {}
            speed = policy.get('fixed_belt_speed_cm_s')
            selected = self.validate(options={
                'preset': 'single', 'belt_speed_cm_s': speed if isinstance(speed, (int, float)) and speed > 0 else 43.5,
                'continuous': True})
            self.job = {'kind': 'observe', 'active': True, 'stopping': False, 'run': str(run),
                'started_monotonic_s': time.monotonic(), 'options': selected, 'exit_code': None,
                'read_only': True, 'demo': self.demo}
            self.error = None
            self._log('只读接入已有运行：'+str(run))

    def rearm(self):
        with self.lock:
            if not self.job or not self.job['active'] or self.job['kind'] != 'camera':
                raise PanelError('“下一件”仅用于正在运行的相机规划预览')
            if not self.demo:
                try:
                    os.kill(self.process.pid, signal.SIGUSR1)
                except ProcessLookupError as error:
                    raise PanelError('相机进程已经结束') from error
            self._log('相机规划预览：重新等待下一件')

    def snapshot(self):
        with self.lock:
            job = None if self.job is None else dict(self.job)
            if job:
                job['elapsed_s'] = max(0., time.monotonic()-job['started_monotonic_s'])
            worker = self.worker_status()
            events = []
            if job and job['kind'] in ('pick', 'observe'):
                events = tail_events(Path(job['run'])/'events.jsonl')
            progress = next((r.get('data', {}) for r in reversed(events)
                             if r.get('event') in ('runtime_progress', 'phase_started')), {})
            return {'demo': self.demo, 'job': job, 'can': dict(self.can_state),
                    'external_owner_busy': self.external_owner,
                    'arm': self.arm_result, 'worker': worker, 'progress': progress,
                    'gripper': self.gripper_result,
                    'error': self.error, 'logs': list(self.logs), 'events': events[-8:]}

    def close(self):
        self.stop()
        process = self.process
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f'原运行仍在结束，请保留终端并检查进程 {process.pid}；未强制终止控制器。', flush=True)
