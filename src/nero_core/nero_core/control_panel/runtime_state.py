"""Incremental, read-only projection of the existing run journal for the UI."""
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
import time
import json
from nero_core.two_second.console import step_name
from nero_core.two_second.diagnostics import json_value


def identity_key(identity):
    return ':'.join(map(str, identity)) if isinstance(identity, (tuple, list)) else None


class RuntimeState:
    def __init__(self):
        self.run = None
        self.offset = 0
        self.pending = b''
        self.inode = None
        self.caught_up = True
        self.config = {}
        self.progress = {}
        self.targets = OrderedDict()
        self.decisions = deque(maxlen=10)
        self.completed = self.skipped = 0
        self.last_wall = None
        self.next_identity = None
        self.cycles = OrderedDict()
        self.error = None

    def update(self, run):
        run = None if run is None else str(run)
        if run != self.run:
            self.__init__()
            self.run = run
        if run is None:
            return
        path = Path(run)/'events.jsonl'
        try:
            stat = path.stat()
            if self.inode is not None and (stat.st_ino != self.inode or stat.st_size < self.offset):
                self.__init__()
                self.run = run
            self.inode = stat.st_ino
            with path.open('rb') as stream:
                stream.seek(self.offset)
                block = stream.read(1024*1024)
                self.offset = stream.tell()
            self.caught_up = self.offset >= stat.st_size
            lines = (self.pending+block).split(b'\n')
            self.pending = lines.pop()
            for line in lines:
                try:
                    self.consume(json.loads(line))
                except (KeyError, ValueError, TypeError, AttributeError):
                    self.error = '运行日志中有不可解析记录'
        except FileNotFoundError:
            pass  # Startup may precede the journal's creation.
        except OSError as error:
            self.error = str(error)

    def consume(self, row):
        name, data = row.get('event'), row.get('data', {})
        if row.get('wall_time_utc'):
            self.last_wall = datetime.fromisoformat(row['wall_time_utc']).timestamp()
        if name == 'run_started':
            self.config = {key: data[key] for key in ('config', 'freeze_source', 'continuous',
                'panel_parameters', 'experimental_direct_drop_guard', 'taught_playback_enabled',
                'velocity_policy', 'follow_duration_s') if key in data}
        elif name in ('runtime_progress', 'phase_started'):
            self.progress = data
        elif name == 'cycle_finished':
            self.completed = data.get('completed_feedback_cycles', self.completed+1)
        if name == 'qualified_target':
            self.cycles[data['cycle_index']] = data['identity']
            while len(self.cycles) > 64:
                self.cycles.popitem(last=False)
        cycle_identity = self.cycles.get(data.get('cycle_index')) if name == 'execution_event' else None
        if name == 'execution_event':
            record = data.get('record', {})
            name, data = record.get('event'), record.get('data', {})
        if name == 'attempt_cancelled':
            details = data.get('details', {})
            if details.get('reason') not in ('lookahead_send_deadline_missed', 'lookahead_start_mismatch'):
                return
            name = 'planning_delivery_missed'
            data = {'reason': data.get('error'), 'failure_details': details, 'next_action': 'skip_target'}
        identity = data.get('identity', data.get('target_identity', cycle_identity))
        key = identity_key(identity)
        if key is None:
            return
        item = self.targets.setdefault(key, {'identity': identity, 'status': 'observed'})
        self.targets.move_to_end(key)
        while len(self.targets) > 64:
            self.targets.popitem(last=False)
        if name == 'lookahead_frozen':
            self.next_identity = key
            item.clear()
            item.update(identity=identity, status='frozen', frame_sequence=data.get('frame_sequence'))
        elif name == 'lookahead_planning_started':
            self.next_identity = key
            item.update(status='planning', submitted_s=row.get('monotonic_s'))
        elif name == 'lookahead_plan_ready':
            self.next_identity = key
            item.update(status='ready', send_deadline_s=data['send_deadline_monotonic_s'],
                        planning_s=data.get('planning_wall_s'), ready_s=data.get('ready_monotonic_s'))
        elif name in ('lookahead_plan_adopted', 'planning_delivery'):
            item['status'] = 'adopted' if name == 'lookahead_plan_adopted' else 'delivered'
            if key == self.next_identity:
                self.next_identity = None
        elif name in ('lookahead_rejected', 'planning_delivery_missed'):
            details = data.get('failure_details', data)
            if name == 'planning_delivery_missed' and data.get('next_action') != 'skip_target':
                item['status'] = 'replanning'
                return
            if item['status'] != 'skipped':
                self.skipped += 1
                self.decisions.appendleft({'identity': identity, 'reason': data.get('reason'),
                    'reason_code': details.get('reason'),
                    'return_confirmed': details.get('return_confirmed'),
                    'send_deadline_s': details.get('send_deadline_monotonic_s'),
                    'monotonic_s': row.get('monotonic_s')})
            item.update(status='skipped', reason=data.get('reason'))
            if key == self.next_identity:
                self.next_identity = None
        elif name in ('lookahead_plan_expired_replan', 'lookahead_planning_deferred'):
            item.update(status='replanning', reason=data.get('reason'))
            item.pop('send_deadline_s', None)

    def snapshot(self, *, finished=False):
        age = None if self.last_wall is None else datetime.now(timezone.utc).timestamp()-self.last_wall
        live = self.caught_up and not finished and age is not None and 0 <= age < 3.
        next_item = self.targets.get(self.next_identity)
        next_item = None if next_item is None else dict(next_item)
        if next_item and next_item.get('send_deadline_s') is not None:
            # Browser clocks are never compared to the producer's monotonic epoch.
            next_item['remaining_send_s'] = next_item['send_deadline_s']-time.monotonic() if live else None
        stage = self.progress.get('execution_stage') or self.progress.get('phase')
        label = step_name(stage) if stage else None
        if label and self.config.get('experimental_direct_drop_guard'):
            label = label.replace('示教', '')
        return {'config': self.config, 'progress': {key: self.progress.get(key) for key in
                    ('phase', 'execution_stage', 'lookahead')}, 'next': next_item,
                'stage_label': label,
                'targets': {key: dict(item) for key, item in self.targets.items()}, 'completed': self.completed,
                'skipped': self.skipped, 'decisions': list(self.decisions),
                'live': live, 'caught_up': self.caught_up, 'journal_age_s': age, 'error': self.error}


class LivePanelStatus:
    """Project the running process's events into one replaceable shared-memory snapshot."""
    def __init__(self, directory):
        self.path = Path(directory)/'preview/status.json'
        self.path.parent.mkdir(exist_ok=True)
        self.state = RuntimeState()
        self.worker = {}
        self.last_write = -float('inf')

    def observe(self, row):
        self.state.consume(row)
        event = row['event']
        if event == 'run_failed':
            self.worker['error'] = row['data'].get('error')
        if event == 'run_finished':
            self.worker['status'] = row['data']['status']
        now = time.monotonic()
        if now-self.last_write < .2 and event not in ('run_started', 'run_failed', 'run_finished'):
            return
        value = {'updated_monotonic_s': now, 'worker': self.worker,
                 'run_view': self.state.snapshot(finished=event == 'run_finished')}
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False))
        temp.replace(self.path)
        self.last_write = now
