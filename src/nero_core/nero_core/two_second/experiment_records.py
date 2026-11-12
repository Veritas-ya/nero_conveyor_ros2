"""Offline, auditable trial ledger. Software feedback never labels physical success."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def initialize(directory, plan):
    with sqlite3.connect(Path(directory)/'trials.sqlite3') as db:
        db.executescript('''
            CREATE TABLE trials (
                trial_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL,
                group_name TEXT NOT NULL, workpiece_id TEXT NOT NULL,
                pair_id TEXT NOT NULL, phase TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'not_started', started_utc TEXT,
                video_file TEXT, video_offset_s REAL, run_path TEXT, cycle_index INTEGER,
                program_json TEXT, physical_grasp INTEGER, physical_place INTEGER,
                physical_return INTEGER, measured_speed_m_s REAL,
                observed_cycle_time_s REAL, observed_termination_time_s REAL,
                reason TEXT, evidence_note TEXT, updated_utc TEXT);
            CREATE UNIQUE INDEX one_binding ON trials(run_path, cycle_index)
                WHERE run_path IS NOT NULL;
            CREATE TABLE audit (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL, action TEXT NOT NULL, utc TEXT NOT NULL,
                payload_json TEXT NOT NULL);
        ''')
        for item in plan:
            db.execute('INSERT INTO trials(trial_id,ordinal,group_name,workpiece_id,pair_id,phase) '
                       'VALUES(?,?,?,?,?,?)', tuple(item[k] for k in
                       ('trial_id', 'ordinal', 'group', 'workpiece_id', 'pair_id', 'phase')))


class Ledger:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        path = self.directory/'trials.sqlite3'
        self.db = sqlite3.connect(path.as_uri()+'?mode=rw', uri=True, timeout=10)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def rows(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM trials ORDER BY ordinal')]

    def _row(self, trial_id):
        row = self.db.execute('SELECT * FROM trials WHERE trial_id=?', (trial_id,)).fetchone()
        if row is None:
            raise ValueError('unknown trial_id')
        return dict(row)

    def _audit(self, trial_id, action, payload):
        self.db.execute('INSERT INTO audit(trial_id,action,utc,payload_json) VALUES(?,?,?,?)',
            (trial_id, action, utc(), json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def begin(self, trial_id, *, video_file, video_offset_s):
        offset = finite(video_offset_s, 'video_offset_s')
        if not str(video_file).strip():
            raise ValueError('video filename is required; file existence is checked at export')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self._row(trial_id)
            if row['status'] != 'not_started':
                raise ValueError('trial already started; never overwrite an existing attempt')
            video = str(Path(video_file).expanduser().resolve())
            self.db.execute("UPDATE trials SET status='pending',started_utc=?,video_file=?,"
                'video_offset_s=?,updated_utc=? WHERE trial_id=?', (utc(), video, offset, utc(), trial_id))
            self._audit(trial_id, 'begin', {'video_file': video, 'video_offset_s': offset})

    def bind(self, trial_id, run, cycle):
        if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 1:
            raise ValueError('cycle must be a positive integer')
        run = Path(run).expanduser().resolve(strict=True)
        row = self._row(trial_id)
        if row['status'] == 'not_started' or row['run_path'] is not None:
            raise ValueError('begin trial first; a trial can bind to only one program attempt')
        expected = self.manifest['groups'][row['group_name']]
        config = self.directory/expected['config']
        if digest(config) != expected['config_sha256']:
            raise ValueError('session experiment configuration has changed')
        program = read_attempt(run, cycle)
        observed = program.pop('source_hashes')
        # Check the generated YAML, calibrations and frozen source, not a name
        # guessed from a successful physical outcome.
        for path, sha in expected['source_hashes'].items():
            if observed.get(path) != sha:
                raise ValueError('run provenance differs from frozen session: '+path)
        effective = program['effective_configuration']
        velocity = effective.get('velocity_policy')
        if not isinstance(velocity, dict) or any(velocity.get(k) != v
               for k, v in expected['velocity_policy'].items()):
            raise ValueError('effective velocity input differs from frozen session (including CLI overrides)')
        for key in ('freeze_position_upstream_m', 'freeze_position_tolerance_m'):
            if effective.get(key) != expected[key]:
                raise ValueError('effective observation freeze differs from frozen session')
        if effective.get('freeze_source') != expected['freeze_source']:
            raise ValueError('effective acquisition source differs from frozen session')
        taught = {k: v['sha256'] for k, v in (effective.get('taught_segments') or {}).items()}
        if taught != expected['taught_sha256']:
            raise ValueError('effective taught paths differ from frozen session (including CLI overrides)')
        ready = program['plan_attempts']
        if any(p['group'] != row['group_name'] for p in ready):
            raise ValueError('run timing group differs from planned trial')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self._row(trial_id)['run_path'] is not None:
                raise ValueError('trial already bound')
            self.db.execute('UPDATE trials SET run_path=?,cycle_index=?,program_json=?,updated_utc=? '
                'WHERE trial_id=?', (str(run), cycle, json.dumps(program, ensure_ascii=False,
                allow_nan=False), utc(), trial_id))
            self._audit(trial_id, 'bind', {'run_path': str(run), 'cycle_index': cycle,
                'evidence_hashes': program['evidence_hashes']})

    def finish(self, trial_id, *, status, grasp=None, place=None, returned=None,
               reason='', evidence_note='', measured_speed_m_s=None, correction_reason=None,
               observed_cycle_time_s=None, observed_termination_time_s=None):
        if status not in ('success', 'failure', 'external_interruption'):
            raise ValueError('status must explicitly be success, failure or external_interruption')
        stages = (grasp, place, returned)
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1))
               for v in stages):
            raise ValueError('physical stages must be integer 0, 1 or omitted')
        if status == 'success' and stages != (1, 1, 1):
            raise ValueError('success requires physical pickup, placement AND return confirmation')
        if status == 'failure' and stages == (1, 1, 1):
            raise ValueError('failure cannot have three successful physical stages')
        if status != 'success' and not reason.strip():
            raise ValueError('failure and interruption require a reason')
        if not evidence_note.strip():
            raise ValueError('record the physical observation or video evidence')
        speed = None if measured_speed_m_s is None else finite(
            measured_speed_m_s, 'independently measured speed', positive=True)
        cycle = None if observed_cycle_time_s is None else finite(
            observed_cycle_time_s, 'observed cycle time', positive=True)
        termination = None if observed_termination_time_s is None else finite(
            observed_termination_time_s, 'observed termination time', positive=True)
        if (cycle is not None and status != 'success') or (termination is not None and status == 'success'):
            raise ValueError('successful cycle time and unsuccessful termination time must be separate')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self._row(trial_id)
            if row['status'] == 'not_started':
                raise ValueError('begin trial before recording an outcome')
            if row['status'] != 'pending' and not (correction_reason and correction_reason.strip()):
                raise ValueError('terminal outcomes require an explicit correction reason')
            if correction_reason is not None and not correction_reason.strip():
                raise ValueError('correction reason cannot be blank')
            payload = dict(status=status, physical_grasp=grasp, physical_place=place,
                physical_return=returned, measured_speed_m_s=speed, reason=reason,
                evidence_note=evidence_note, correction_reason=correction_reason,
                observed_cycle_time_s=cycle, observed_termination_time_s=termination)
            self.db.execute('UPDATE trials SET status=?,physical_grasp=?,physical_place=?,physical_return=?, '
                'measured_speed_m_s=?,reason=?,evidence_note=?,observed_cycle_time_s=?, '
                'observed_termination_time_s=?,updated_utc=? WHERE trial_id=?',
                (status, *stages, speed, reason, evidence_note, cycle, termination, utc(), trial_id))
            self._audit(trial_id, 'correct' if correction_reason else 'finish',
                        {'before': row, 'after': payload})


def read_attempt(run, cycle):
    """Read archived cycle or journal without reconstructing missing observations."""
    run = Path(run)
    evidence = {}
    def read(path):
        evidence[str(path)] = digest(path)
        return json.loads(path.read_text())
    journal = []
    if (run/'events.jsonl').exists():
        path = run/'events.jsonl'
        evidence[str(path)] = digest(path)
        # A partial final line is evidence of an incomplete archive, not an
        # empty successful run. Bind only a flushed/finished journal.
        journal = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [r['data'] for r in journal if r.get('event') == 'run_started']
    summary = read(run/'summary.json') if (run/'summary.json').exists() else {}
    hashes = summary.get('source_hashes') or (starts[-1].get('source_hashes') if starts else None)
    if not hashes:
        raise ValueError('run has no source provenance; do not assign it to a formal group')
    directory = run/f'cycle_{cycle:03d}'
    if (directory/'events.json').exists():
        events = read(directory/'events.json')
    else:
        events = [r['data']['record'] for r in journal if r.get('event') == 'execution_event'
                  and r.get('data', {}).get('cycle_index') == cycle]
    qualified = [r['data'] for r in journal if r.get('event') == 'qualified_target'
                 and r.get('data', {}).get('cycle_index') == cycle]
    if not events and not qualified:
        raise ValueError('cycle not found; detection misses must be annotated without a program binding')
    submissions = [r for r in events if r.get('event') == 'planning_submitted']
    identities = {json.dumps(r['data']['identity']) for r in submissions}
    identities.update(json.dumps(r['identity']) for r in qualified)
    if len(identities) != 1:
        raise ValueError('program attempt does not identify exactly one target')
    ready = [r['data'] for r in events if r.get('event') == 'experiment_plan_ready']
    delivered = [r['data'] for r in events if r.get('event') == 'planning_delivery']
    failures = [r['data'] for r in events if r.get('event') == 'experiment_plan_failed']
    deliveries = [r['data'] for r in events if r.get('event') == 'experiment_delivery_missed']
    submitted = submissions[0]['monotonic_s'] if submissions else None
    planned = ready[-1] if ready else {}
    matched_delivery = [d for d in delivered if planned and
        d['delivery_monotonic_s'] >= planned['timing']['completed_monotonic_s']]
    capture = (submissions[0]['data'].get('capture_monotonic_s') if submissions else
               qualified[0].get('capture_monotonic_s'))
    report = read(directory/'report.json') if (directory/'report.json').exists() else None
    finished = [r for r in events if r.get('event') == 'stage_finished']
    program_end = finished[-1]['monotonic_s'] if finished else None
    first_dispatch = None
    trace = directory/'execution_trace.npz'
    if trace.exists():
        import numpy as np
        evidence[str(trace)] = digest(trace)
        with np.load(trace, allow_pickle=False) as data:
            stages = data['sample_stage']
            selected = data['dispatch_succeeded'] & np.isfinite(data['dispatch_start_monotonic_s'])
            selected &= np.array(['intercept' in str(s) for s in stages], dtype=bool)
            values = data['dispatch_start_monotonic_s'][selected]
            if len(values):
                first_dispatch = float(values.min())
    history = []
    for p in ready:
        history.extend(p.get('timing', {}).get('rounds', []))
    for f in failures:
        history.extend(f.get('details', {}).get('timing_experiment', {}).get('rounds', []))
    return {'source_hashes': hashes, 'target_identity': json.loads(next(iter(identities))),
        'effective_configuration': {key: summary.get(key, (starts[-1] if starts else {}).get(key))
            for key in ('velocity_policy', 'freeze_source', 'freeze_position_upstream_m',
                        'freeze_position_tolerance_m', 'taught_segments')},
        'plan_attempts': ready, 'planning_failures': failures, 'delivery_misses': deliveries,
        'diagnostic_errors': [r['data'] for r in events if r.get('event') == 'experiment_record_error'],
        'planning_rounds': history, 'program_report': report, 'evidence_hashes': evidence,
        'capture_monotonic_s': capture, 'submission_monotonic_s': submitted,
        'first_intercept_dispatch_monotonic_s': first_dispatch,
        'post_planning_to_delivery_s': (matched_delivery[-1]['delivery_monotonic_s']-
            planned['timing']['completed_monotonic_s'] if matched_delivery else None),
        'planning_wall_s_all_rounds': sum(r['wall_s'] for r in history) if history else None,
        'regeneration_count': (max(0, len(history)-1) if history else None),
        'program_submission_to_last_stage_s': (program_end-submitted
            if program_end is not None and submitted is not None else None),
        'exposure_to_submission_s': (submitted-capture if submitted is not None and capture is not None else None),
        'submission_to_first_dispatch_s': (first_dispatch-submitted
            if first_dispatch is not None and submitted is not None else None),
        'deadline_margin_at_first_dispatch_s': (planned['send_deadline_monotonic_s']-first_dispatch
            if first_dispatch is not None and planned else None),
        'clock_domain': 'host_monotonic_within_this_run',
        'physical_success_inferred': False}


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    denominator = 1+z*z/n
    center = (k/n+z*z/(2*n))/denominator
    half = z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(rows):
    groups = {}
    for phase in sorted({r['phase'] for r in rows}):
        for group in ('A', 'B', 'C'):
            selected = [r for r in rows if r['phase'] == phase and r['group_name'] == group]
            ended = [r for r in selected if r['status'] in ('success', 'failure')]
            success = sum(r['status'] == 'success' for r in ended)
            groups[phase+':'+group] = {'planned': len(selected), 'completed': len(ended),
                'successes': success, 'success_rate': success/len(ended) if ended else None,
                'wilson_95': wilson(success, len(ended)),
                'not_started': sum(r['status'] == 'not_started' for r in selected),
                'pending': sum(r['status'] == 'pending' for r in selected),
                'external_interruptions': sum(r['status'] == 'external_interruption' for r in selected)}
            times = []
            for r in ended:
                if r['program_json']:
                    value = json.loads(r['program_json']).get('planning_wall_s_all_rounds')
                    if value is not None:
                        times.append(value)
            times.sort()
            def quantile(q):
                x = (len(times)-1)*q
                lo, hi = math.floor(x), math.ceil(x)
                return times[lo]+(times[hi]-times[lo])*(x-lo) if times else None
            groups[phase+':'+group]['planning_time_s'] = {'n': len(times),
                'median': quantile(.5), 'p95': quantile(.95), 'max': max(times) if times else None,
                'percentile_convention': 'linear interpolation at (n-1)*q; includes bound failures'}
    comparisons = {}
    for phase in sorted({r['phase'] for r in rows}):
        for low, high in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
            pairs = {}
            for row in rows:
                if row['phase'] == phase and row['status'] in ('success', 'failure'):
                    pairs.setdefault(row['pair_id'], {})[row['group_name']] = row['status'] == 'success'
            complete = [p for p in pairs.values() if low in p and high in p]
            better = sum(p[high] and not p[low] for p in complete)
            worse = sum(p[low] and not p[high] for p in complete)
            discordant = better+worse
            exact = min(1., 2*sum(math.comb(discordant, k)
                for k in range(min(better, worse)+1))/(2**discordant)) if discordant else 1.
            comparisons[f'{phase}:{high}-{low}'] = {'complete_pairs': len(complete),
                'only_higher_group_succeeds': better, 'only_lower_group_succeeds': worse,
                'success_rate_difference': (better-worse)/len(complete) if complete else None,
                'exact_mcnemar_p': exact if complete else None}
    per_workpiece = {}
    for r in rows:
        key = ':'.join([r['phase'], r['group_name'], r['workpiece_id']])
        stats = per_workpiece.setdefault(key, {'completed': 0, 'successes': 0})
        if r['status'] in ('success', 'failure'):
            stats['completed'] += 1
            stats['successes'] += r['status'] == 'success'
    for stats in per_workpiece.values():
        n, k = stats['completed'], stats['successes']
        stats.update(success_rate=k/n if n else None, wilson_95=wilson(k, n))
    return {'groups': groups, 'per_workpiece': per_workpiece, 'paired_comparisons': comparisons,
        'statistical_scope': 'Descriptive within the tested workpieces; shared batch/history can induce dependence. '
                             'Pilot and formal results are separate; planned and pending trials are never failures.'}


def export(ledger, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with ledger.db:
        ledger.db.execute('BEGIN')
        rows = ledger.rows()
        audit = [dict(r) for r in ledger.db.execute('SELECT * FROM audit ORDER BY seq')]
    # Export full program data separately so the editable CSV stays compact.
    metrics = ('planning_wall_s_all_rounds', 'post_planning_to_delivery_s', 'regeneration_count',
        'first_intercept_dispatch_monotonic_s', 'exposure_to_submission_s',
        'submission_to_first_dispatch_s', 'deadline_margin_at_first_dispatch_s',
        'program_submission_to_last_stage_s')
    columns = ([k for k in rows[0] if k != 'program_json'] if rows else ['trial_id']) + list(metrics)
    with (output/'trials.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            program = json.loads(r['program_json']) if r['program_json'] else {}
            writer.writerow({k: r.get(k, program.get(k)) for k in columns})
    (output/'program_evidence.json').write_text(json.dumps({r['trial_id']: json.loads(r['program_json'])
        for r in rows if r['program_json']}, ensure_ascii=False, indent=2, allow_nan=False))
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    summary = summarize(rows)
    summary['quality_warnings'] = [f'{r["trial_id"]}: video file missing' for r in rows
        if r['status'] != 'not_started' and (not r['video_file'] or not Path(r['video_file']).is_file())]
    summary['quality_warnings'] += [f'{r["trial_id"]}: no independently measured speed' for r in rows
        if r['status'] in ('success', 'failure') and r['measured_speed_m_s'] is None]
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary
