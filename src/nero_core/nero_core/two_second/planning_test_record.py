"""Record a bounded camera-only planning test, once per accepted track identity."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from threading import RLock

import numpy as np

from .collision_capture import frame_metadata
from .diagnostics import json_value
from .reference import EntryNotReady
from .synchronized_reference import admit_synchronized_observation, freeze_window_for_speed
from .trajectory_preview import TrajectoryPreview
from .velocity_source import prediction_velocity


def wall_time():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')


class PlanningTestRecord:
    def __init__(self, directory, limit, *, velocity_policy=None):
        self.directory = Path(directory)
        self.limit = limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('attempt count must be a positive integer')
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory/'summary.json').exists():
            raise ValueError('测试记录目录已存在，请使用新目录，避免覆盖')
        self.rows = []
        self.seen = set()
        self.sessions = 0
        self.started = wall_time()
        self.closed = False
        self.velocity_policy = velocity_policy
        self.lock = RLock()
        self.pending_inputs = {}
        self.archives = []
        self.save()

    @property
    def complete(self):
        return len(self.rows) == self.limit and all(r['status'] != 'planning' and
            r.get('evidence_status') != 'pending' and r.get('path_archive_status') != 'pending'
            for r in self.rows)

    def camera_session(self):
        self.sessions += 1
        return self.sessions

    def begin(self, session, done, track):
        peer = done['peer']
        identity = (peer.clock_epoch, peer.track_generation, track.track_id)
        key = (session, *identity)
        if key in self.seen or len(self.rows) >= self.limit:
            return None
        self.seen.add(key)
        frame = done['geometry']['frame']
        index = len(self.rows)+1
        row = {'attempt':index, 'status':'planning', 'camera_session':session,
            'identity':identity, 'frame_sequence':frame.header.sequence,
            'started_at':wall_time(), 'capture_monotonic_s':done['capture'],
            'capture_age_at_submission_s':time.monotonic()-done['capture'],
            'model_path_generated':False, 'timing_feasible':None,
            'failure_reason':None, 'can_connections':0, 'robot_commands_sent':0}
        stem = self.directory/f'attempt_{index:02d}'
        row['evidence_file'] = str(stem.with_suffix('.npz').resolve())
        row['evidence_status'] = 'pending'
        self.rows.append(row)
        self.save()
        # Keep the original immutable exposure alive, but do not compress it
        # before planning. Save both successful and rejected inputs afterwards.
        meta = frame_metadata(frame, done['geometry']['d2c'])
        self.pending_inputs[index] = dict(metadata=np.asarray(json.dumps(json_value(meta))),
            frame_rgb=frame.rgb, frame_depth_m=frame.depth_m,
            object_points_base_m=done['detection']['object_points'],
            grasps=np.asarray(done['geometry']['grasps']))
        detection = done['detection']
        # Preserve the exact cached belt plane/scene too, so future replays do
        # not need to estimate a different plane from this exposure.
        self.pending_inputs[index]['detection_metadata'] = np.asarray(json.dumps(json_value({
            key:detection[key] for key in ('plane','belt_region') if key in detection})))
        self.pending_inputs[index]['target_metadata'] = np.asarray(json.dumps(json_value({
            'identity':identity, 'capture_monotonic_s':done['capture'],
            'observed_position_base_m':track.observed_position_base,
            'tracking_velocity_base_m_s':track.velocity_base_m_s})))
        for key in ('scene_points','scene_pixel_indices'):
            if key in detection:
                self.pending_inputs[index][key] = detection[key]
        print(f'[记录 {index}/{self.limit}] 目标 {identity}，开始规划', flush=True)
        return row

    def finish(self, row, *, result=None, error=None, wall_s, archive_pool=None):
        with self.lock:
            return self._finish(row, result=result, error=error, wall_s=wall_s, archive_pool=archive_pool)

    def _finish(self, row, *, result=None, error=None, wall_s, archive_pool=None):
        row.update(finished_at=wall_time(), planning_wall_s=wall_s)
        if error is not None:
            row.update(status='planning_failed', error_type=type(error).__name__,
                failure_reason=str(error), failure_details=json_value(getattr(error, 'details', {})))
        else:
            report = result['report']
            lead = report['actual_remaining_send_lead_s']
            row.update(status='path_generated' if lead >= 0 else 'path_generated_deadline_missed',
                model_path_generated=True, timing_feasible=bool(lead >= 0),
                remaining_send_lead_s=lead, trajectory_duration_s=report['trajectory_duration_s'],
                path_report_file=result['file']+'.json',
                path_arrays_file=result['file']+'.npz',
                candidate_failures=report['candidate_failures'],
                preview_timing=report['timing'],
                failure_reason=None if lead >= 0 else f'模型路径已生成，但真实发送截止已过 {-lead*1000:.1f} ms')
            if 'archive_future' in result:
                row['path_archive_status'] = 'pending'
                self.archives.append((row, result['archive_future'], 'path_archive_status'))
        arrays = self.pending_inputs.pop(row['attempt'])
        if archive_pool is None:
            try:
                np.savez_compressed(row['evidence_file'], **arrays)
                row['evidence_status'] = 'saved'
            except Exception as error:
                row.update(evidence_status='failed', evidence_error=str(error))
        else:
            self.archives.append((row, archive_pool.submit(np.savez_compressed, row['evidence_file'], **arrays), 'evidence_status'))
        # Keep a standalone readable record and append a completed-only journal.
        with (self.directory/'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(json_value(row), ensure_ascii=False)+'\n')
        self.save()
        message = row['failure_reason'] or '已生成模型路径，真实时间检查有余量'
        print(f'[记录 {row["attempt"]}/{self.limit}] {message}', flush=True)

    def save(self):
        with self.lock:
            self._save()

    def refresh_archives(self, *, wait=False):
        with self.lock:
            changed = False
            for row, future, field in self.archives:
                if row[field] != 'pending' or (not wait and not future.done()):
                    continue
                try:
                    future.result()
                    row[field] = 'saved'
                except Exception as error:
                    row[field] = 'failed'
                    row['evidence_error' if field == 'evidence_status' else 'path_archive_error'] = str(error)
                changed = True
            if changed:
                self.save()

    def _save(self):
        completed = [r for r in self.rows if r['status'] != 'planning']
        report = {'mode':'camera_only_planning_test', 'started_at':self.started,
            'updated_at':wall_time(), 'target_attempts':self.limit,
            'started_attempts':len(self.rows), 'completed_attempts':len(completed),
            'model_paths_generated':sum(r['model_path_generated'] for r in completed),
            'planning_failures':sum(r['status'] == 'planning_failed' for r in completed),
            'deadline_misses':sum(r['status'] == 'path_generated_deadline_missed' for r in completed),
            'complete':self.complete, 'camera_closed':self.closed,
            'can_connections':0, 'robot_commands_sent':0,
            'velocity_policy':self.velocity_policy,
            'counting_rule':'One attempt per independently identified target accepted for planning; candidate retries are included in the same attempt.',
            'pending_evidence_files':sum(r.get('evidence_status') == 'pending' for r in self.rows),
            'archive_failures':sum(r.get('evidence_status') == 'failed' or
                r.get('path_archive_status') == 'failed' for r in self.rows),
            'limitation':'Configured standby is assumed; planning uses a real monotonic clock. Deadline lead includes result preparation, excludes asynchronous archival and GUI delivery; this is not hardware execution certification.',
            'attempts':self.rows}
        temporary = self.directory/'summary.json.tmp'
        temporary.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2))
        temporary.replace(self.directory/'summary.json')
        lines = ['相机规划测试记录（不驱动机械臂）',
            f'完成 {len(completed)}/{self.limit} 次；生成模型路径 {report["model_paths_generated"]} 次；规划失败 {report["planning_failures"]} 次；生成后已错过发送截止 {report["deadline_misses"]} 次。',
            '按进入规划的独立目标计次；未识别/测速未合格不计次，原因保存在同目录 diagnostics.jsonl。',
            '使用假设等待位与真实规划时钟；截止余量计入结果准备，不含后台存档和界面轮询；路径生成不代表真机执行成功。', '']
        if self.velocity_policy is not None:
            speed = self.velocity_policy.get('fixed_belt_speed_cm_s')
            lines.insert(1, f'本轮固定预测带速：{speed:g} cm/s。' if speed is not None else
                '本轮预测带速来源：'+self.velocity_policy['source'])
        for row in self.rows:
            duration = row.get('planning_wall_s')
            duration = '进行中' if duration is None else f'{duration:.3f} 秒'
            lines.append(f'{row["attempt"]:02d}. {row["status"]}；耗时 {duration}；'+
                (row['failure_reason'] or ('尚在规划' if row['status']=='planning' else '无规划拒绝；真实发送截止有余量')))
        if self.complete:
            lines += ['', '10次记录已满，已停止提交新的规划；相机画面可继续保留。' if self.limit == 10 else '记录已满，已停止提交新的规划。']
        (self.directory/'record.txt').write_text('\n'.join(lines)+'\n')


class RecordingTrajectoryPreview(TrajectoryPreview):
    def __init__(self, config, output, record):
        super().__init__(config, output)
        self.record = record
        self.session = record.camera_session()

    def submit(self, done, gate):
        from .preview import qualified_tracks
        if self.record.complete:
            self.status = f'已完成 {self.record.limit}/{self.record.limit} 次记录；相机继续显示'
            if any(r.get('evidence_status') == 'failed' or r.get('path_archive_status') == 'failed'
                   for r in self.record.rows):
                self.status += '；有存档失败，请查看 summary.json'
            return
        if len(self.record.rows) >= self.record.limit:
            self.status = '规划记录已满，等待后台存档完成'
            return
        if self.future is not None or done.get('geometry') is None or done.get('capture') is None:
            return
        tracks = qualified_tracks(self.config, done['peer'])
        now = time.monotonic()
        if len(tracks) != 1 or now-self.last_submitted < 1.:
            return
        if not 0 <= now-done['capture'] <= self.config.raw['maximum_observation_age_s']:
            return
        track = tracks[0]
        velocity = prediction_velocity(self.config, track.velocity_base_m_s,
                                      done['detection']['plane']['normal_base'])
        # Do not spend a trial on an observation that production still waits
        # to admit. The real planner retains all geometry/candidate failures.
        try:
            window = freeze_window_for_speed(self.config, velocity@self.config.direction)
            admit_synchronized_observation(done['capture'],
                np.asarray(track.observed_position_base)@self.config.direction,
                velocity@self.config.direction, self.config.follow_bounds, now,
                window['distance_m'], minimum_upstream_distance_m=window.get('minimum_distance_m'))
        except EntryNotReady:
            return
        identity = (self.session, done['peer'].clock_epoch,
                    done['peer'].track_generation, track.track_id)
        if identity in self.record.seen:
            return
        self.last_submitted = now
        self.status = f'记录第 {len(self.record.rows)+1}/{self.record.limit} 次：正在规划'
        self.future = self.pool.submit(self._record_calculation, done, track, gate, self.generation)

    def _record_calculation(self, done, track, gate, generation):
        row = self.record.begin(self.session, done, track)
        if row is None:
            raise RuntimeError('该目标已记录或记录次数已满')
        started = time.monotonic()
        try:
            result = self._calculate(done, track, gate, generation)
        except Exception as error:
            self.record.finish(row, error=error, wall_s=time.monotonic()-started, archive_pool=self.archive_pool)
            raise
        self.record.finish(row, result=result, wall_s=time.monotonic()-started, archive_pool=self.archive_pool)
        return result

    def poll(self):
        result = super().poll()
        self.record.refresh_archives()
        if self.record.complete:
            self.status = f'已完成 {self.record.limit}/{self.record.limit} 次记录；相机继续显示'
            if any(r.get('evidence_status') == 'failed' or r.get('path_archive_status') == 'failed'
                   for r in self.record.rows):
                self.status += '；有存档失败，请查看 summary.json'
        else:
            self.status = f'记录 {len([r for r in self.record.rows if r["status"] != "planning"])}/{self.record.limit}；'+self.status.split('；')[-1]
        return result

    def close(self):
        super().close()
        self.record.refresh_archives(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=10)
    parser.add_argument('--belt-speed-cm-s', type=float,
        help='本轮相机测试的固定预测带速，单位cm/s；不改现场配置文件')
    args = parser.parse_args(argv)
    from .config import CellConfig
    from .velocity_source import with_fixed_belt_speed, policy
    config = CellConfig.load(args.config)
    if args.belt_speed_cm_s is not None:
        try:
            config = with_fixed_belt_speed(config, args.belt_speed_cm_s/100.)
        except ValueError as error:
            parser.error(str(error))
    record = PlanningTestRecord(args.output, args.attempts, velocity_policy=policy(config))
    import yaml
    (args.output/'source_site_config.yaml').write_bytes(config.source.read_bytes())
    (args.output/'site_config.yaml').write_text(yaml.safe_dump(json_value(config.raw),
        allow_unicode=True, sort_keys=False))
    for key, filename in [('tcp_calibration_file','tcp_calibration.yaml'),
                          ('camera_calibration_file','camera_calibration.yaml')]:
        (args.output/filename).write_bytes(Path(config.raw[key]).read_bytes())
    from . import trajectory_preview
    previous = trajectory_preview.TrajectoryPreview
    trajectory_preview.TrajectoryPreview = lambda config, output: RecordingTrajectoryPreview(config, output, record)
    try:
        from .camera_view import main as camera_main
        camera_args = ['--config',str(args.config),'--trajectory-preview',
            '--output',str(args.output),'--screenshot',str(args.output/'window.png')]
        if args.belt_speed_cm_s is not None:
            camera_args += ['--fixed-belt-speed-m-s', str(args.belt_speed_cm_s/100.)]
        return camera_main(camera_args)
    finally:
        trajectory_preview.TrajectoryPreview = previous
        record.closed = True
        record.save()


if __name__ == '__main__':
    raise SystemExit(main())
