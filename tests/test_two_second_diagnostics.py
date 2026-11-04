"""Offline diagnostic fault injection. No camera, CAN or robot is opened."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import signal
import sys
from threading import Event
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from nero_core.two_second.diagnostics import VisionProgress, RunJournal, ProgressMonitor
from nero_core.two_second.vision import LiveVision
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7TrackSnapshot, Nero7FrameSnapshot
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, TargetNotReady


def wait_until(predicate):
    deadline = time.monotonic()+2.
    while not predicate():
        assert time.monotonic() < deadline, 'diagnostic writer failed to make progress'
        time.sleep(.002)


def test_transient_blockers_survive_sampling_and_clocks_are_not_double_counted():
    now = [10.]
    progress = VisionProgress(clock=lambda: now[0])
    progress.frame(sequence=7, clock_epoch=0)
    now[0] += .1
    progress.block('waiting_speed', 'speed_invalid', samples=3, span_s=.1)
    progress.block('waiting_speed', 'speed_invalid', samples=4, span_s=.2)
    progress.frame(sequence=1, clock_epoch=1)  # camera clock reset
    now[0] += .2
    progress.block('waiting_entry', 'too_far_upstream', distance_m=.46)
    with progress.measure('detection'):
        now[0] += .05
    progress.finish()
    now[0] += 100.
    result = progress.snapshot()
    assert result['frames_received'] == 2
    assert result['blocker_counts'] == {'speed_invalid': 2, 'too_far_upstream': 1}
    previous = result['last_blocker_by_reason']['speed_invalid']
    assert previous['frame'] == {'sequence': 7, 'clock_epoch': 0}
    assert previous['details']['samples'] == 4
    assert result['last_blocker']['frame']['clock_epoch'] == 1
    assert sum(result['stage_wall_s'].values()) == pytest.approx(.35)
    assert result['vision_wall_s'] == pytest.approx(.35)
    assert result['operation_timings_s']['detection']['total_s'] == pytest.approx(.05)


def test_journal_is_readable_before_exit_and_serializes_numpy(tmp_path):
    journal = RunJournal(tmp_path)
    try:
        assert journal.emit('waiting', velocity=np.array([.1, 0., np.nan]), valid=np.bool_(True))
        wait_until(lambda: journal.snapshot()['written'] == 1)
        row = json.loads(journal.path.read_text())
        assert row['data'] == {'velocity': [.1, 0., None], 'valid': True}
        assert row['monotonic_s'] > 0 and row['wall_time_utc'].endswith('+00:00')
    finally:
        summary = journal.close()
    assert summary['enqueued'] == summary['written'] == 1
    assert not summary['errors'] and summary['dropped'] == 0
    assert not summary['writer_alive']


def test_bounded_queue_drops_and_reports_instead_of_blocking_a_producer(tmp_path):
    journal = RunJournal(tmp_path, capacity=1)
    original = journal.file
    entered, release = Event(), Event()
    class SlowFile:
        def write(self, value):
            entered.set()
            assert release.wait(2.)
            return original.write(value)
        def flush(self): original.flush()
        def close(self): original.close()
    journal.file = SlowFile()
    try:
        assert journal.emit('first')
        assert entered.wait(2.)
        assert journal.emit('second')
        begin = time.monotonic()
        assert not journal.emit('overflow')
        assert time.monotonic()-begin < .1
    finally:
        release.set()
        summary = journal.close()
    assert summary['written'] == 2 and summary['dropped'] == 1


def test_writer_io_error_is_reported_without_throwing_into_execution(tmp_path):
    journal = RunJournal(tmp_path)
    original = journal.file
    class BrokenFile:
        def write(self, value): raise OSError('injected full disk')
        def close(self): original.close()
    journal.file = BrokenFile()
    assert journal.emit('test')
    summary = journal.close()
    assert summary['written'] == 0 and 'injected full disk' in summary['errors'][0]


def vision_fixture(monkeypatch, case):
    module = importlib.import_module('nero_core.two_second.vision')
    captured = time.monotonic()-.01
    if case == 'observation_stale_before_detection': captured -= 2.
    fake_now = [time.monotonic()]
    if case == 'observation_stale_after_detection':
        monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda:fake_now[0]))
    frame = SimpleNamespace(header=SimpleNamespace(sequence=7, clock_epoch=0,
        source_time=12., received_time=captured+.001))
    config = SimpleNamespace(direction=np.array([1.,0.,0.]), follow_bounds=np.array([-.2,.28]),
        canonical_tcp=np.eye(4), raw={'minimum_speed_span_s': .45, 'maximum_observation_age_s':1.,
            'freeze_upstream_distance_m':.35, 'planning_reserve_s':1.5,
            'maximum_measured_speed_m_s':.3, 'maximum_target_height_m':.12})
    track = Nero7TrackSnapshot(1, 1, (-.5,0.,.04), (-.5,0.,.04), (.1,0.,0.), True,
        12.,12.,16,.5,.003,((.001,0.),(0.,.001)),0,200.,True)
    if case == 'no_qualified_tracks': track = replace(track, speed_valid=False, span_s=.1)
    if case == 'too_far_upstream': track = replace(track, observed_position_base=(-.66,0.,.04))
    if case == 'already_at_entry': track = replace(track, observed_position_base=(-.1,0.,.04))
    if case == 'insufficient_planning_time': track = replace(track, observed_position_base=(-.25,0.,.04))
    tracks = () if case == 'no_tracks' else ((track, replace(track, track_id=2))
        if case == 'multiple_qualified_tracks' else (track,))
    peer = Nero7FrameSnapshot(12.,7,0,1,case != 'meter_not_current',tracks,'test_measurement')
    model = SimpleNamespace(measured_belt_surface=SimpleNamespace(normal_base=np.array([0.,0.,1.])))
    state = {'joint_stamp': captured, 'pose_stamp': captured}
    if case == 'feedback_time_missing': state = {}
    if case == 'feedback_frame_skew': state['joint_stamp'] = captured-.3
    vision = LiveVision(None, {}, config, model, lambda: None if case == 'feedback_missing' else state)
    vision.geometry_model=model  # explicit offline model; no model loader or device
    class Camera:
        def __init__(self): self.count = 0
        def get_depth_to_color_cv(self): return np.eye(4)
        def read_latest(self, **kw):
            self.count += 1
            if self.count > 1:
                if vision.geometry_future is not None:
                    vision.geometry_future.result(timeout=2.)
                    vision._finish_geometry()
                vision.stop.set()
                raise RuntimeError('offline input exhausted normally')
            return frame
    vision.camera = Camera()
    vision.meter = SimpleNamespace(latest_snapshot=peer, update=lambda *a: None,
        parameters={'tracker_gate_radius':.05})
    monkeypatch.setattr(module, 'apply_runtime_depth_calibration', lambda frame,*a: frame)
    monkeypatch.setattr(module, 'depth_capture_timing', lambda f:{'depth_sample_monotonic_estimate':captured})
    monkeypatch.setattr(module, 'mapped_stamp', lambda state,key:state.get(key+'_stamp'))
    x,y,z = np.meshgrid([-.51,-.5,-.49],[-.01,0.,.01],[.02,.04,.06])
    points = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    if case == 'detection_track_mismatch': points[:,0] += .1
    def detect(*a, **kw):
        if case == 'observation_stale_after_detection': fake_now[0] += 1.1
        if case == 'detection_no_workpiece':
            raise TargetNotReady('injected no object', reason='no_workpiece', details={'candidate_count':0})
        return {'object_points':points, 'belt_region':{'plane_to_base':np.eye(4)},
                'plane':{'normal_base':np.array([0.,0.,1.])}}
    monkeypatch.setattr(module, 'detect_target', detect)
    if case == 'belt_surface_not_ready':
        from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurfaceNotReady
        vision.meter = None
        def fail(*a, **kw): raise BeltSurfaceNotReady('injected no belt', reason='no_green')
        monkeypatch.setattr(module.BeltSurface, 'from_frame', fail)
    return vision


@pytest.mark.parametrize('case', ['feedback_missing', 'feedback_time_missing', 'feedback_frame_skew',
    'no_tracks', 'meter_not_current', 'no_qualified_tracks', 'multiple_qualified_tracks',
    'too_far_upstream', 'already_at_entry', 'insufficient_planning_time',
    'detection_no_workpiece', 'detection_track_mismatch', 'belt_surface_not_ready',
    'observation_stale_before_detection', 'observation_stale_after_detection'])
def test_real_vision_worker_records_each_skip_before_continuing(monkeypatch, case):
    vision = vision_fixture(monkeypatch, case)
    vision._run()
    assert vision.error is None and vision.latest is None
    snapshot = vision.diagnostics()
    assert snapshot['frames_received'] == 1
    detail=snapshot['grasp_geometry'] if case in ('detection_no_workpiece',
        'detection_track_mismatch','observation_stale_after_detection') else snapshot
    assert detail['blocker_counts'] == {case: 1}
    assert detail['last_blocker']['frame']['sequence'] == 7
    if case == 'no_qualified_tracks':
        reasons = snapshot['tracks'][0]['rejection_reasons']
        assert 'speed_invalid' in reasons and 'speed_span_too_short' in reasons


def test_successful_vision_still_produces_same_exposure_target(monkeypatch):
    vision = vision_fixture(monkeypatch, 'ready')
    vision._run()
    assert vision.error is None
    target = vision.poll()
    assert target.sequence == 7 and target.identity == (0,1,1)
    assert target.speed_report['source'] == 'nero7_xy_kalman'
    assert vision.diagnostics()['target_ready_evaluations'] == 1
    assert not vision.diagnostics()['blocker_counts']


def test_pointcloud_speed_flows_from_same_exposure_to_frozen_target(monkeypatch):
    vision=vision_fixture(monkeypatch,'ready')
    vision.config.raw['velocity_source']='pointcloud_xy_kalman'
    peer=vision.meter.latest_snapshot
    track=replace(peer.tracks[0],velocity_base_m_s=(.075,0.,0.),
                  legacy_velocity_base_m_s=(.06,0.,0.),
                  speed_position_base_m=(-.49,0.,.04),velocity_source='pointcloud_xy_kalman')
    vision.meter.latest_snapshot=replace(peer,tracks=(track,))
    vision._run();target=vision.poll()
    assert target is not None and target.identity==(0,1,1)
    np.testing.assert_allclose(target.velocity,[.075,0.,0.])
    assert target.speed_report['source']=='pointcloud_xy_kalman'
    assert target.speed_report['legacy_velocity_base_m_s']==(.06,0.,0.)
    assert target.speed_report['speed_position_base_m']==(-.49,0.,.04)


def test_degenerate_visible_points_report_specific_obb_failure():
    from nero_core.two_second.grasp import VisibleOBBError, visible_belt_aligned_obb
    points = np.column_stack((np.zeros(30), np.zeros(30), np.linspace(.01, .1, 30)))
    with pytest.raises(VisibleOBBError, match='cannot define'):
        visible_belt_aligned_obb(points)


def test_degenerate_obb_frame_is_skipped_and_later_frame_can_produce_target(monkeypatch):
    from nero_core.two_second import vision as module
    from nero_core.two_second.grasp import VisibleOBBError
    vision = vision_fixture(monkeypatch, 'ready')
    frame = vision.camera.read_latest()
    peer = vision.meter.latest_snapshot
    state = vision.read_feedback()
    captured = state['joint_stamp']
    original = module.obb_grasps
    def invalid_then_valid(*args):
        if not calls:
            calls.append(True)
            raise VisibleOBBError('visible XY points cannot define an oriented box')
        return original(*args)
    calls = []
    monkeypatch.setattr(module, 'obb_grasps', invalid_then_valid)
    args = (frame, np.eye(4), peer, peer.tracks[0], state, captured, 0,
            vision.model.measured_belt_surface, .05)
    assert vision._prepare_target(*args) is None
    snapshot = vision.geometry_progress.snapshot()
    assert snapshot['blocker_counts']['obb_invalid_visible_points'] == 1
    assert snapshot['last_blocker']['frame']['sequence'] == 7
    next_frame = SimpleNamespace(header=SimpleNamespace(**{**vars(frame.header), 'sequence': 8}))
    result = vision._prepare_target(next_frame, *args[1:])
    assert result is not None and result[2].identity == (0, 1, 1) and result[2].sequence == 8
    vision.geometry_pool.shutdown(wait=True)


def test_obb_does_not_hide_unrelated_configuration_errors(monkeypatch):
    from nero_core.two_second import vision as module
    vision = vision_fixture(monkeypatch, 'ready')
    frame = vision.camera.read_latest()
    peer, state = vision.meter.latest_snapshot, vision.read_feedback()
    def invalid_calibration(*args):
        raise ValueError('invalid calibration')
    monkeypatch.setattr(module, 'obb_grasps', invalid_calibration)
    with pytest.raises(ValueError, match='invalid calibration'):
        vision._prepare_target(frame, np.eye(4), peer, peer.tracks[0], state,
                               state['joint_stamp'], 0, vision.model.measured_belt_surface, .05)
    vision.geometry_pool.shutdown(wait=True)


def test_invalid_camera_exposure_is_discarded_and_old_geometry_invalidated(monkeypatch):
    import nero_core.two_second.vision as module
    vision=vision_fixture(monkeypatch,'ready')
    old_generation=vision.geometry_generation
    vision.latest=SimpleNamespace(identity=(0,1,999))
    def bad_time(_):
        raise PickRejected('injected camera timestamp error',details={'reason':'global_after_system'})
    monkeypatch.setattr(module,'depth_capture_timing',bad_time)
    vision._run()
    assert vision.error is None and vision.latest is None
    assert vision.geometry_generation>old_generation and vision.reset_tracking
    block=vision.diagnostics()['last_blocker']
    assert block['reason']=='invalid_capture_time'
    assert block['details']['timestamp_details']['reason']=='global_after_system'


def test_valid_camera_frame_after_invalid_one_restarts_tracking(monkeypatch):
    import nero_core.two_second.vision as module
    vision=vision_fixture(monkeypatch,'ready')
    first=vision.camera.read_latest()
    second=SimpleNamespace(header=SimpleNamespace(sequence=8,clock_epoch=0,
        source_time=first.header.source_time,received_time=first.header.received_time))
    class Camera:
        count=0
        def get_depth_to_color_cv(self):return np.eye(4)
        def read_latest(self,**kwargs):
            self.count+=1
            if self.count==1:return first
            if self.count==2:return second
            if vision.geometry_future is not None:
                vision.geometry_future.result(timeout=2.);vision._finish_geometry()
            vision.stop.set();raise RuntimeError('offline input exhausted normally')
    vision.camera=Camera()
    vision.meter.latest_snapshot=replace(vision.meter.latest_snapshot,sequence=8)
    resets=[];vision.meter.reset=lambda:resets.append(True)
    def exposure(frame):
        if frame.header.sequence==7:raise PickRejected('invalid exposure')
        return {'depth_sample_monotonic_estimate':frame.header.received_time-.001}
    monkeypatch.setattr(module,'depth_capture_timing',exposure)
    vision._run()
    assert resets==[True] and vision.error is None
    assert vision.poll().sequence==8


def test_invalid_exposure_diagnostics_identify_the_actual_clock_disagreement():
    from nero_core.two_second.measured_rgbd import depth_capture_timing
    frame=SimpleNamespace(header=SimpleNamespace(source_time=12.,received_time=10.1),
        timestamp_metadata={'selected_source':'depth_device',
            'depth':{'device_time_s':12.,'global_time_s':100.,'system_time_s':99.},
            'host_clock_reference':{'unix_s':100.,'monotonic_s':10.,'read_span_s':.001}})
    with pytest.raises(PickRejected) as error:depth_capture_timing(frame)
    assert error.value.details['reason']=='global_after_system'
    assert error.value.details['global_minus_system_s']==1.


def test_camera_failure_preserves_failed_stage_and_reaches_main(monkeypatch):
    vision = vision_fixture(monkeypatch, 'ready')
    def fail(**kw): raise RuntimeError('injected camera lost')
    monkeypatch.setattr(vision.camera, 'read_latest', fail)
    vision._run()
    with pytest.raises(RuntimeError, match='injected camera lost'):
        vision.poll()
    last = vision.diagnostics()['last_blocker']
    assert last['reason'] == 'vision_exception'
    assert last['details']['failed_stage'] == 'waiting_rgbd'


def wire_offline_entry(monkeypatch, mode, cancel_signal=signal.SIGINT):
    """Patch device modules themselves: even --execute cannot open a device."""
    from nero_core.two_second.config import CellConfig
    path = Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    config = CellConfig.load(path)
    calls = []
    class Device:
        def __init__(self, *a):
            calls.append('construct_test_device')
            self.snapshots = 0
            self.returned = False
            self.robot = SimpleNamespace(set_speed_percent=lambda value: calls.append(('speed',value)))
        def configure_joint_command_mode(self, mode): calls.append(('mode',mode))
        def start(self):
            if mode == 'can_down': raise RuntimeError('injected CAN Network is down')
        def close(self): calls.append('close_test_device')
        def snapshot(self):
            self.snapshots += 1
            joints = config.standby.copy()
            if (mode in ('wrong_start','return_send_failure','return_cancel','loaded_start','closed_start')
                    and not self.returned) or (mode == 'start_drift' and self.snapshots > 1):
                joints[6] += .1
            if mode == 'return_cancel' and self.returned:
                signal.raise_signal(cancel_signal)
            return {'joints':joints.tolist(), 'pose':[0.]*6,
                'width_m':0. if mode=='closed_start' else .1,
                'force_n':2. if mode=='loaded_start' else 0.,'joint_stamp':time.monotonic()}
        snapshot_nowait = snapshot
        def move_joints(self, joints):
            assert mode in ('wrong_start','return_send_failure','return_cancel'), 'other test modes must not command motion'
            np.testing.assert_allclose(joints, config.standby)
            calls.append('return_wait_target'); self.returned = True
            if mode == 'return_send_failure': raise RuntimeError('injected return target send failure')
        def stop(self): calls.append('test_stop')
        def prepare_motion(self,*a,**kw): raise AssertionError('test must not prepare motion')
    class Camera:
        def __init__(self,*a): calls.append('construct_test_camera')
        def start(self): pass
        def close(self): calls.append('close_test_camera')
    for name, cls, attribute in (
        ('mujoco_conveyor_sorting.runtime.nero_pick_hardware', Device, 'Hardware'),
        ('mujoco_conveyor_sorting.runtime.backends.orbbec', Camera, 'OrbbecCameraBackend')):
        module = ModuleType(name)
        setattr(module, attribute, cls)
        monkeypatch.setitem(sys.modules, name, module)
    class Planner:
        version = 0
        model = SimpleNamespace(check_feedback=lambda s:None, tcp=lambda:np.eye(4),
                                command_lower=np.full(7,-3.),command_upper=np.full(7,3.))
        def __init__(self,*a): pass
        def diagnostics(self): return {'stage':'injected_offline_planner'}
        def intercept(self,*a,**kw):
            calls.append('test_plan')
            raise PickRejected('injected trajectory rejection', details={'derivative':2})
    planning = importlib.import_module('nero_core.two_second.planning')
    monkeypatch.setattr(planning,'MeasuredPlanner',Planner)
    monkeypatch.setattr(planning,'calibrated_model',lambda *a:None)
    class Vision:
        clock_epoch = 0
        def __init__(self,*a):
            self.progress = VisionProgress()
            self.count = 0
        def start(self): return self
        def close(self): self.progress.finish()
        def consume(self,*a): pass
        def diagnostics(self): return self.progress.snapshot()
        def poll(self):
            self.count += 1
            self.progress.frame(sequence=self.count, clock_epoch=0)
            if mode in ('planning_fail','start_drift'):
                now=time.monotonic()
                return SimpleNamespace(identity=(0,1,1),capture_s=now-.001,received_s=now,device_s=12.,
                    epoch=0,sequence=1,speed_report={'source':'test_double'},grasps=(np.eye(4),))
            self.progress.block('waiting_qualified_speed','no_tracks')
            if self.count == 3: signal.raise_signal(cancel_signal)
    monkeypatch.setattr(importlib.import_module('nero_core.two_second.vision'),'LiveVision',Vision)
    entry = importlib.import_module('nero_core.two_second.main')
    monkeypatch.setattr(entry,'ProgressMonitor',lambda journal, snapshot:ProgressMonitor(journal,snapshot,interval_s=.002))
    return entry, path, calls


@pytest.mark.parametrize('mode', ['can_down','waiting','planning_fail','wrong_start','start_drift',
                                 'return_send_failure','return_cancel','loaded_start','closed_start'])
def test_entry_saves_startup_cancel_and_planning_failures_without_devices(tmp_path,monkeypatch,mode):
    entry, path, calls = wire_offline_entry(monkeypatch,mode)
    output=tmp_path/mode
    assert entry.main(['--config',str(path),'--execute','--output',str(output)]) == 1
    summary=json.loads((output/'summary.json').read_text())
    rows=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert rows[0]['event']=='run_started' and rows[-1]['event']=='run_finished'
    assert summary['journal']['enqueued']==summary['journal']['written']==len(rows)
    assert not summary['journal']['errors'] and summary['journal']['dropped']==0
    assert not summary['journal']['writer_alive']
    assert 'close_test_device' in calls
    if mode=='can_down':
        assert summary['last_runtime']['phase']=='can_startup'
        assert summary['vision_diagnostics'] is None
        assert 'construct_test_camera' not in calls
        assert any(r['event']=='run_failed' for r in rows)
    elif mode in ('return_send_failure','return_cancel','loaded_start','closed_start'):
        assert summary['last_runtime']['phase']=='startup_readiness'
        assert summary['vision_diagnostics'] is None
        assert 'construct_test_camera' not in calls and 'test_plan' not in calls
        recovery = summary['startup_return_wait']
        if mode in ('return_send_failure','return_cancel'):
            assert recovery['joint_command_attempts']==1 and recovery['electronic_stop_sent']
            assert calls.count('test_stop')==1
            assert recovery['status']==('cancelled' if mode=='return_cancel' else 'stopped')
            assert summary['status']==recovery['status']
        else:
            assert 'return_wait_target' not in calls and 'test_stop' not in calls
            assert summary['details']['reason']==('startup_gripper_force' if mode=='loaded_start' else 'startup_gripper_width')
    elif mode=='wrong_start':
        assert summary['status']=='cancelled'
        assert summary['startup_return_wait']['status']=='wait_reached'
        assert summary['startup_return_wait']['joint_target_commands']==1
        assert summary['startup_return_wait']['connection_reused']
        assert calls.count('construct_test_device')==1
        assert calls.count('return_wait_target')==1
        assert calls.index('return_wait_target') < calls.index('construct_test_camera')
        assert [c for c in calls if isinstance(c,tuple) and c[0]=='mode']==[('mode','js'),('mode','j'),('mode','js')]
        assert 'test_stop' not in calls and 'test_plan' not in calls
    elif mode=='start_drift':
        assert summary['details']['original_details']['reason']=='standby_joint_mismatch'
        assert 'construct_test_camera' in calls and 'test_plan' not in calls
    elif mode=='waiting':
        assert summary['status']=='cancelled' and summary['received_signal']=='SIGINT'
        assert summary['cycles']==[]
        assert summary['vision_diagnostics']['blocker_counts']=={'no_tracks':3}
        assert summary['last_runtime']['phase']=='waiting_for_qualified_target'
        assert any(r['event']=='runtime_progress' for r in rows)
        assert summary['startup_return_wait']['status']=='already_at_wait'
        assert 'return_wait_target' not in calls
    else:
        assert summary['details']['original_error']=='injected trajectory rejection'
        assert summary['error_type']=='RecoveryFailed'
        assert summary['last_runtime']['execution_stage']=='recovery_checking'
        assert any(r['event']=='execution_event' and r['data']['record']['event']=='planning_submitted' for r in rows)
        assert (output/'cycle_001/events.json').exists()


def test_sigterm_also_flushes_waiting_diagnostics(tmp_path,monkeypatch):
    entry,path,_=wire_offline_entry(monkeypatch,'waiting',cancel_signal=signal.SIGTERM)
    output=tmp_path/'sigterm'
    assert entry.main(['--config',str(path),'--execute','--output',str(output)])==1
    result=json.loads((output/'summary.json').read_text())
    assert result['received_signal']=='SIGTERM' and result['status']=='cancelled'
    assert result['vision_diagnostics']['last_blocker']['reason']=='no_tracks'


def test_cancel_after_readiness_never_starts_camera(tmp_path,monkeypatch):
    entry,path,calls=wire_offline_entry(monkeypatch,'waiting')
    original=entry.prepare_start_state
    def readiness(*args,**kwargs):
        result=original(*args,**kwargs)
        signal.raise_signal(signal.SIGINT)
        return result
    monkeypatch.setattr(entry,'prepare_start_state',readiness)
    output=tmp_path/'cancel_before_camera'
    assert entry.main(['--config',str(path),'--execute','--output',str(output)])==1
    assert 'construct_test_camera' not in calls
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='cancelled'


def test_execute_terminal_is_concise_and_full_diagnostics_stay_on_disk(tmp_path,monkeypatch,capsys):
    entry,path,calls=wire_offline_entry(monkeypatch,'planning_fail')
    output=tmp_path/'concise_execution'
    assert entry.main(['--config',str(path),'--execute','--output',str(output)])==1
    lines=capsys.readouterr().out.splitlines()
    assert len(lines)<=15 and all(line.startswith('[') and '{' not in line for line in lines)
    assert any('[轨迹] 正在' in line for line in lines)
    assert not any('[轨迹] 已得到' in line for line in lines)
    assert any('[停止] 恢复状态检查' in line for line in lines)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['details']['original_details']=={'derivative':2}
    assert summary['source_hashes'] and summary['last_runtime']['planner']
    assert summary['journal']['observer_errors']==[]


def test_preview_child_is_passive_and_does_not_add_camera_or_can_connections(tmp_path,monkeypatch):
    entry,path,calls=wire_offline_entry(monkeypatch,'waiting')
    children=[]
    def spawn(argv,**kwargs):
        children.append(argv)
        return SimpleNamespace(pid=12345,poll=lambda:None)
    monkeypatch.setattr(entry.subprocess,'Popen',spawn)
    output=tmp_path/'preview_wiring'
    assert entry.main(['--config',str(path),'--execute','--preview','--output',str(output)])==1
    assert len(children)==1 and '--follow-run' in children[0]
    assert '--execute' not in children[0] and '--read-robot-feedback' not in children[0]
    assert calls.count('construct_test_camera')==calls.count('construct_test_device')==1
    result=json.loads((output/'summary.json').read_text())
    assert result['preview_process_pid']==12345
    assert not result['visualization']['writer_alive'] and not result['visualization']['errors']


def test_active_pick_shares_camera_frame_without_restarting_target_processing(monkeypatch):
    vision=vision_fixture(monkeypatch,'ready')
    vision.active_task=(0,1,1)
    frames=[]
    vision.frame_sink=lambda frame,d2c:frames.append((frame.header.sequence,d2c))
    vision._run()
    assert len(frames)==1 and frames[0][0]==7
    assert vision.latest is None and vision.error is None
    residence=vision.diagnostics()['stage_wall_s']
    assert residence['task_active_camera_only']>0
    assert 'tracking_and_speed' not in residence


def test_saved_wait_pose_failure_reports_actual_joint_differences():
    from nero_core.two_second.execution import validate_start_state
    expected = np.array([2.728,43.334,-.263,98.814,3.253,-4.163,29.269])
    observed = np.array([2.077,51.779,-7.205,86.176,-.420,-2.428,49.118])
    cfg=SimpleNamespace(standby=np.deg2rad(expected))
    with pytest.raises(PickRejected) as rejected:
        validate_start_state(cfg, {'joints':np.deg2rad(observed)})
    details=rejected.value.details
    np.testing.assert_allclose(details['joint_errors_deg'],observed-expected,atol=1e-10)
    assert details['failing_joint_numbers']==list(range(1,8))
    assert details['tolerance_rad']==.005
    assert max(abs(x) for x in details['joint_errors_deg'])==pytest.approx(19.849)


@pytest.mark.parametrize('position_error,force,width,reason', [
    (.005,0.,.1,None), (np.nextafter(.005,np.inf),0.,.1,'standby_joint_mismatch'),
    (0.,.2,.1,'startup_gripper_force'), (0.,0.,.09,'startup_gripper_width')])
def test_shared_start_check_keeps_original_tolerances(position_error,force,width,reason):
    from nero_core.two_second.execution import validate_start_state
    cfg=SimpleNamespace(standby=np.zeros(7),settings=SimpleNamespace(min_force_n=.2,open_width_m=.1))
    q=np.zeros(7);q[0]=position_error
    state={'joints':q,'force_n':force,'width_m':width}
    if reason is None:
        validate_start_state(cfg,state)
    else:
        with pytest.raises(PickRejected) as rejected:
            validate_start_state(cfg,state)
        assert rejected.value.details['reason']==reason
