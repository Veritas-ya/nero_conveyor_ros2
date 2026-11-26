"""Matched-lane publication window and direct-native exposure window."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from nero_core.two_second import tracked_single_exposure as native
from nero_core.two_second import vision as vision_module
from nero_core.two_second.config import CellConfig
from nero_core.two_second.fixed_freeze import DepthCandidate
from nero_core.two_second.velocity_source import with_fixed_belt_speed, speed_report
from nero_core.two_second.vision import LiveVision, MeasuredTarget


@pytest.fixture
def cell(request):
    path = Path(__file__).resolve().parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'
    cfg = replace(with_fixed_belt_speed(CellConfig.load(path), .5), direction=np.array([1.,0.,0.]))
    if getattr(request, 'param', None) == 'direct':
        cfg.raw.update(freeze_source='single_exposure_native_depth',depth_separation_enabled=True)
    surface = NS(normal_base=np.array([0.,0.,1.]), plane_to_base=np.eye(4),
                 hull_xy=np.array([[-1.3,-.4],[-1.3,.4],[1.,.4],[1.,-.4]]))
    v = LiveVision(None, {}, cfg, NS(measured_belt_surface=surface), lambda: None)
    v.clock_epoch = 0
    v.progress_snapshot = NS(track_generation=1)
    v.enable_lookahead()
    events = []
    v.event_sink = lambda name, **fields: events.append((name, fields))
    yield v, events
    v.close()


def measurement(v, stamp, distance=.73, *, index=0, native_source=False, seq=10, y=0.):
    center = np.array([v.config.follow_bounds[0]-distance, y, .04])
    mask = np.zeros((4,8), bool)
    mask[:,2*index:2*index+2] = True
    detection = dict(belt_region=dict(plane_to_base=np.eye(4)), mask=mask,
        obb=dict(axes_world_columns=np.eye(3), center_world_m=center, extent_m=[.08,.014,.04]))
    velocity = np.array([.5,0.,0.])
    report = speed_report(v.config, NS(sample_count=2), velocity)
    if native_source:
        report['acquisition'] = 'single_exposure_native_depth_fallback'
    return MeasuredTarget((0,-1 if native_source else 1,1+index), stamp, stamp+.001, 9., 0, seq,
        center, velocity, (np.eye(4),), np.tile(center,(30,1)), detection, {}, report)


@pytest.mark.parametrize(('distance','age','allowed'),[
    (.751,.05,False), (.75,.1,True), (.65,.098,True), (.65,.102,False), (.599,0.,False)])
@pytest.mark.parametrize('backup',[False, True])
def test_publication_window_and_later_planning(cell, monkeypatch, distance, age, allowed, backup):
    v, events = cell
    monkeypatch.setattr(vision_module, 'time', NS(monotonic=lambda:10.+age))
    item = measurement(v,10.,distance,native_source=backup)
    with v.lock:
        publication = v._accept_geometry_locked(0,None if backup else 1,item)
    v._notify_frozen(item, publication)
    assert (v.latest is item) == allowed
    if allowed:
        assert events[-1][1]['published_upstream_distance_m'] == pytest.approx(distance-age*.5)
        assert v.pending_admission(item, 10.7)  # a valid freeze survives crossing the line
    else:
        assert events[-1][0] == 'target_freeze_rejected'


@pytest.mark.parametrize('outcome',['matched','failed','timed_out','woke_late','reset'])
def test_single_owner_if_tracked_and_native_finish_in_either_order(cell,monkeypatch,outcome):
    v,events = cell
    now = [10.1]
    monkeypatch.setattr(vision_module,'time',NS(monotonic=lambda:now[0]))
    monkeypatch.setattr(native,'time',NS(monotonic=lambda:now[0]))
    callbacks=[]
    class Timer:
        def __init__(self, _delay, fn): callbacks.append(fn)
        def start(self): pass
        def cancel(self): pass
    monkeypatch.setattr(native,'Timer',Timer)
    matched = measurement(v,10.)
    fallback = measurement(v,10.,native_source=True)
    job = Future()
    v.geometry_future = job
    job.add_done_callback(v._geometry_completed)
    decision = Future()
    decision.set_result((10., matched.center, .03, job))
    v.single_frame_backup._prefer_match(0,fallback,decision,10.26)
    assert v.latest is None
    if outcome in ('timed_out','woke_late'):
        now[0] = 10.24 if outcome=='timed_out' else 10.27
        callbacks[0]()
    if outcome=='reset':
        v.geometry_generation += 1
    job.set_result(None if outcome=='failed' else (0,1,matched))
    callbacks[0]()
    expected = {'matched':matched,'failed':fallback,'timed_out':fallback,'woke_late':None,'reset':None}
    assert v.latest is expected[outcome]
    assert sum(name=='target_frozen' for name, _ in events) == int(v.latest is not None)


def test_different_frames_keep_close_neighbours_and_suppress_one_piece(cell,monkeypatch):
    v,_=cell
    monkeypatch.setattr(vision_module,'time',NS(monotonic=lambda:10.12))
    first = measurement(v,10.1,native_source=True)
    with v.lock:
        assert v._accept_geometry_locked(0,None,first)
    # A tracked result from an earlier exposure can arrive later.
    duplicate = measurement(v,10.,.78,seq=9)
    assert v.repeats_recent_target(duplicate.center,duplicate.capture_s)
    assert not v.repeats_recent_target(duplicate.center+[0.,.019,0.],duplicate.capture_s)
    v.consume(first.identity)
    assert v.repeats_recent_target(first.center+.05*first.velocity,10.15)


@pytest.mark.parametrize('cell', ['hybrid', 'direct'], indirect=True)
@pytest.mark.parametrize(('merged','valid_feedback','count'),[(False,True,2),(True,True,0),(False,False,0)])
def test_camera_loop_freezes_two_distinct_unmatched_components(cell,monkeypatch,merged,valid_feedback,count):
    v,events=cell
    stamp=time.monotonic()-.01
    examples=[measurement(v,stamp,index=i,y=i*.019) for i in range(2)]
    frame=NS(header=NS(sequence=10,source_time=9.,received_time=stamp+.001,clock_epoch=0))
    candidates=[DepthCandidate(t.center,8,np.flatnonzero(t.detection['mask'])) for t in examples]
    class Locator:
        def __init__(self,*args,**kwargs): pass
        def locate(self,*args,**kw): return candidates.copy(),{}
    monkeypatch.setattr(native,'NativeDepthLocator',Locator)
    monkeypatch.setattr('nero_core.two_second.native_single_pass.SinglePassLocator',Locator)
    def geometry(cfg,frame,d2c,surface,candidate,*args,**kwargs):
        item=examples[next(i for i,c in enumerate(candidates) if c is candidate)]
        det={**item.detection,'object_points':item.points}
        if merged:
            det['mask']=examples[0].detection['mask']|examples[1].detection['mask']
        return det,item.center,item.velocity,item.grasps,1.
    monkeypatch.setattr(native,'frozen_geometry',geometry)
    monkeypatch.setattr(vision_module,'apply_runtime_depth_calibration',lambda f,*a:f)
    monkeypatch.setattr(vision_module,'depth_capture_timing',lambda f:{'depth_sample_monotonic_estimate':stamp})
    v.capture_state_reader=lambda captured: {'joint_stamp':stamp,'pose_stamp':stamp} if valid_feedback else None
    monkeypatch.setattr(vision_module,'mapped_stamp',lambda state,key:state[key+'_stamp'])
    class Camera:
        def get_depth_to_color_cv(self): return np.eye(4)
        def read_latest(self,*,after_sequence,**kwargs):
            if after_sequence is None: return frame
            v.single_frame_backup.future.result(timeout=1.)
            v.stop.set()
            raise RuntimeError('end of offline stream')
    v.camera=Camera()
    if v.native_band_only:
        assert v.meter is None and v.tracked_feedback_worker is None
        monkeypatch.setattr(v,'_schedule_geometry',lambda *a:pytest.fail('direct lane attempted matching'))
    else:
        v.meter=NS(update=lambda *a:None,latest_snapshot=NS(current=False,tracks=(),reason='no_match',track_generation=1))
    v._run()
    assert v.error is None
    frozen=[fields for name,fields in events if name=='target_frozen']
    assert len(frozen)==count
    assert len({f['identity'] for f in frozen})==count
    source = 'single_exposure_native_depth' if v.native_band_only else 'single_exposure_native_depth_fallback'
    assert all(f['freeze_source']==source and
               f['capture_monotonic_s']==stamp and f['published_upstream_distance_m']>=.60 for f in frozen)


@pytest.mark.parametrize('option,completion', [([], False), (['--rgb-cloud-completion'], True),
                                               (['--no-rgb-cloud-completion'], False)])
def test_fixed_entry_selects_direct_native_band_and_depth_separation(capsys,option,completion):
    import json
    from nero_core.two_second.fixed_freeze_pick import main
    cfg = Path(__file__).resolve().parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'
    assert main(['--config',str(cfg),'--belt-speed-cm-s','37.5','--check-config',*option]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['freeze_source'] == 'single_exposure_native_depth'
    assert report['depth_separation_enabled']
    assert report['rgb_cloud_completion_enabled'] is completion
    assert report['velocity_policy']['internal_tracking_kalman_still_runs'] is False
    assert report['tracked_freeze_upstream_limit_m'] == .75
    assert report['tracked_freeze_publish_before_upstream_m'] == .60
    assert report['native_single_freeze_policy'] == 'upstream_exposure'
    assert report['native_single_min_capture_upstream_m'] == .45


@pytest.mark.parametrize('cell',['direct'],indirect=True)
@pytest.mark.parametrize('distance,age,accepted',[
    (.75,.20,True),(.66,.21,True),(.52,.21,True),(.45,.10,True),
    (.449,.01,False),(.751,.20,False),(.69,1.01,False),(.65,-.01,False)])
def test_native_band_uses_exposure_position_and_retains_real_stamp(cell,monkeypatch,distance,age,accepted):
    v,events=cell
    monkeypatch.setattr(vision_module,'time',NS(monotonic=lambda:10.+age))
    item=measurement(v,10.,distance,native_source=True)
    with v.lock: publication=v._accept_geometry_locked(0,None,item)
    v._notify_frozen(item,publication)
    assert (v.latest is item) == accepted
    if accepted:
        assert item.capture_s==10.
        assert events[-1][1]['freeze_gate']=='native_exposure_band'
        assert events[-1][1]['published_upstream_distance_m']==pytest.approx(distance-.5*age)


@pytest.mark.parametrize('cell',['direct'],indirect=True)
def test_native_feedback_deadline_and_planning_use_the_same_exposure_policy(cell):
    from nero_core.two_second.tracked_freeze import pairing_deadline
    from nero_core.two_second.fixed_freeze import capture_admission
    from nero_core.two_second.synchronized_reference import freeze_window_for_speed
    from nero_core.two_second.reference import EntryNotReady
    v,_=cell;cfg=v.config;surface=v.model.measured_belt_surface
    item=measurement(v,10.,.55,native_source=True)
    region={'plane_to_base':surface.plane_to_base,'hull_xy_m':surface.hull_xy}
    assert pairing_deadline(cfg,10.,surface,.75)==11.
    window=freeze_window_for_speed(cfg,.5,belt_region=region)
    assert window['minimum_distance_m']==.45 and window['distance_m']==.75
    assert capture_admission(cfg,10.,item.center,item.velocity,10.3,belt_region=region)>10.3
    with pytest.raises(EntryNotReady):
        capture_admission(cfg,10.,item.center+.12*cfg.direction,item.velocity,10.3,belt_region=region)
