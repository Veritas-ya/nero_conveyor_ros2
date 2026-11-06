"""Fixed capture position: late exposure/ID cannot become a later freeze."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nero_core.two_second.reference import EntryNotReady
from nero_core.two_second.synchronized_reference import freeze_window_for_speed,admit_synchronized_observation
from nero_core.two_second.preview import freeze_candidate
from test_two_second_diagnostics import vision_fixture
from test_camera_grasp_preview import fixture as camera_fixture
from test_two_second_hardware import site


def cfg():
    return NS(follow_bounds=np.array([-.2,.38]),raw={
        'freeze_position_upstream_m':.45,'freeze_position_tolerance_m':.02,
        'freeze_upstream_distance_m':.10,'synchronized_adaptive_freeze':True,
        'synchronized_max_freeze_upstream_distance_m':.60,'planning_reserve_mode':'measured'})


@pytest.mark.parametrize('speed', [.1,.435,.5])
def test_fixed_capture_band_does_not_move_with_speed_or_adaptive_window(speed):
    window=freeze_window_for_speed(cfg(),speed)
    assert window['policy']=='fixed_capture_position'
    assert window['distance_m']==.45
    assert window['minimum_distance_m']==pytest.approx(.43)


@pytest.mark.parametrize('distance,reason', [(.451,'too_far_upstream'),(.45,None),(.44,None),
                                          (.43,None),(.429,'past_fixed_freeze_position')])
def test_capture_position_boundaries(distance,reason):
    window=freeze_window_for_speed(cfg(),.435)
    call=lambda:admit_synchronized_observation(10.,-.2-distance,.435,[-.2,.38],10.2,
        window['distance_m'],minimum_upstream_distance_m=window['minimum_distance_m'])
    if reason:
        with pytest.raises(EntryNotReady) as e:call()
        assert e.value.reason==reason
    else:
        assert call()==pytest.approx(10.+distance/.435)


def test_geometry_delivery_keeps_original_capture_position_and_time():
    # The object moves 13 cm while geometry is processed. The original
    # exposure remains in the band; it must not be restamped or reprojected.
    value=admit_synchronized_observation(10.,-.64,.435,[-.2,.38],10.3,.45,
        minimum_upstream_distance_m=.43)
    assert value==pytest.approx(10.+.44/.435)


def test_null_fixed_position_preserves_existing_measured_window():
    config=cfg();config.raw['freeze_position_upstream_m']=None
    window=freeze_window_for_speed(config,.435)
    assert window['distance_m']==.60 and 'minimum_distance_m' not in window


def test_camera_only_preview_obeys_same_capture_band_and_preserves_exposure():
    config,track,peer,detection=camera_fixture()
    config.raw.update(intercept_strategy='synchronized',freeze_position_upstream_m=.31,
        freeze_position_tolerance_m=.02)
    value=freeze_candidate(config,track,peer,detection,10.,10.3,.05)
    assert value['capture_s']==10. and value['freeze_s']==10.3
    np.testing.assert_array_equal(value['track_position_base_m'],track.observed_position_base)
    late=replace(track,observed_position_base=(-.48,0.,.03))
    with pytest.raises(EntryNotReady) as e:
        freeze_candidate(config,late,peer,detection,10.,10.3,.05)
    assert e.value.reason=='past_fixed_freeze_position'


@pytest.mark.parametrize('position,expected', [(-.53,False),(-.51,True),(-.48,False)])
def test_fixed_camera_geometry_uses_in_band_frame_without_waiting_200ms(monkeypatch,position,expected):
    from concurrent.futures import Future
    from nero_core.two_second import measured_rgbd,speed_overlay
    from nero_core.two_second.preview import CameraPreview
    from mujoco_conveyor_sorting.runtime import depth_calibration
    config,track,peer,_=camera_fixture()
    config.raw.update(intercept_strategy='synchronized',freeze_position_upstream_m=.31,
        freeze_position_tolerance_m=.02)
    peer.tracks=(replace(track,observed_position_base=(position,0.,.03)),)
    monkeypatch.setattr(depth_calibration,'apply_runtime_depth_calibration',lambda frame,*a:frame)
    monkeypatch.setattr(measured_rgbd,'depth_capture_timing',lambda frame:{'depth_sample_monotonic_estimate':10.})
    monkeypatch.setattr(speed_overlay,'live_speed_snapshot',lambda *a,**kw:{})
    monkeypatch.setattr(CameraPreview,'scene_cloud',lambda *a:np.empty((0,3)))
    preview=CameraPreview(config,None,clock=lambda:10.05)
    preview.epoch=0;preview.last_geometry_s=10.
    preview.surface=NS(matches_frame=lambda *a:True,normal_base=np.array([0.,0.,1.]))
    preview.meter=NS(latest_snapshot=peer,update=lambda *a,**kw:{'diagnostics':{}},
        parameters={'speed_min_samples':8,'kalman_max_velocity_std':.01})
    calls=[]
    monkeypatch.setattr(preview.pool,'submit',lambda *a,**kw:calls.append(a) or Future())
    try:
        result=preview.process(NS(header=NS(clock_epoch=0,sequence=12,source_time=10.)),np.eye(4))
        assert result['stage']=='tracking'
        assert bool(calls) is expected
    finally:
        preview.close()


def test_late_camera_planning_observation_does_not_consume_a_trial(tmp_path,monkeypatch):
    from concurrent.futures import Future
    from test_camera_planning_test_record import sample
    from nero_core.two_second.planning_test_record import PlanningTestRecord,RecordingTrajectoryPreview
    config,done,track=sample()
    config.raw.update(freeze_position_upstream_m=.45,freeze_position_tolerance_m=.02)
    record=PlanningTestRecord(tmp_path,1)
    preview=RecordingTrajectoryPreview(config,tmp_path/'paths',record)
    calls=[]
    monkeypatch.setattr(preview.pool,'submit',lambda *a,**kw:calls.append(a) or Future())
    try:
        preview.submit(done,.1)
        assert not calls and not record.rows
        track.observed_position_base=np.array([-.64,0.,1.])
        preview.submit(done,.1)
        assert len(calls)==1 and preview.future is not None
    finally:
        preview.close()


def sequence_vision(monkeypatch,positions,*,geometry_delay=0.):
    from nero_core.two_second import vision as module
    v=vision_fixture(monkeypatch,'ready')
    v.config.raw.update(intercept_strategy='synchronized',freeze_position_upstream_m=.30,
        freeze_position_tolerance_m=.02)
    first=v.camera.read_latest();peer=v.meter.latest_snapshot
    captured=v.read_feedback()['joint_stamp'];now=[captured+.06];index=[0]
    monkeypatch.setattr(module,'time',NS(monotonic=lambda:now[0]))
    monkeypatch.setattr(module,'depth_capture_timing',lambda frame:{'depth_sample_monotonic_estimate':frame.capture})
    original_detect=module.detect_target;calls=[]
    def detect(*a,**kw):
        calls.append(1);now[0]+=geometry_delay
        return original_detect(*a,**kw)
    monkeypatch.setattr(module,'detect_target',detect)
    class Camera:
        def get_depth_to_color_cv(self):return np.eye(4)
        def read_latest(self,**kw):
            if v.geometry_future is not None:
                v.geometry_future.result(timeout=2.);v._finish_geometry()
            i=index[0]
            if i==len(positions):
                v.stop.set();raise RuntimeError('offline sequence complete')
            index[0]+=1;capture=captured+i/30;now[0]=max(now[0],capture+.06)
            track=replace(peer.tracks[0],observed_position_base=(positions[i],0.,.04))
            v.meter.latest_snapshot=replace(peer,tracks=(track,))
            return NS(header=NS(**{**vars(first.header),'sequence':7+i,
                'source_time':12.+i/30,'received_time':capture+.001}),capture=capture)
    v.camera=Camera()
    return v,calls


def test_live_worker_waits_before_line_then_freezes_same_in_band_exposure(monkeypatch):
    v,calls=sequence_vision(monkeypatch,[-.51,-.49],geometry_delay=.2)
    v._run()
    assert v.error is None and len(calls)==1
    target=v.poll()
    assert target is not None and target.sequence==8
    assert target.entry_position[0]==-.49
    assert not v.freeze_position_missed


def test_late_id_stays_skipped_even_if_noisy_position_returns_to_band(monkeypatch):
    v,calls=sequence_vision(monkeypatch,[-.47,-.49])
    v._run()
    assert v.error is None and not calls and v.poll() is None
    identity=(0,1,1)
    assert v.freeze_position_missed=={identity} and identity in v.consumed
    assert v.active_task is None
    assert v.diagnostics()['freeze_position_missed_targets']==1
    assert 'past_fixed_freeze_position' in v.diagnostics()['tracks'][0]['rejection_reasons']


@pytest.mark.parametrize('position,tolerance', [(0.,.02),(-.1,.02),(float('nan'),.02),
    (float('inf'),.02),(.02,.02),(.45,0.),(.45,float('inf'))])
def test_invalid_fixed_capture_configuration_rejected(site,position,tolerance):
    import yaml
    from nero_core.two_second.config import CellConfig
    path,data=site
    data.update(intercept_strategy='synchronized',trajectory_profile='smooth',
        prealign_orientation_s=1.2,freeze_position_upstream_m=position,freeze_position_tolerance_m=tolerance)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='freeze_position'):
        CellConfig.load(path)
