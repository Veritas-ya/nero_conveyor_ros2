"""Immutable experiment evidence and explicit failure accounting; no devices."""
from copy import deepcopy
import json
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.interpolate import PPoly

from nero_core.two_second.experiment_archive import ExperimentArchive
from nero_core.two_second.diagnostics import VisionProgress
from nero_core.two_second.trajectory import TimedPath, joint_curve
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from test_collision_visualization import fixture


def target():
    planner, frame = fixture()
    t = planner.target
    t.received_s=1.01; t.device_s=21.; t.center=np.array([0.,0.,1.]);t.entry_position=t.center.copy()
    t.speed_report={'source':'operator_fixed_belt_speed','configured_speed_m_s':.1}
    t.grasps=(np.eye(4),)
    t.detection.update(obb={'center_world_m':[0,0,1], 'axes_world_columns':np.eye(3), 'extent_m':[.02,.03,.04]},
        belt_region={'plane_to_base':np.eye(4)}, scene_points=np.array([[1.,2.,3.]]))
    return t


def load(path):
    with np.load(path, allow_pickle=False) as a:
        return json.loads(str(a['metadata'])), {k:a[k].copy() for k in a.files if k!='metadata'}


def test_each_target_survives_preview_changes_and_buffers_are_owned(tmp_path):
    t=target(); original=t.points.copy(); archive=ExperimentArchive(tmp_path,lambda *a,**k:None)
    archive.offer_target(t); archive.offer_target(t)
    t.points.setflags(write=True);t.detection['frame'].rgb.setflags(write=True)
    t.points[:]=99; t.detection['frame'].rgb[:]=255
    t.identity=(0,2,8);t.sequence=8; archive.offer_target(t)
    result=archive.close(); assert result['complete'] and result['saved']==2
    files=sorted((tmp_path/'experiment').glob('*_target.npz'))
    meta,data=load(files[0]); np.testing.assert_array_equal(data['points'],original)
    assert not np.all(data['frame_rgb']==255)
    assert meta['identity']==[0,1,2] and meta['physical_success'] is None
    assert load(files[1])[0]['identity']==[0,2,8]


def test_replanning_saves_each_input_failure_and_exact_polynomial(tmp_path):
    t=target(); state={'joints':[0.]*7,'width_m':.1}
    curve=joint_curve([0.,.5,1.],np.array([[0.]*7,[.1]*7,[.2]*7]))
    plan=NS(path=TimedPath(curve,t.identity,5,'test',1.,.1),target=t,scene_revision=2,
        selected_grasp=np.eye(4),timings_s={'ik':.1},failures=(),
        reference=NS(capture_s=1.,start_s=2.,duration_s=1.,capture_tcp_world=np.array([1.,2.,3.])))
    error=PickRejected('injected limit',details={'derivative':2})
    calls=[]
    def intercept(target,state,**kwargs):
        calls.append(deepcopy(state))
        if len(calls)==1: raise error
        return plan
    planner=NS(intercept=intercept,clock=lambda:1.2)
    archive=ExperimentArchive(tmp_path,lambda *a,**k:None);archive.bind_planner(planner)
    with pytest.raises(PickRejected) as caught:planner.intercept(t,state)
    assert caught.value is error
    state['joints'][0]=.0068
    assert planner.intercept(t,state) is plan
    state['joints'][0]=99
    assert archive.close()['complete']
    root=tmp_path/'experiment'; inputs=[json.loads(p.read_text()) for p in sorted(root.glob('*_planning_input.json'))]
    assert [d['start_state']['joints'][0] for d in inputs]==[0.,.0068]
    failed=json.loads(next(root.glob('*_planning_failed.json')).read_text())
    assert failed['details']=={'derivative':2} and failed['input_record'].endswith('planning_input')
    meta,data=load(next(root.glob('*_plan.npz')))
    rebuilt=PPoly(data['curve_c'],data['curve_x'])
    for order in range(3):np.testing.assert_allclose(rebuilt(np.linspace(0,1,100),order),curve(np.linspace(0,1,100),order))
    assert meta['reference']['capture_tcp_world']==[1.,2.,3.]


def test_geometry_rejection_retains_exact_frame_seed_surface_and_feedback(tmp_path):
    t=target();frame=t.detection['frame'];progress=VisionProgress()
    def reject(*args):progress.block('geometry','target_size',length=.3);return None
    vision=NS(_prepare_single=reject,d2c=np.eye(4),geometry_progress=progress)
    surface=BeltSurface(np.eye(4),np.array([[-1.,1.]]*2),np.array([[0.,0.],[1.,0.],[0.,1.]]),
        {'normal':[0,0,1]},geometry_fingerprint=(b'not_json',))
    candidate=NS(position=np.zeros(3),point_count=3,seed_pixel_indices=np.array([10,11,12]))
    archive=ExperimentArchive(tmp_path,lambda *a,**k:None);archive.bind_single_exposure(vision)
    assert vision._prepare_single(frame,surface,candidate,t.state,1.,4) is None
    assert archive.close()['complete']
    root=tmp_path/'experiment';meta,data=load(next(root.glob('*_geometry_input.npz')))
    assert meta['identity']==[0,4,7] and meta['capture_state']==t.state
    assert 'geometry_fingerprint' not in meta['surface']
    np.testing.assert_array_equal(data['seed_pixel_indices'],candidate.seed_pixel_indices)
    outcome=json.loads(next(root.glob('*_geometry_result.json')).read_text())
    assert outcome['status']=='rejected_or_cancelled'
    assert outcome['diagnostics']['last_blocker']['reason']=='target_size'


def test_full_queue_reports_gap_and_does_not_wait_for_disk(tmp_path):
    entered,release=Event(),Event();messages=[]
    archive=ExperimentArchive(tmp_path,lambda *a,**k:messages.append((a,k)),capacity=1)
    original=archive._write
    def blocked(*args):entered.set();assert release.wait(3.);original(*args)
    archive._write=blocked
    try:
        archive._offer('test',{'n':1});assert entered.wait(2.)
        archive._offer('test',{'n':2})
        assert archive._offer('test',{'n':3}) is None
    finally:release.set();result=archive.close()
    assert not result['complete'] and result['dropped']==1 and result['saved']==2
    assert messages and result['errors']


def test_disk_error_is_visible_and_never_claims_complete(tmp_path):
    archive=ExperimentArchive(tmp_path,lambda *a,**k:None)
    def fail(*args):raise OSError('injected full disk')
    archive._write=fail;archive._offer('test',{'n':1});result=archive.close()
    assert not result['complete'] and result['saved']==0
    assert 'injected full disk' in result['errors'][0]
