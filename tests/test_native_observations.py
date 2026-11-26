"""Native observation admission, provenance and delayed feedback, offline."""
from io import StringIO
from types import SimpleNamespace as NS
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from nero_core.native_observation.geometry import NativeGeometry, Parameters
from nero_core.native_observation.association import MotionAssociation
from nero_core.native_observation.shadow import NativeShadow
from nero_core.two_second.synchronized_reference import native_observation_entry
from nero_core.two_second.reference import EntryNotReady


def cell_frame():
    y,x=140,200
    k=NS(width=x,height=y,fx=450.,fy=450.,cx=100.,cy=70.,
         matrix=np.array([[450.,0.,100.],[0.,450.,70.],[0.,0.,1.]]),distortion=np.zeros(5))
    t=np.diag([1.,-1.,-1.,1.]);t[2,3]=1.
    z=np.ones((y,x));rgb=np.zeros((y,x,3),np.uint8);rgb[:,:,1]=160
    for u,color in [(38,[255,0,0]),(87,[1,1,1]),(135,[255,255,255])]:
        z[55:76,u:u+28]=.95
        rgb[55:76,u:u+28]=color
    f=NS(depth_m=z,depth_intrinsics=k,color_intrinsics=k,camera_to_world_cv=t,rgb=rgb,
         header=NS(clock_epoch=3,sequence=1))
    surface=NS(plane_to_base=np.eye(4),detection_hull_xy=None,hull_xy=np.array(
        [[-.21,-.14],[.21,-.14],[.21,.14],[-.21,.14]]),matches_frame=lambda *a:True,
        normal_base=np.array([0.,0.,1.]))
    cfg=NS(raw={'maximum_target_height_m':.12,'perception_detector_workspace_y_m':[-.7,1.1],
                'velocity_source':'operator_fixed_belt_speed','maximum_observation_age_s':1.},
           canonical_tcp=np.eye(4))
    return cfg,f,surface


def test_non_green_native_components_share_exact_pixels_and_isolate_missing_depth():
    cfg,f,surface=cell_frame()
    det=NativeGeometry(surface,f,cfg)
    observations=det.extract(f,np.eye(4),4)
    assert len(observations)==3
    assert len({i for o in observations for i in o.indices})==sum(len(o.indices) for o in observations)
    f.depth_m[55:76,87:115]=0.
    f.depth_m[58,43]=.70  # One unsupported high pixel.
    result=det.extract(f,np.eye(4),4)
    assert len(result)==2
    for o in result:
        assert len(o.grasps)==2 and np.isfinite(o.points).all()
        assert np.allclose(o.center,o.grasps[0][:3,3])
        assert np.allclose(o.points[:,2],.05)
        assert len(o.points)==len(o.indices)
    f.rgb[:]=[0,160,0]
    assert not det.extract(f,np.eye(4),4)


def test_velocity_association_ignores_order_and_short_occlusion_and_rejects_ambiguous_claim():
    a=MotionAssociation(.03);v=np.array([.37,0.,0.])
    pts=np.array([[0.,.04,0.],[0.,-.04,0.]])
    t=a.update(0,1,10.,pts,v,2.); t[0].frozen=True
    a.update(0,2,10.1,[],v,2.)
    out=a.update(0,3,10.5,(pts+.5*v)[::-1],v,2.)
    assert [x.id for x in out]==[1,0] and out[1].frozen
    split=np.array([pts[0]+.6*v,pts[0]+.6*v+[0.,.002,0.]])
    assert a.update(0,4,10.6,split,v,2.)==[None,None]
    with pytest.raises(ValueError):a.update(0,4,10.6,pts,v,2.)
    assert not a.update(1,1,1.,pts[:1],v,2.)[0].frozen


def test_long_thin_low_part_is_measured_with_length_as_diagnostic():
    cfg,frame,surface=cell_frame()
    frame.rgb[:]=[0,160,0]
    frame.depth_m[:]=1.
    frame.rgb[55:60,40:125]=[240,10,5]
    frame.depth_m[55:60,40:125]=.991
    ob,=NativeGeometry(surface,frame,cfg).extract(frame,np.eye(4),4)
    assert .006 < min(ob.extent_m[:2]) < .012
    assert ob.box['above_diagnostic_length']


def test_planner_entry_for_good_geometry_still_checks_clock_direction_and_passage():
    assert native_observation_entry(10.,-.9,.375,[-.1,.4],10.09,1.)>10.09
    for now in (9.9,11.1):
        with pytest.raises(EntryNotReady):native_observation_entry(10.,-.9,.375,[-.1,.4],now,1.)
    with pytest.raises(EntryNotReady):native_observation_entry(10.,.35,.375,[-.1,.4],10.5,1.)
    with pytest.raises(ValueError):native_observation_entry(10.,-.9,0.,[-.1,.4],10.,1.)


def test_late_feedback_promotes_only_its_original_observation(tmp_path):
    cfg,f,surface=cell_frame()
    det=NativeGeometry(surface,f,cfg);ob=det.extract(f,np.eye(4),4)[0]
    t=time.monotonic()
    states={}
    model=NS(classify_robot_points=lambda points,*a,**kw:NS(removed_mask=np.zeros(len(points),bool)))
    s=NativeShadow(cfg,None,tmp_path,states.get,model)
    s.writer=StringIO();s.detector=det
    track=NS(frozen=True)
    s.pending['orig']=t,ob,f,np.eye(4),track
    s.flush_pending()
    assert 'orig' in s.pending and not s.writer.getvalue()
    valid={'joints':[0.]*7,'capture_state_role':'self_filter_only_not_motion_feedback','capture_joint_pairing':{
        'available':True,'extrapolated':False,'joint_interpolation_stamp':t,
        'bracket_bounds_s':[[t-.001,t+.001]]*7}}
    states[t]=dict(valid,capture_joint_pairing=dict(valid['capture_joint_pairing'],joint_interpolation_stamp=t+.002))
    s.flush_pending()
    assert 'orig' in s.pending
    states[t]=valid;s.flush_pending()
    row=json.loads(s.writer.getvalue())
    assert row['key']=='orig' and row['event']=='paired_geometry_ready' and not row['planning_submitted']
    assert not s.pending
    s.pending['old']=t-2.,ob,f,np.eye(4),track
    s.flush_pending()
    assert s.counts['unpaired_geometry_retained']==1
    assert not track.frozen
