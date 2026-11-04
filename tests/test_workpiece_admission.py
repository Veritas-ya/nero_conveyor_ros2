"""Synthetic depth cases exercise admission before IDs; no physical claims."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.workpiece_admission import MeasuredWorkpieceAdmission
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from vendor.nero7_speed_reference.nero7_perception.geometry import CameraGeometry
from vendor.nero7_speed_reference.nero7_perception.detection import load_detection_params
from pathlib import Path


def fixture():
    plane=np.eye(4)
    hull=np.array([[-.35,-.25],[.35,-.25],[.35,.25],[-.35,.25]],np.float32)
    surface=BeltSurface(plane,np.array([[-.35,.35],[-.25,.25]]),hull,{})
    camera=CameraGeometry(np.array([[200.,0,80],[0,200.,60],[0,0,1.]]),160,120,'rgb')
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    depth=np.ones((120,160),np.float32);mask=np.zeros(depth.shape,np.uint8)
    rgb=np.empty((120,160,3),np.uint8);rgb[:]=[20,140,20]
    frame=SimpleNamespace(rgb=rgb)
    params=load_detection_params(Path(__file__).parents[1]/'src/nero_core/vendor/nero7_speed_reference/upstream_perception.yaml')
    return surface,camera,T,depth,mask,frame,params


def component(camera,T,depth,mask,frame,rect,z=.96,green=False):
    x,y,w,h=rect;mask[y:y+h,x:x+w]=1;depth[y:y+h,x:x+w]=z
    frame.rgb[y:y+h,x:x+w]=[20,140,20] if green else [150,150,150]
    u,v=x+(w-1)/2,y+(h-1)/2
    point=camera.rays([u],[v])[0]*z
    return {'u':u,'v':v,'centroid_base':T[:3,:3]@point+T[:3,3],
            'area_px':w*h,'width_px':np.sqrt(w*h),'height':1-z,'belt_z':0.}


def test_real_depth_volume_is_admitted_and_original_measurement_is_preserved():
    s,c,T,z,m,f,p=fixture();d=component(c,T,z,m,f,(65,45,25,25))
    before=z.copy();accepted,r=MeasuredWorkpieceAdmission(s,.12).filter([d],m,z,c,T,f,p)
    assert len(accepted)==1 and accepted[0] is d
    assert r['accepted_candidates']==1 and r['decisions'][0]['height_m']==pytest.approx(.04,abs=1e-6)
    np.testing.assert_array_equal(z,before)


def test_speed_center_uses_the_admitted_same_frame_component_without_changing_tracking_point():
    s,c,T,z,m,f,p=fixture();d=component(c,T,z,m,f,(65,45,25,25))
    accepted,r=MeasuredWorkpieceAdmission(s,.12,include_speed_position=True).filter([d],m,z,c,T,f,p)
    assert len(accepted)==1 and 'speed_position_base' not in d
    np.testing.assert_allclose(accepted[0]['centroid_base'],d['centroid_base'])
    assert accepted[0]['speed_position_point_count']>0
    assert np.isfinite(accepted[0]['speed_position_base']).all()


@pytest.mark.parametrize('rect,z,green',[
    ((65,45,25,25),1.,False),       # gray flat patch/shadow, zero true height
    ((65,45,25,25),.8,False),       # above configured 12 cm ceiling
    ((65,45,25,25),.96,True),       # green belt-shaped depth island
    ((65,45,5,5),.96,False),        # too few valid points
    ((15,45,100,8),.96,False),      # rail-sized footprint
    ((0,45,25,25),.96,False),       # too close to/outside the measured belt edge
])
def test_background_or_unsupported_component_is_rejected_before_ID(rect,z,green):
    s,c,T,depth,m,f,p=fixture();d=component(c,T,depth,m,f,rect,z,green)
    accepted,r=MeasuredWorkpieceAdmission(s,.12).filter([d],m,depth,c,T,f,p)
    assert not accepted and r['rejected_candidates']==1
    assert r['decisions'][0]['id_assigned'] is False


def test_missing_depth_cannot_become_a_workpiece():
    s,c,T,z,m,f,p=fixture();d=component(c,T,z,m,f,(65,45,25,25));z[m>0]=np.nan
    accepted,r=MeasuredWorkpieceAdmission(s,.12).filter([d],m,z,c,T,f,p)
    assert not accepted


def test_existing_robot_self_filter_is_applied_before_tracking():
    s,c,T,z,m,f,p=fixture();d=component(c,T,z,m,f,(65,45,25,25));state=object();calls=[]
    def classify(points,actual,**kwargs):
        assert actual is state;calls.append(len(points))
        return SimpleNamespace(removed_mask=np.ones(len(points),bool))
    accepted,r=MeasuredWorkpieceAdmission(s,.12).filter([d],m,z,c,T,f,p,
        robot_model=SimpleNamespace(classify_robot_points=classify),robot_state=state)
    assert not accepted and calls and r['decisions'][0]['robot_points_removed']==625


def test_only_depth_admitted_candidates_reach_existing_tracker(monkeypatch):
    s,c,T,z,m,f,p=fixture()
    good=component(c,T,z,m,f,(65,35,25,25))
    flat=component(c,T,z,m,f,(20,70,25,25),1.)
    intr=SimpleNamespace(matrix=c.K,width=160,height=120,distortion=(0.,)*5)
    f.depth_m=z;f.depth_intrinsics=f.color_intrinsics=intr;f.camera_to_world_cv=T
    f.header=SimpleNamespace(source_time=1.,sequence=1,clock_epoch=0)
    meter=Nero7BeltVelocityMeter(s,belt_direction_base=[1.,0.])
    meter._geometry(f,np.eye(4));received=[]
    meter._tracker=SimpleNamespace(tracks=[],ambiguous_frames=0,last_ambiguous=None,
        update=lambda ds,stamp:received.extend(ds))
    monkeypatch.setattr(meter,'_detect',lambda *a:([good,flat],m))
    monkeypatch.setattr(meter,'_registered_depth',lambda *a:z)
    gate=MeasuredWorkpieceAdmission(s,.12)
    report=meter.update(f,np.eye(4),candidate_filter=gate.filter)
    assert len(received)==1 and received[0] is good
    assert report['diagnostics']['depth_admission']['rejected_candidates']==1
