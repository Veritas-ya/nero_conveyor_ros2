from types import SimpleNamespace
import numpy as np
import pytest
from test_workpiece_admission import fixture
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.workpiece_admission import MeasuredWorkpieceAdmission


def test_corrected_speed_is_bound_to_current_track_and_never_reuses_stale_points():
    surface,camera,T,depth,mask,frame,_=fixture()
    intr=SimpleNamespace(matrix=camera.K,width=160,height=120,distortion=(0.,)*5)
    frame.depth_m=depth;frame.depth_intrinsics=frame.color_intrinsics=intr;frame.camera_to_world_cv=T
    meter=Nero7BeltVelocityMeter(surface,belt_direction_base=[1.,0.],
                                  maximum_measured_speed_m_s=.3,pointcloud_speed=True)
    meter._geometry(frame,np.eye(4));meter._epoch=0
    for i in range(25):
        stamp=1.+i*.04
        detection={'centroid_base':np.array([.04*(stamp-1),0.,.04]),
                   'speed_position_base':np.array([.06*(stamp-1),0.,.04]),
                   'speed_position_point_count':200,'u':80.,'v':60.,
                   'height':.04,'belt_z':0.,'width_px':20.,'area_px':400.}
        meter._tracker.update([detection],stamp)
        meter._point_speed.update(meter._tracker.tracks,[detection],stamp)
    report=meter._report(stamp,current=True,sequence=25);track=meter.latest_snapshot.tracks[0]
    assert report['motion_source']=='pointcloud_xy_kalman' and track.speed_valid
    assert track.velocity_base_m_s[0]==pytest.approx(.06,abs=.002)
    assert track.legacy_velocity_base_m_s[0]==pytest.approx(.04,abs=.002)
    assert track.speed_position_base_m[0]==pytest.approx(.06*(stamp-1))
    # No point from this exposure: the old corrected speed cannot be frozen.
    stamp+=.04;detection={**detection,'centroid_base':np.array([.04*(stamp-1),0.,.04])}
    detection.pop('speed_position_base')
    meter._tracker.update([detection],stamp)
    meter._point_speed.update(meter._tracker.tracks,[detection],stamp)
    meter._report(stamp,current=True,sequence=26)
    assert not meter.latest_snapshot.tracks[0].speed_valid
    assert meter.latest_snapshot.tracks[0].velocity_base_m_s is None
    meter.reset()
    assert meter._point_speed is None and meter.latest_snapshot is None


def test_sensor_to_meter_path_uses_current_component_and_qualifies_speed(monkeypatch):
    surface,camera,T,_,_,frame,_=fixture()
    intr=SimpleNamespace(matrix=camera.K,width=160,height=120,distortion=(0.,)*5)
    frame.depth_intrinsics=frame.color_intrinsics=intr;frame.camera_to_world_cv=T
    meter=Nero7BeltVelocityMeter(surface,belt_direction_base=[1.,0.],
                                  maximum_measured_speed_m_s=.3,pointcloud_speed=True)
    admission=MeasuredWorkpieceAdmission(surface,.12,include_speed_position=True)
    state={}
    monkeypatch.setattr(meter,'_detect',lambda *args:(state['detections'],state['mask']))
    monkeypatch.setattr(meter,'_registered_depth',lambda *args:frame.depth_m)
    for i in range(25):
        # Integer RGB pixels move slowly enough to remain one visual track.
        x=55+i//4; y=45; w=h=25
        frame.depth_m=np.ones((120,160),np.float32);frame.depth_m[y:y+h,x:x+w]=.96
        frame.rgb[:]=[20,140,20];frame.rgb[y:y+h,x:x+w]=[150,150,150]
        mask=np.zeros((120,160),np.uint8);mask[y:y+h,x:x+w]=1
        u,v=x+12,y+12;point=T[:3,:3]@(camera.rays([u],[v])[0]*.96)+T[:3,3]
        state.update(mask=mask,detections=[{'u':u,'v':v,'centroid_base':point,
            'height':.04,'belt_z':0.,'width_px':25.,'area_px':625.}])
        frame.header=SimpleNamespace(source_time=1.+i*.04,sequence=i,clock_epoch=0)
        report=meter.update(frame,np.eye(4),candidate_filter=admission.filter)
    track=meter.latest_snapshot.tracks[0]
    assert report['motion_source']=='pointcloud_xy_kalman'
    assert track.speed_valid and track.velocity_base_m_s[0]>0
    assert track.speed_position_base_m is not None
    assert track.velocity_source=='pointcloud_xy_kalman'
    previous_generation=meter.latest_snapshot.track_generation
    # A camera clock reset cannot carry a previously qualified speed forward.
    frame.header=SimpleNamespace(source_time=.1,sequence=0,clock_epoch=1)
    reset_report=meter.update(frame,np.eye(4),candidate_filter=admission.filter)
    assert reset_report['reason']=='camera_clock_reset'
    assert meter.latest_snapshot.track_generation>previous_generation
    assert all(not t.speed_valid for t in meter.latest_snapshot.tracks)
