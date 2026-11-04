import numpy as np
import pytest
from experiments.two_second_grasp.perception_panel import render_panel, visual_speed_world, speed_hud_values


def test_panel_uses_frame_without_mutating_sensor_arrays():
    rgb=np.full((120,160,3),80,np.uint8)
    depth=np.full((120,160),.5,np.float32)
    labels=np.full((120,160),-1,np.int32);labels[80:100,20:40]=7
    frame={'top_rgb':rgb,'top_depth':depth,'top_instance_labels':labels,
           'nero_capture':{'sequence':5,'measurements':{7:{}}}}
    before=(rgb.copy(),depth.copy(),labels.copy())
    out=render_panel(frame,status='waiting',completed=0,requested=5,fixed_speed_m_s=.058)
    assert out.shape==(120,320,3) and out.dtype==np.uint8
    for value, expected in zip((rgb,depth,labels),before):np.testing.assert_array_equal(value,expected)


def test_panel_rejects_mismatched_depth():
    with pytest.raises(ValueError,match='share image dimensions'):
        render_panel({'top_rgb':np.zeros((2,2,3),np.uint8),'top_depth':np.ones((3,2)),
                      'top_instance_labels':np.zeros((2,2))},status='test',completed=0,requested=1)


def test_live_speed_uses_selected_camera_track_and_world_axis_only():
    track=lambda v: {'velocity_measurement': {'speed_valid':True,'motion_source':'nero7_xy_kalman',
                                             'velocity_base_m_s':v}}
    frame={'timestamp':2.,'nero_capture':{'measurements':{4:track([0.,-.06,0.]),8:track([0.,-.12,0.])}}}
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    assert visual_speed_world(frame,rotation) is None  # ambiguous tracks are not silently selected
    live=visual_speed_world(frame,rotation,track_id=4)
    np.testing.assert_allclose(live['velocity_world_m_s'],[.06,0,0])
    lines=speed_hud_values(live,now_s=2.1,frozen_velocity=[.055,0,0],belt_speed=.059,truth_speed=.058)
    assert '6.00' in lines[0] and '5.50' in lines[1] and '5.90' in lines[2] and '5.80' in lines[3]
    assert 'no fresh track' in speed_hud_values(live,now_s=2.3,frozen_velocity=None,belt_speed=.059)[0]
    assert visual_speed_world(frame,rotation,track_id=99) is None
