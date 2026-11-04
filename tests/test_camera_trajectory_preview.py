"""Camera-only route preview retains exposure identity and sends no commands."""
import json
from types import SimpleNamespace
import numpy as np
from test_collision_visualization import fixture
from nero_core.two_second import trajectory_preview as module


def test_preview_stores_planned_fk_and_same_exposure_without_hardware(tmp_path, monkeypatch):
    _, frame = fixture()
    points = np.tile([[.0,.0,1.]], (24,1))
    cfg = SimpleNamespace(standby=np.zeros(7), settings=SimpleNamespace(open_width_m=.1), canonical_tcp=np.eye(4),
                          raw={'command_advance_s':.1})
    stamp = [10.]
    def clock():
        stamp[0] += .001
        return stamp[0]
    monkeypatch.setattr(module.time, 'monotonic', clock)
    constructed = []
    class Model:
        def set_state(self, q, width): self.q=np.asarray(q)
        def flange(self): return np.eye(4)
        def tcp(self):
            t=np.eye(4);t[0,3]=self.q[0];return t
    class Path:
        duration=1.;planning_s=.2
        def position(self,t): return np.array([t,0,0,0,0,0,0])
    class Planner:
        def __init__(self,config,model,clock):
            self.clock=clock; self.model=model; constructed.append(self)
        def intercept(self,target,state,*,cancelled):
            assert self.clock() >= target.capture_s
            before = self.clock()
            assert self.clock() > before
            assert not cancelled()
            return SimpleNamespace(path=Path(),selected_grasp=np.eye(4),failures=(),timings_s={},
                reference=SimpleNamespace(contact_xyz=np.array([1.,0.,1.]),contact_s=2.,start_s=self.clock()+.102))
    monkeypatch.setattr(module,'calibrated_model',lambda config:Model())
    monkeypatch.setattr(module,'MeasuredPlanner',Planner)
    monkeypatch.setattr(module,'obb_grasps',lambda *a:([np.eye(4)],{}))
    monkeypatch.setattr(module,'prediction_velocity',lambda *a:np.array([.1,0,0]))
    monkeypatch.setattr(module,'speed_report',lambda *a:{'source':'test'})
    monkeypatch.setattr(module,'MeasuredTarget',lambda *a:SimpleNamespace(capture_s=a[1],speed_report=a[12]))
    peer=SimpleNamespace(clock_epoch=0,track_generation=1)
    track=SimpleNamespace(track_id=2,observed_position_base=np.array([0.,0.,1.]),velocity_base_m_s=[.1,0,0])
    d={'object_points':points,'belt_region':{'plane_to_base':np.eye(4)},'plane':{'normal_base':[0,0,1]}}
    done={'detection':d,'geometry':{'frame':frame,'d2c':np.eye(4)},'capture':1.,'peer':peer}
    worker=module.TrajectoryPreview(cfg,tmp_path)
    try:
        result=worker._calculate(done,track,.1,worker.generation)
        worker.flush_archives()
        second=worker._calculate(done,track,.1,worker.generation)
        worker.flush_archives()
        assert len(constructed) == 1
        assert constructed[0].measured_post_planning_s > 0
    finally:worker.close()
    report=json.loads((tmp_path/'path_epoch_000_frame_000000007.json').read_text())
    assert report['can_connections']==report['robot_commands']==0
    assert report['not_authorized_for_hardware_execution'] and report['frame_sequence']==7
    assert report['initial_state']=='assumed_configured_standby_not_measured'
    assert report['timing'].startswith('real_monotonic_')
    assert report['planning_wall_s'] > 0
    assert report['actual_remaining_send_lead_s'] < report['planning_remaining_send_lead_s']
    assert report['planning_remaining_send_lead_s'] > 0 > report['actual_remaining_send_lead_s']
    assert '错过发送截止' in second['status']
    assert report['actual_remaining_send_lead_s'] == report['actual_send_deadline_monotonic_s']-report['result_ready_monotonic_s']
    with np.load(result['file']+'.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['frame_rgb'],frame.rgb)
        np.testing.assert_allclose(saved['tcp_xyz'][[0,-1],0],[0,1])
        np.testing.assert_allclose(saved['predicted_object_points'][0],[.1,0.,1.])
        assert saved['joints'].shape==(21,7)
