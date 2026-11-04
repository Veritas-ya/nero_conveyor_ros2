"""Position-only evaluation must exactly preserve the certified reference."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.trajectory import TimedPath, joint_curve


def test_position_only_joint_evaluation_matches_full_q_at_knots_and_limits():
    times=np.array([0.,.4,.8,1.2,1.6])
    joints=np.sin(times[:,None]*np.arange(1.,8.)[None,:])*.04
    path=TimedPath(joint_curve(times,joints,zero_end_acceleration=True),(0,1,2),3,'test',1.6,0.)
    for t in [-1.,0.,np.nextafter(.4,0.),.4,np.nextafter(.4,1.),.9,1.6,2.]:
        np.testing.assert_array_equal(path.position(t),path.sample(t)[0])


def test_smooth_reference_position_matches_full_derivatives_at_phase_boundaries():
    q=Rotation.from_euler('xyz',[.2,-.3,.5]).as_quat()[[3,0,1,2]]
    ref=SmoothIntercept(1.,2.,2.,[.1,.2,.03],[.05,-.02,.002],[0.,0.,.2],[1.,0.,0.,0.],q)
    for boundary in (0.,ref.duration_s,ref.follow_end_s,ref.full_duration_s):
        for t in (np.nextafter(boundary,-np.inf),boundary,np.nextafter(boundary,np.inf)):
            actual,quat=ref.sample(t)
            np.testing.assert_array_equal(actual,ref.position_derivatives(t)[0])
            assert np.isfinite(quat).all()


def test_position_only_path_preserves_ownership_of_returned_samples():
    path=TimedPath(lambda t,d=0: np.ones(7)*(t+d),(0,1,2),1,'test',1.,0.)
    p=path.position(.5);p[0]=99.
    assert path.position(.5)[0]==pytest.approx(.5)


@pytest.mark.parametrize('stamp',[1.,1.5,1.9,2.,2.1,3.])
def test_translated_target_collision_and_saved_cloud_match_whole_cloud_oracle(stamp):
    from scipy.spatial import cKDTree
    from test_collision_visualization import fixture
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    planner,_=fixture()
    planner.target_tree=cKDTree(planner.target.points)
    expected=planner.target.points+planner.target.velocity*(stamp-planner.target.capture_s)
    expected_hits=planner.model.inside_geom(expected,0,.004)
    packets=[];planner.collision_sink=packets.append
    if expected_hits.any():
        with pytest.raises(PickRejected,match='工件点云碰撞'):
            planner.valid(np.zeros(7),.1,stamp=stamp)
        np.testing.assert_array_equal(packets[0]['arrays']['checked_points_base_m'],expected)
        np.testing.assert_array_equal(packets[0]['arrays']['hit_points_base_m'],expected[expected_hits])
    else:
        planner.valid(np.zeros(7),.1,stamp=stamp)
        assert not packets


def test_no_nearby_target_skips_full_cloud_materialization(monkeypatch):
    from scipy.spatial import cKDTree
    from test_collision_visualization import fixture
    planner,_=fixture()
    planner.target.points=np.array([[2.,0.,1.],[3.,0.,1.]])
    planner.target_tree=cKDTree(planner.target.points)
    monkeypatch.setattr(planner,'target_points_at',lambda stamp: pytest.fail('no target is near the robot'))
    planner.valid(np.zeros(7),.1,stamp=2.)


def test_terminal_reports_planned_orientation_change_only_after_delivery():
    from nero_core.two_second.console import ConsoleProgress
    output=[];console=ConsoleProgress(write=output.append)
    def send(event,data):
        console.observe({'event':'execution_event','monotonic_s':1.,'data':{'record':{'event':event,'data':data}}})
    start=np.eye(4);target=np.eye(4)
    target[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    send('planning_submitted',{'start_tcp_base':start.tolist()})
    assert not any('计划旋转' in line for line in output)
    send('planning_delivery',{'selected_grasp_tcp_base':target.tolist(),'planning_wall_s':1.})
    assert any('接近过程中计划旋转 90.0°' in line for line in output)


def test_belt_height_broad_phase_keeps_every_nearest_collision_including_edges():
    from scipy.spatial import cKDTree
    from nero_core.two_second.belt_collision_scene import possible_belt_neighbors
    rng=np.random.default_rng(1030)
    rotation=Rotation.from_euler('xyz',[.05,-.03,.8]).as_matrix()
    origin=np.array([-.4,.2,-.1]);normal=rotation[:,2]
    local_scene=rng.uniform([-.2,-.4,-.004999],[.2,.4,.004999],size=(5000,3))
    scene=local_scene@rotation.T+origin
    local_target=rng.uniform([-.2,-.4,-.04],[.2,.4,.04],size=(3000,3))
    points=local_target@rotation.T+origin
    points=np.vstack((points,scene[0]+normal*.007,scene[1]-normal*.007,
                      origin+normal*(.013+2e-13),origin-normal*(.013+2e-13)))
    nearest=cKDTree(scene).query(points)[0]
    potential=possible_belt_neighbors(points,origin,normal,.008)
    assert (nearest<.008).any()
    assert potential[nearest<.008].all()
    assert potential[-2:].all()  # arithmetic boundary uncertainty stays covered


def test_support_hull_separation_never_discards_actual_below_belt_vertices():
    from pathlib import Path
    from scipy.spatial import ConvexHull
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.planning import calibrated_model, _support_hull_separated
    model=calibrated_model(CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'))
    rng=np.random.default_rng(3022)
    for geom in model.gripper_geometries:
        vertices=model.mesh_vertices[geom]
        hull=vertices[ConvexHull(vertices).vertices]
        for _ in range(80):
            rotation=Rotation.random(random_state=rng).as_matrix()
            normal=Rotation.random(random_state=rng).as_matrix()[:,2]
            origin=rng.normal(size=3)
            # Sample both clearly separated and nearly-touching positions.
            min_height=np.min(vertices@(rotation.T@normal))
            offset=float(rng.choice([-.002,-1.001e-6,-1e-6,-.999e-6,0.,.002]))
            position=origin+normal*(offset-min_height)
            separated=_support_hull_separated(hull,rotation,position,origin,normal)
            exact_min=np.min((vertices@rotation.T+position-origin)@normal)
            if separated:
                assert exact_min >= -1e-6
