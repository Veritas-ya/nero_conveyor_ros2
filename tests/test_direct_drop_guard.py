from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.direct_drop_guard import check_guard, load_guard
from nero_core.two_second.runtime_sweep import geometry_config
from nero_core.two_second.taught import TaughtSegment


def test_guard_checks_both_payload_and_gripper_inside_radius():
    class Geom:
        name='finger'
    model=SimpleNamespace(gripper_geometries={0},mesh_vertices={0:np.array([[0.,0.,.2]])},
        data=SimpleNamespace(geom_xmat=np.eye(3).reshape(1,9),geom_xpos=np.zeros((1,3))),
        model=SimpleNamespace(geom=lambda _:Geom()))
    guard={'drop_tcp_xyz_base_m':[0.,0.,0.], 'radius_m':.2,'required_min_z_m':.15,
           'source_sha256':'source'}
    planner=SimpleNamespace(config=SimpleNamespace(raw={'experimental_direct_drop_guard':guard}),
        model=model,payload_local=np.array([[0.,0.,.16]]),stats={})
    tcp=np.eye(4)
    check_guard(planner,tcp,payload=True)
    planner.payload_local[0,2]=.14
    with pytest.raises(PickRejected,match='低于') as error:
        check_guard(planner,tcp,payload=True)
    assert error.value.details['geometry']=='estimated_payload'
    tcp[0,3]=.21
    check_guard(planner,tcp,payload=True)
    tcp[0,3]=0.;planner.payload_local[0,2]=.16;model.mesh_vertices[0][0,2]=.14
    with pytest.raises(PickRejected,match='低于') as error:
        check_guard(planner,tcp,payload=True)
    assert error.value.details['geometry']=='finger'
    check_guard(planner,tcp,payload=False)
    planner.bin_departure_tree = object()
    planner.payload_local = None
    with pytest.raises(PickRejected, match='低于') as error:
        check_guard(planner, tcp, payload=False)
    assert error.value.details['geometry'] == 'finger'


def test_real_nominal_guard_is_bound_to_model_tcp_and_teaching():
    root=Path(__file__).parents[1]
    cfg=CellConfig.load(root/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml')
    segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    guard=load_guard(root/'src/nero_bringup/config/calibration/bin_rim_nominal_20261114.yaml',cfg,segment)
    assert guard['required_min_z_m']==pytest.approx(.15610894293727378)
    assert guard['physical_calibration_verified'] is False
    cfg.raw['experimental_direct_drop_guard']=guard
    assert geometry_config(cfg).raw['experimental_direct_drop_guard']==guard


def test_guard_rejects_wrong_demonstration():
    root=Path(__file__).parents[1]
    cfg=CellConfig.load(root/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml')
    segment=TaughtSegment.load(cfg.raw['taught_return_file'],cfg,'return')
    with pytest.raises(ValueError,match='示教文件不一致'):
        load_guard(root/'src/nero_bringup/config/calibration/bin_rim_nominal_20261114.yaml',cfg,segment)


def test_bin_recovery_lift_tracks_world_z_and_checks_the_previous_part():
    from scipy.spatial import cKDTree
    from nero_core.two_second.bin_recovery import bin_lift_goal
    from nero_core.two_second.direct_drop_guard import lowest_gripper_point
    from nero_core.two_second.planning import MeasuredPlanner
    from nero_core.two_second.prepared_cycle import nominal_state

    config_dir = Path(__file__).parents[1]/'src/nero_bringup/config'
    cfg = CellConfig.load(config_dir/'two_second.hardware.tracked_freeze.yaml')
    segment = TaughtSegment.load(cfg.raw['taught_transfer_file'], cfg, 'transfer')
    cfg.raw['experimental_direct_drop_guard'] = load_guard(
        config_dir/'calibration/bin_rim_nominal_20261114.yaml', cfg, segment)
    p = MeasuredPlanner(cfg)
    p.plane = np.eye(4)
    p.support = p.plane[:3, 2]
    p.support_origin = np.zeros(3)
    p.belt_bounds = np.array([[-1., 1.], [-1., 1.]])
    q = segment.joints[-1]
    width = cfg.settings.open_width_m
    state = nominal_state(p.model, q, width)
    start = p.model.tcp().copy()
    # Top of a synthetic workpiece between the open fingers, at the drop pose.
    p.bin_departure_tree = cKDTree([start[:3, 3] + [0., 0., .03]])
    goal, required = bin_lift_goal(p, state)
    path = p.route(state, stage='recovery_bin_lift', goal_pose=goal, start_joints=q)
    np.testing.assert_allclose(path.q0, q, atol=1e-9)
    np.testing.assert_allclose(path.curve([0., path.duration], 1), 0., atol=1e-8)
    np.testing.assert_allclose(path.curve([0., path.duration], 2), 0., atol=1e-7)
    heights = []
    for t in np.linspace(0., path.duration, 51):
        p.model.set_state(path.position(t), width)
        pose = p.model.tcp()
        np.testing.assert_allclose(pose[:2, 3], start[:2, 3], atol=1e-5)
        np.testing.assert_allclose(pose[:3, :3], start[:3, :3], atol=1e-5)
        heights.append(pose[2, 3])
    assert np.min(np.diff(heights)) >= -1e-7
    assert heights[-1] == pytest.approx(goal[2, 3], abs=1e-6)
    assert lowest_gripper_point(p.model)[0] > required

    # Replacing the part by one intersecting the finger during the same lift
    # must veto delivery even though the IK and the endpoint are feasible.
    p.model.set_state(path.position(path.duration/2), width)
    finger = next(iter(p.model.fingers))
    point = (p.model.mesh_vertices[finger].mean(axis=0) @ p.model.data.geom_xmat[finger].reshape(3, 3).T
             + p.model.data.geom_xpos[finger])
    p.bin_departure_tree = cKDTree([point])
    with pytest.raises(PickRejected, match='上一件释放点云相交'):
        p.route(state, stage='recovery_bin_lift', goal_pose=goal, start_joints=q)
