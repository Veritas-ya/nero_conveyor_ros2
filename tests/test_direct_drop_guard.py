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
