"""Vectorized MuJoCo contact inspection must keep the original semantics."""
import numpy as np
import mujoco
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, _first_native_penetrating_contact


def legacy_first(data):
    contacts=tuple((int(c.geom1),int(c.geom2),float(c.dist)) for c in data.contact[:data.ncon])
    return next((contact for contact in contacts if contact[2]<0),None)


def test_native_contact_scan_matches_all_contacts_on_varied_robot_states():
    model=RobotModel();rng=np.random.default_rng(20261031)
    states=[np.zeros(7)] + [rng.uniform(model.command_lower,model.command_upper) for _ in range(100)]
    for q in states:
        model.set_state(q,.1)
        expected=legacy_first(model.data)
        actual=_first_native_penetrating_contact(model.data)
        assert actual==expected


def test_native_contact_scan_preserves_negative_contact_and_order():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="a" type="sphere" size="0.1" pos="0 0 0"/>
      <geom name="b" type="sphere" size="0.1" pos="0.15 0 0"/>
      <body name="moving" pos="0.075 0 0"><freejoint/>
        <geom name="c" type="sphere" size="0.1" mass="1"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert data.ncon>=2 and legacy_first(data) is not None
    assert _first_native_penetrating_contact(data)==legacy_first(data)
    data.qpos[:3]=[2,2,2];mujoco.mj_forward(model,data)
    assert legacy_first(data) is None and _first_native_penetrating_contact(data) is None
