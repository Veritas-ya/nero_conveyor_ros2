"""Read-only audit of the compiled AGX motor telemetry contract.

This establishes units/transmission/sign, not contact identity or pad force.
No physics step, rendering, contact inspection or hardware operation occurs.
"""
from __future__ import annotations

import numpy as np


def audit_agx_motor_contract(model, controller):
    import mujoco
    actuators = np.asarray(controller.actuator_ids, dtype=int)
    joints = np.asarray(controller.joint_ids, dtype=int)
    opened = np.asarray(controller.open_joint_positions, dtype=float)
    closed = np.asarray(controller.closed_joint_positions, dtype=float)
    checks = {"one_actuator": actuators.shape == (1,),
              "two_slide_jaws": joints.shape == (2,) and all(
                  int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE) for j in joints),
              "opposed_equal_encoder_stroke": opened.shape == closed.shape == (2,)
                  and np.allclose(opened - closed, [.05, -.05], atol=1e-9)}
    act = int(actuators[0]) if actuators.size else None
    symmetry_id = None
    if joints.shape == (2,):
        for index in range(int(model.neq)):
            if (int(model.eq_type[index]) == int(mujoco.mjtEq.mjEQ_JOINT)
                    and int(model.eq_obj1id[index]) == int(joints[1])
                    and int(model.eq_obj2id[index]) == int(joints[0])
                    and bool(model.eq_active0[index])
                    and np.allclose(model.eq_data[index, :5], [0., -1., 0., 0., 0.])):
                symmetry_id = index
                break
    checks["active_q2_equals_minus_q1_constraint"] = symmetry_id is not None
    checks["joint_transmission_to_first_jaw"] = bool(act is not None and joints.size
        and int(model.actuator_trntype[act]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(model.actuator_trnid[act, 0]) == int(joints[0]))
    checks["unit_gear"] = bool(act is not None and np.allclose(
        model.actuator_gear[act], [1., 0., 0., 0., 0., 0.]))
    valid = bool(all(checks.values()))
    sign = float(-np.sign(opened[0] - closed[0])) if opened.size and closed.size else None
    return {
        "verified": valid, "checks": checks,
        "telemetry_contract": {
            "motor_effort_units": "simulated_actuator_force_N",
            "motor_force_mapping": "single_joint_gear1_symmetry_coupled" if valid else "unverified",
            "motor_closing_effort_sign": sign,
        },
        "actuator_names": [model.actuator(int(i)).name for i in actuators],
        "jaw_joint_names": [model.joint(int(i)).name for i in joints],
        "gear": None if act is None else model.actuator_gear[act].tolist(),
        "gain_parameters": None if act is None else model.actuator_gainprm[act].tolist(),
        "bias_parameters": None if act is None else model.actuator_biasprm[act].tolist(),
        "force_range_N": None if act is None else model.actuator_forcerange[act].tolist(),
        "joint_frictionloss_N": model.dof_frictionloss[model.jnt_dofadr[joints]].tolist(),
        "joint_damping_N_s_m": model.dof_damping[model.jnt_dofadr[joints]].tolist(),
        "symmetry_constraint_solref": None if symmetry_id is None else model.eq_solref[symmetry_id].tolist(),
        "ideal_quasistatic_relation": "closing motor effort = left plus right opposing jaw-axis loads",
        "ideal_symmetric_3N_example": "3 N common motor load gives approximately 1.5 N per jaw under balanced aligned static contacts",
        "normal_force_not_measured": True,
        "not_guaranteed": ["true bilateral contact", "target object identity",
            "minimum pad normal force", "grip stability", "real AGX motor current-to-force calibration"],
        "mapping_limitations": ["joint friction and damping", "jaw inertia during motion",
            "compliant equality constraint", "contact normal not parallel to jaw axis", "unequal contact loading"],
    }
