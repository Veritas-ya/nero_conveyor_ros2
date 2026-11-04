"""Request-independent rigid geometry, evaluated once per exact full qpos.

This object belongs to one isolated planning worker. Geometry facts are cached;
scene/target/policy verdicts are deliberately not. It never advances physics.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import time

import mujoco
import numpy as np


_POSE_FIELDS = ('xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'geom_xpos',
                'geom_xmat', 'site_xpos', 'site_xmat', 'xanchor', 'xaxis')
_JAC_FIELDS = ('subtree_com', 'cdof')


@dataclass
class StateFeatures:
    full_qpos: np.ndarray
    poses: dict
    contacts: tuple | None = None
    jacobian_fields: dict | None = None
    jacobians: dict = field(default_factory=dict)
    singular_values: dict = field(default_factory=dict)


class SharedStateEvaluator:
    def __init__(self, model, data, *, minimal=True, maximum_states=8192):
        self.model, self.data = model, data
        # Flex/plugin position dependencies are not assumed equivalent to a
        # rigid chain. The original full pipeline remains available for them.
        self.cache_enabled = not (model.nflex or model.nplugin
                                  or model.ntendon or model.nmocap)
        self.minimal = bool(minimal and self.cache_enabled)
        self.maximum_states = int(maximum_states)
        self._states = OrderedDict()
        self.current = None
        self.metrics = dict(new_states=0, exact_geometry_reused=0,
            kinematics_calls=0, full_forward_calls=0, collision_calls=0,
            jacobian_calls=0, svd_calls=0, restore_calls=0,
            kinematics_s=0., contacts_s=0., jacobian_s=0., svd_s=0.)

    def prepare(self, full_qpos):
        qpos = np.asarray(full_qpos, dtype=float)
        key = qpos.tobytes()
        state = self._states.get(key) if self.cache_enabled else None
        if state is not None:
            self.metrics['exact_geometry_reused'] += 1
            self._states.move_to_end(key)
            if state is not self.current:
                self.data.qpos[:] = state.full_qpos
                for name, array in state.poses.items():
                    getattr(self.data, name)[:] = array
                if state.jacobian_fields is not None:
                    for name, array in state.jacobian_fields.items():
                        getattr(self.data, name)[:] = array
                self.metrics['restore_calls'] += 1
            self.current = state
            return state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.
        began = time.perf_counter()
        if self.minimal:
            mujoco.mj_kinematics(self.model, self.data)
            self.metrics['kinematics_calls'] += 1
        else:
            mujoco.mj_forward(self.model, self.data)
            self.metrics['full_forward_calls'] += 1
        self.metrics['kinematics_s'] += time.perf_counter()-began
        state = StateFeatures(qpos.copy(), {name: getattr(self.data, name).copy()
                              for name in _POSE_FIELDS})
        if not self.minimal:
            state.contacts = tuple((int(c.geom1), int(c.geom2), float(c.dist))
                                  for c in self.data.contact[:self.data.ncon])
            state.jacobian_fields = {name: getattr(self.data, name).copy() for name in _JAC_FIELDS}
        if self.cache_enabled:
            self._states[key] = state
            if len(self._states) > self.maximum_states:
                self._states.popitem(last=False)
        self.current = state
        self.metrics['new_states'] += 1
        return state

    def contacts(self):
        state = self.current
        if state.contacts is None:
            began = time.perf_counter()
            mujoco.mj_collision(self.model, self.data)
            state.contacts = tuple((int(c.geom1), int(c.geom2), float(c.dist))
                                  for c in self.data.contact[:self.data.ncon])
            self.metrics['contacts_s'] += time.perf_counter()-began
            self.metrics['collision_calls'] += 1
        return state.contacts

    def jacobian(self, site_id, dofs):
        state = self.current
        key = (int(site_id), tuple(int(i) for i in dofs))
        if key not in state.jacobians:
            began = time.perf_counter()
            if state.jacobian_fields is None:
                mujoco.mj_comPos(self.model, self.data)
                state.jacobian_fields = {name: getattr(self.data, name).copy() for name in _JAC_FIELDS}
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, int(site_id))
            state.jacobians[key] = np.vstack((jp[:, dofs], jr[:, dofs]))
            self.metrics['jacobian_calls'] += 1
            self.metrics['jacobian_s'] += time.perf_counter()-began
        return state.jacobians[key]

    def singular_values(self, site_id, dofs, rotation_scale=.30):
        key = (int(site_id), tuple(int(i) for i in dofs), float(rotation_scale))
        state = self.current
        if key not in state.singular_values:
            scaled = self.jacobian(site_id, dofs).copy()
            scaled[3:] *= rotation_scale
            began = time.perf_counter()
            state.singular_values[key] = np.linalg.svd(scaled, compute_uv=False)
            self.metrics['svd_s'] += time.perf_counter()-began
            self.metrics['svd_calls'] += 1
        return state.singular_values[key]

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0. if key.endswith('_s') else 0

    def clear(self):
        self.current = None
        self._states.clear()
