"""Shadow full-planner observation from the live Nero7/network simulation.

This bridge never sends actions or treats an observation as a certificate.
The historical controller remains the owner of physical commands.
"""
from copy import deepcopy
from uuid import uuid4
from nero_core.vision_full_bridge import prepare_full_observation, CandidateRejected
from mujoco_conveyor_sorting.runtime.types import TargetIdentity


class LiveFullObservation:
    def __init__(self, *, max_observation_age_s, max_result_age_s):
        self.max_observation_age_s = max_observation_age_s
        self.max_result_age_s = max_result_age_s
        self.session_id = uuid4().hex
        self.identity = None
        self.provenance = None
        self.candidate = None
        self.latest = None
        self.events = []
        self._last_recorded = None

    def accept_network_result(self, provenance, candidate, *, clock_epoch):
        if clock_epoch is None or provenance.get('track_generation') is None:
            self.events.append({'event': 'network_result_rejected', 'reason': 'missing_camera_identity'})
            return False
        if self.identity is not None:
            self.events.append({'event': 'network_result_rejected', 'reason': 'task_already_active'})
            return False
        capture = float(provenance['capture_sim_s'])
        track = int(provenance['track_id'])
        generation = int(provenance['track_generation'])
        self.identity = TargetIdentity(self.session_id, str(track), uuid4().hex, capture,
            int(clock_epoch), track_id=track, track_generation=generation)
        self.provenance, self.candidate = deepcopy(provenance), deepcopy(candidate)
        self.events.append({'event': 'network_bound', 'capture_sim_s': capture,
                            'track_id': track, 'clock_epoch': int(clock_epoch),
                            'track_generation': generation})
        return True

    def update(self, observation):
        if self.identity is None:
            return None
        try:
            value = prepare_full_observation(provenance=self.provenance, candidate=self.candidate,
                raw_observation=observation, result_identity=self.identity,
                active_identity=self.identity, max_observation_age_s=self.max_observation_age_s,
                max_result_age_s=self.max_result_age_s)
        except (CandidateRejected, KeyError, ValueError) as error:
            reason = str(error)
            if self._last_recorded != ('rejected', reason):
                self.events.append({'event': 'visual_input_unavailable', 'reason': reason,
                                    'sim_s': float(observation['timestamp'])})
                self._last_recorded = ('rejected', reason)
            self.latest = None
            return None
        self.latest = value
        p = value.observation['predictions'][self.identity.track_id]
        stamp = float(observation['timestamp'])
        if self._last_recorded is None or self._last_recorded[0] != 'ok' or stamp-self._last_recorded[1] >= .1:
            self.events.append({'event': 'full_observation', 'sim_s': stamp,
                'network_capture_sim_s': value.capture_time_s,
                'visual_observed_sim_s': float(p['observed_timestamp']),
                'visual_velocity_m_s': p['estimated_velocity'].tolist(),
                'current_tcp_pose_world_wxyz': p['current_ee_grasp_pose'].tolist(),
                'object_observed_pose_world_wxyz': p['object_observed_pose'].tolist(),
                'scope': 'planner_input_only_not_certified_or_sent'})
            self._last_recorded = ('ok', stamp)
        return value

    def cancel(self, reason):
        if self.identity is not None:
            self.events.append({'event': 'cancelled', 'reason': str(reason)})
        self.identity = self.provenance = self.candidate = self.latest = None
        self._last_recorded = None
