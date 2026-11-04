"""Convert an admitted visual NN candidate to the original full-pick inputs.

No scene certificate or motion command is produced here. All timestamps are
simulation capture-clock seconds; host/device/ROS times must not be mixed in.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import numpy as np

from experiments.graspgenx_sim.continuous import GraspTrackOverride
from experiments.graspgenx_sim.input_integrity import validate_candidate_admission
from experiments.continuous_intercept_sim.saved_network_grasp import SavedNetworkObservationView
from experiments.continuous_intercept_sim.full_preplanned import tcp_prediction
from mujoco_conveyor_sorting.runtime.types import TargetIdentity


class CandidateRejected(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details


@dataclass(frozen=True)
class FullObservation:
    observation: dict
    admission: dict
    target_identity: TargetIdentity
    capture_time_s: float


def prepare_full_observation(*, provenance: dict, candidate: dict, raw_observation: dict,
                             result_identity: TargetIdentity, active_identity: TargetIdentity | None,
                             max_observation_age_s: float, max_result_age_s: float | None):
    """Bind capture-time TCP to current *visual* object, using original adapters.

    The caller must create result_identity at capture and keep the same identity
    until cancellation or target/clock change. Admission uses the existing visual
    freshness/scene/generation check; no guessed speed or new threshold is used.
    The resulting prediction is a planning input, never an executable path.
    """
    if not isinstance(result_identity, TargetIdentity) or active_identity != result_identity:
        raise CandidateRejected('inactive_or_replaced_task')
    track_id = int(provenance['track_id'])
    if result_identity.track_id is not None and result_identity.track_id != track_id:
        raise CandidateRejected('identity_track_mismatch')
    if result_identity.track_generation is not None and str(result_identity.track_generation) != str(provenance.get('track_generation')):
        raise CandidateRejected('identity_generation_mismatch')
    capture = float(provenance['capture_sim_s'])
    if not np.isfinite(capture):
        raise CandidateRejected('invalid_capture_time')
    if (result_identity.first_valid_observed_time is not None
            and result_identity.first_valid_observed_time > capture + 1e-6):
        raise CandidateRejected('capture_precedes_identity')
    now = float(raw_observation['timestamp'])
    matches = [item for item in raw_observation['objects'] if int(item['object_id']) == track_id]
    if len(matches) != 1:
        raise CandidateRejected('missing_or_ambiguous_visual_track')
    prediction = raw_observation['predictions'].get(track_id)
    if prediction is None:
        raise CandidateRejected('missing_visual_prediction')
    live = {**matches[0], 'observed_timestamp': prediction.get('observed_timestamp')}
    admission = validate_candidate_admission(provenance, live, int(raw_observation['scene_epoch']), now,
        max_observation_age_s=max_observation_age_s, max_result_age_s=max_result_age_s)
    if not admission['allowed']:
        raise CandidateRejected('visual_candidate_rejected', admission)
    if 'tcp_pose_world_wxyz' not in candidate:
        raise CandidateRejected('candidate_tcp_pose_missing')
    override = GraspTrackOverride(track_id, candidate['tcp_pose_world_wxyz'],
        provenance['track_position_world_m'], capture, float(candidate['closing_width_m']),
        float(candidate['score']), source='graspgenx')
    # Keep the original object's exposure/current estimates intact so
    # tcp_prediction can reconstruct the independent object-to-TCP transform.
    source = deepcopy(raw_observation)
    source['objects'] = [deepcopy(matches[0])]
    source['predictions'] = {track_id: deepcopy(prediction)}
    source['objects'][0]['position'] = np.asarray(prediction['current_estimated_pose'])[:3].copy()
    view = SavedNetworkObservationView()
    view.overrides = {track_id: override}
    converted = view.decorate_observation(source)
    p = converted['predictions'][track_id]
    p['source_object_observed_pose'] = np.asarray(prediction['observed_pose']).copy()
    p['source_object_current_estimated_pose'] = np.asarray(prediction['current_estimated_pose']).copy()
    converted['predictions'][track_id] = tcp_prediction(p, p['current_ee_grasp_pose'])
    converted['objects'][0]['prediction'] = converted['predictions'][track_id]
    converted['target_identity'] = result_identity
    return FullObservation(converted, admission, result_identity, capture)
