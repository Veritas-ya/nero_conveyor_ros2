"""Bind an admitted GraspGen-X result to the same measured Nero7 capture."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .input_integrity import validate_candidate_admission
from nero_core.vision_full_bridge import CandidateRejected


def bind_measured_network(*, session, captured, task_id, provenance, candidate,
                          raw_observation, world_to_base, clock_epoch,
                          max_observation_age_s, max_result_age_s):
    """Use real visual identity and timestamps; never read MuJoCo objects.

    ``captured`` must previously have been submitted to the session. The
    original network candidate's zero contact-limited closing command remains
    in its metadata; point-cloud width is only an observed geometry dimension.
    """
    identity = session.identity
    if clock_epoch != identity.clock_epoch or captured.clock_epoch != clock_epoch:
        raise CandidateRejected('camera_clock_changed')
    if (int(provenance['track_id']) != int(task_id) or
        abs(float(provenance['capture_sim_s'])-captured.capture_s) > 1e-6 or
        str(provenance.get('track_generation')) != str(identity.track_generation)):
        raise CandidateRejected('network_capture_mismatch')
    live_items = [x for x in raw_observation.get('objects', []) if int(x['object_id']) == int(task_id)]
    pred = raw_observation.get('predictions', {}).get(int(task_id))
    if len(live_items) != 1 or pred is None:
        raise CandidateRejected('missing_or_ambiguous_visual_track')
    live = {**live_items[0], 'observed_timestamp': pred.get('observed_timestamp')}
    admission = validate_candidate_admission(provenance, live, int(raw_observation['scene_epoch']),
        float(raw_observation['timestamp']), max_observation_age_s=max_observation_age_s,
        max_result_age_s=max_result_age_s)
    if not admission['allowed']:
        raise CandidateRejected('visual_candidate_rejected', admission)
    pose = np.asarray(candidate['tcp_pose_world_wxyz'], dtype=float)
    canonical = np.asarray(candidate['T_world_canonical'], dtype=float)
    transform = np.asarray(world_to_base, dtype=float)
    if (pose.shape != (7,) or canonical.shape != (4,4) or transform.shape != (4,4)
            or not np.isfinite(np.r_[pose,canonical.ravel(),transform.ravel()]).all()):
        raise ValueError('network pose and camera-to-base transform must be finite')
    tcp = np.eye(4)
    tcp[:3,:3] = Rotation.from_quat(pose[3:][[1,2,3,0]]).as_matrix()
    tcp[:3,3] = pose[:3]
    tcp_base = transform @ tcp
    closing_axis_base = transform[:3,:3] @ canonical[:3,0]
    points = np.asarray(captured.detection['object_points'], dtype=float)
    projected = points @ closing_axis_base
    observed_width = float(np.diff(np.quantile(projected, [.01,.99]))[0])
    if not np.isfinite(observed_width) or observed_width <= 0.:
        raise ValueError('network closing axis has no measurable object width')
    source = ('visible_obb_geometry_only'
        if candidate.get('grasp_position_mode') == 'visible_obb_geometry_only' else
        'experimental_visible_obb_center_with_graspgenx_orientation'
        if candidate.get('grasp_position_mode') == 'visible_belt_aligned_obb_center_tcp' else
        'experimental_visible_bbox_center_with_graspgenx_orientation'
        if candidate.get('grasp_position_mode') == 'visible_world_axis_aligned_box_center_tcp'
        else 'graspgenx_diffusion_measured_rendered_depth')
    selected = {**captured.detection, 'grasp': tcp_base, 'width': observed_width,
        'target_geometry_source': source,
        'network_candidate': deepcopy(candidate), 'network_provenance': deepcopy(provenance),
        'network_capture_center_base_m': np.median(points, axis=0),
        'network_capture_grasp_base': tcp_base.copy()}
    bound = session.bind_selected_grasp(selected, identity=identity,
        capture_timestamp=captured.capture_s, capture_sequence=captured.sequence)
    return bound, admission
