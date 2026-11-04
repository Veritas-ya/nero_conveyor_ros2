"""Offline eye-to-hand diagnosis; imports no robot or camera driver.

F=T_base_flange, C=T_camera_board, X=T_base_camera, Y=T_flange_board.
All translations are metres. The ideal relation is F @ Y == X @ C.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .eye_to_hand import CalibrationSample, Chessboard, detect_chessboard, estimate_target_pose, solve_eye_to_hand
from .geometry import inverse, pose_xyz_rpy


def write_document(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported JSON value {type(item).__name__}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=encode)+"\n", encoding="utf-8")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotation_error_deg(first, second):
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    sine = np.linalg.norm([relative[2, 1]-relative[1, 2],
                          relative[0, 2]-relative[2, 0],
                          relative[1, 0]-relative[0, 1]]) / 2
    cosine = np.clip((np.trace(relative)-1)/2, -1, 1)
    return float(np.degrees(np.arctan2(sine, cosine)))


def pose_to_xyz_rpy(transform):
    """Serialize to the exact metre/radian format consumed by pose_xyz_rpy."""
    rotation = np.asarray(transform)[:3, :3]
    cp = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = np.arctan2(-rotation[2, 0], cp)
    if cp > 1e-10:
        roll, yaw = np.arctan2(rotation[2, 1], rotation[2, 2]), np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll, yaw = 0., np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.r_[np.asarray(transform)[:3, 3], roll, pitch, yaw].tolist()


def pose_error(estimated, truth):
    return {"rotation_error_deg": rotation_error_deg(estimated, truth),
            "translation_error_mm": float(np.linalg.norm(np.asarray(estimated)[:3, 3]
                                                          - np.asarray(truth)[:3, 3])*1000)}


def pair_rows(samples, selected_pairs=None):
    mapping = {str(row['sample_id']): row for row in samples}
    ids = list(mapping)
    pairs = (selected_pairs if selected_pairs is not None else
             [(a, b) for i, a in enumerate(ids) for b in ids[i+1:]])
    result = []
    for first, second in pairs:
        first, second = str(first), str(second)
        a, b = mapping.get(first), mapping.get(second)
        row = {"from_id": first, "to_id": second, "status": "missing_detection",
               "robot_rotation_deg": None, "vision_rotation_deg": None,
               "robot_minus_vision_deg": None, "abs_angle_difference_deg": None}
        if a is not None and b is not None:
            row['robot_rotation_deg'] = rotation_error_deg(a['F'], b['F'])
            if a.get('C_est') is not None and b.get('C_est') is not None:
                row['vision_rotation_deg'] = rotation_error_deg(a['C_est'], b['C_est'])
                row['robot_minus_vision_deg'] = row['robot_rotation_deg']-row['vision_rotation_deg']
                row['abs_angle_difference_deg'] = abs(row['robot_minus_vision_deg'])
                row['status'] = 'ok'
        result.append(row)
    return result


def held_out_point_rows(estimated, truth):
    # Independent workspace points, not board corner points or fit samples.
    points = np.array([[.10, -.30, .15], [.25, -.15, .30], [-.15, -.20, .40],
                       [.35, .10, .20], [-.20, .15, .25], [0., 0., .50],
                       [.30, -.35, .45], [-.10, .05, .10]])
    camera_points = (inverse(truth) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    base_estimated = (np.asarray(estimated) @ np.c_[camera_points, np.ones(len(points))].T).T[:, :3]
    return [{"point_id": f"test_{i:02d}", "base_truth_x_m": float(a[0]),
             "base_truth_y_m": float(a[1]), "base_truth_z_m": float(a[2]),
             "error_mm": float(np.linalg.norm(a-b)*1000)} for i, (a, b) in enumerate(zip(points, base_estimated))]


def solve_samples(rows, solver_config, X_gt):
    samples = [CalibrationSample(row['F'], row['C_est'], row.get('reprojection_rms_px', 0.), row['sample_id'])
               for row in rows if row.get('C_est') is not None]
    result = {"algorithm": dict(solver_config), "valid_sample_count": len(samples),
              "status": "not_solved", "external_error": None, "held_out_points": []}
    try:
        solved = solve_eye_to_hand(samples, **solver_config)
    except ValueError as error:
        result['reason'] = str(error)
        return result
    result.update(status='solved', solution=solved,
                  external_error=pose_error(solved['camera_to_base_cv'], X_gt),
                  held_out_points=held_out_point_rows(solved['camera_to_base_cv'], X_gt))
    return result


def run_phase_a(output, config):
    output = Path(output)
    random = np.random.default_rng(config['seed'])
    X = pose_xyz_rpy(config['phase_a']['X_gt_xyz_rpy_m_rad'])
    Y = pose_xyz_rpy(config['phase_a']['Y_gt_xyz_rpy_m_rad'])
    rows, inputs = [], []
    max_format_error = 0.
    for i in range(config['phase_a']['samples']):
        original = pose_xyz_rpy(np.r_[random.uniform([-.25, -.35, .1], [.35, .15, .5]),
                                     random.uniform(-1.1, 1.1, 3)])
        serialized = pose_to_xyz_rpy(original)
        # This is the same transformation function used by NERO pose capture.
        F = pose_xyz_rpy(serialized)
        max_format_error = max(max_format_error, float(np.max(np.abs(F-original))))
        C = inverse(X) @ original @ Y
        rows.append({"sample_id": f"math_{i:03d}", "F": F, "C_est": C})
        inputs.append({"sample_id": rows[-1]['sample_id'], "source": "synthetic_math_only",
                       "feedback_xyz_rpy_m_rad": serialized, "F": F.tolist(), "C": C.tolist()})
    solved = solve_samples(rows, config['solver'], X)
    pairs = pair_rows(rows)
    max_pair = max(row['abs_angle_difference_deg'] for row in pairs)
    passed = (solved['status'] == 'solved' and
              solved['external_error']['rotation_error_deg'] < 1e-6 and
              solved['external_error']['translation_error_mm'] < 1e-5 and max_pair < 1e-6)
    result = {"phase": "A", "executed": True, "passed": passed,
              "scope": "pure pose mathematics; not physical robot accuracy",
              "notation": "F Y = X C; F=T_base_flange,C=T_camera_board,X=T_base_camera,Y=T_flange_board",
              "X_gt": X.tolist(), "Y_gt": Y.tolist(), "samples": len(rows),
              "feedback_format": "metres/radians, Rz(yaw) Ry(pitch) Rx(roll)",
              "format_roundtrip_max_matrix_error": max_format_error,
              "relative_angle_difference_max_deg": max_pair, "calibration": solved}
    write_document(output/'phase_a'/'inputs.json', inputs)
    write_document(output/'phase_a'/'result.json', result)
    write_rows(output/'phase_a'/'pairs.csv', pairs)
    write_rows(output/'phase_a'/'held_out_points.csv', solved['held_out_points'])
    return result


def perturbed_mount(Y, F, board, case):
    """Equivalent pose-dependent mount rotation, not an elastic mechanics model.

    Rotate about a flange-frame axis passing through the board centre. A fixed
    angle throughout a dataset remains rigid; flex varies with flange attitude.
    """
    Y, F = np.asarray(Y), np.asarray(F)
    axis = np.asarray(case.get('axis_flange', [0, 1, 0]), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError('mount axis must be a nonzero finite flange-frame vector')
    axis /= np.linalg.norm(axis)
    kind, amplitude = case['kind'], float(case.get('amplitude_deg', 0.))
    if not np.isfinite(amplitude):
        raise ValueError('mount amplitude must be finite')
    gravity_base = np.array([0., 0., -1.])
    signal = float(np.dot(F[:3, 0], gravity_base))
    if kind == 'normal':
        angle = 0.
    elif kind == 'constant':
        angle = amplitude
    elif kind == 'flex':
        angle = amplitude*signal
    else:
        raise ValueError('unknown mount kind')
    centre_board = board.object_points.mean(axis=0)
    centre_flange = Y[:3, :3] @ centre_board + Y[:3, 3]
    delta = cv2.Rodrigues(axis*np.deg2rad(angle))[0]
    changed = Y.copy()
    changed[:3, :3] = delta @ Y[:3, :3]
    changed[:3, 3] = centre_flange - changed[:3, :3] @ centre_board
    return changed, {'kind': kind, 'amplitude_setting_deg': amplitude,
                     'applied_angle_deg': angle, 'axis_flange': axis.tolist(),
                     'pivot_flange_m': centre_flange.tolist(),
                     'pivot_board_m': centre_board.tolist(),
                     'orientation_signal': signal,
                     'law': 'theta=amplitude*dot(base_R_flange[:,0],[0,0,-1])' if kind == 'flex' else 'constant angle',
                     'scope': 'equivalent deterministic sensitivity setting, not measured support deformation'}


def image_origin_corners(image_rgb, corners, board):
    """Identify physical corner zero using only a visible red origin mark.

    Ordinary chessboard detection may flip rows/columns. The red mark is outside
    the negative-x/negative-y corner; no camera/board truth selects a PnP pose.
    This is an explicit simulation substitute for checking the real ORIGIN mark.
    """
    pixels = np.asarray(image_rgb)
    mask = ((pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 110) &
            (pixels[:, :, 2] < 110) & (pixels[:, :, 0] > pixels[:, :, 1]+30)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < 3:
        raise ValueError('physical red origin marker is not visible; numbering not established')
    marker = centres[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    grid = np.asarray(corners).reshape(board.rows, board.columns, 2)
    ends = [(0, 0), (0, board.columns-1), (board.rows-1, 0), (board.rows-1, board.columns-1)]
    row, column = min(ends, key=lambda rc: np.linalg.norm(grid[rc]-marker))
    if row:
        grid = grid[::-1, :, :]
    if column:
        grid = grid[:, ::-1, :]
    return grid.reshape(-1, 2).copy(), {'method': 'image-only red physical marker nearest grid endpoint',
                                     'marker_pixel': marker.tolist(),
                                     'flip_rows': bool(row), 'flip_columns': bool(column),
                                     'used_ground_truth': False}


def _visibility(C, board, intrinsics):
    points = board.object_points
    camera_points = points @ C[:3, :3].T+C[:3, 3]
    projected = cv2.projectPoints(points, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3],
                                  np.asarray(intrinsics['matrix']), np.zeros(5))[0].reshape(-1, 2)
    in_view = bool(np.all(camera_points[:, 2] > 0) and np.all(projected >= [0, 0]) and
                   np.all(projected < [intrinsics['width'], intrinsics['height']]))
    centre = camera_points.mean(axis=0)
    front = bool(np.dot(C[:3, 2], centre) < 0)
    return {'all_inner_corners_in_view': in_view, 'printed_front_facing_camera': front,
            'inner_corner_bounds_px': [projected.min(axis=0).tolist(), projected.max(axis=0).tolist()]}


def _finite_summary(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return (None if not len(values) else {'mean': float(values.mean()), 'max': float(values.max()),
                                         'rms': float(np.sqrt(np.mean(values*values)))})


def _image_case(renderer, board, poses, pairs, X, Y, case, config, output, video):
    output = Path(output)
    rows, metrics = [], []
    for index, pose in enumerate(poses):
        F_state = renderer.set_joints(pose['q7_rad'])
        Y_actual, perturbation = perturbed_mount(Y, F_state, board, case)
        renderer.set_board_mount(Y_actual)
        truth = renderer.get_frame_truth()
        if not np.allclose(truth['X'], X, atol=1e-12, rtol=0):
            raise RuntimeError('camera moved within the image dataset')
        image = renderer.render()
        sample_id = pose['sample_id']
        # Feedback and pixels use this exact mj_forward state; no true SDK pose
        # is substituted into the ideal rendered baseline.
        feedback = pose_to_xyz_rpy(truth['F'])
        F = pose_xyz_rpy(feedback)
        size = [image.shape[1], image.shape[0]]
        reference = {'source': 'actual_mujoco_camera_model', 'K': renderer.intrinsics['matrix'],
                     'D': renderer.intrinsics['distortion'], 'image_size': size}
        context = {'image_size': size, 'image_size_source': 'rendered_rgb_pixels',
                   'application_undistorted': False, 'sdk_rgb_rectification': 'not_applicable_mujoco',
                   'calibration_reference': reference}
        limits = renderer.joint_limits_rad
        q = np.asarray(pose['q7_rad'])
        outside = [renderer.joint_names[j] for j in range(7) if q[j] < limits[j, 0] or q[j] > limits[j, 1]]
        row = {'sample_id': sample_id, 'origin': pose['origin'], 'q7_rad': q, 'F': F,
               'F_render_truth': truth['F'], 'feedback_xyz_rpy_m_rad': feedback,
               'C_gt': truth['C'], 'C_est': None, 'Y_actual': truth['Y'],
               'perturbation': perturbation, 'joint_limit_violations': outside,
               'joint_positions_clamped': False, 'detected': False,
               'visibility': _visibility(truth['C'], board, renderer.intrinsics),
               'occlusion': renderer.board_occlusion_check()}
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        axes_image = overlay.copy()
        try:
            raw_audit = {}
            _, _, corners = detect_chessboard(image, board, renderer.intrinsics,
                                               pnp_audit=raw_audit, image_context=context)
            corners, numbering = image_origin_corners(image, corners, board)
            audit = {}
            C, rms = estimate_target_pose(corners, board, renderer.intrinsics,
                                          pnp_audit=audit, image_context=context)
            # Ground truth enters only evaluation, after the existing detector
            # and existing ITERATIVE PnP have completed.
            error = pose_error(C, truth['C'])
            row.update(detected=True, C_est=C, corners=corners, numbering=numbering,
                       pnp_inputs=audit, raw_detector_pnp_inputs=raw_audit,
                       reprojection_rms_px=rms, pnp_error=error,
                       used_for_calibration=rms <= config['solver']['max_reprojection_px'])
            cv2.drawChessboardCorners(overlay, (board.columns, board.rows), corners.astype(np.float32), True)
            point = tuple(np.rint(corners[0]).astype(int))
            cv2.circle(overlay, point, 7, (0, 0, 255), 2)
            cv2.putText(overlay, 'ORIGIN (image marker)', point, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
            cv2.drawFrameAxes(axes_image, np.asarray(renderer.intrinsics['matrix']), np.zeros(5),
                             cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], board.square_size_m*2)
        except (ValueError, cv2.error) as error:
            row['failure_reason'] = str(error)
            row['used_for_calibration'] = False
            cv2.putText(overlay, 'DETECTION/NUMBERING FAILED', (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        paths = {}
        for directory, picture in (('images', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
                                    ('corners', overlay), ('axes', axes_image)):
            path = output/directory/f'{sample_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), picture):
                raise OSError(f'could not save {path}')
            paths[directory] = str(path)
        row['images'] = paths
        caption = overlay.copy()
        title = f"{case['name']} | {pose['origin']} | {sample_id} | mount {perturbation['applied_angle_deg']:+.3f}deg"
        cv2.rectangle(caption, (0, 0), (image.shape[1], 38), (15, 15, 15), -1)
        cv2.putText(caption, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        if video is not None:
            rgb_caption = cv2.cvtColor(caption, cv2.COLOR_BGR2RGB)
            video.append_data(rgb_caption)
            video.append_data(rgb_caption)
        rows.append(row)
        metrics.append({'sample_id': sample_id, 'origin': pose['origin'], 'detected': row['detected'],
                        'used_for_calibration': row['used_for_calibration'],
                        'pnp_rotation_error_deg': row.get('pnp_error', {}).get('rotation_error_deg'),
                        'pnp_translation_error_mm': row.get('pnp_error', {}).get('translation_error_mm'),
                        'reprojection_rms_px': row.get('reprojection_rms_px'),
                        'actual_mount_rotation_deg': perturbation['applied_angle_deg'],
                        'in_view': row['visibility']['all_inner_corners_in_view'],
                        'front_facing': row['visibility']['printed_front_facing_camera'],
                        'occlusion_status': row['occlusion']['status'],
                        'occluded': row['occlusion']['occluded'],
                        'fully_occluded_corner_count': row['occlusion'].get('fully_occluded_corner_count'),
                        'partially_occluded_corner_count': row['occlusion'].get('partially_occluded_corner_count'),
                        'joint_limit_violations': ','.join(outside),
                        'failure_reason': row.get('failure_reason', '')})
        print(f"{case['name']} {pose['origin']} {sample_id}: "
              f"{'detected' if row['detected'] else 'FAILED'}", flush=True)
    # Same quality cutoff as real solve; all failures remain in images/metrics.
    fit_rows = [r for r in rows if r['used_for_calibration']]
    calibration = solve_samples(fit_rows, config['solver'], X)
    pair_result = pair_rows(rows, pairs)
    result = {'case': case, 'executed': True, 'samples_total': len(rows),
              'detected_count': sum(r['detected'] for r in rows),
              'detection_success_rate': sum(r['detected'] for r in rows)/len(rows),
              'valid_calibration_samples': len(fit_rows), 'calibration': calibration,
              'detection_failed_count': sum(not r['detected'] for r in rows),
              'out_of_view_count': sum(not r['visibility']['all_inner_corners_in_view'] for r in rows),
              'occluded_frame_count': sum(r['occlusion']['occluded'] is True for r in rows),
              'occlusion_unknown_frame_count': sum(r['occlusion']['status'] != 'available' for r in rows),
              'pnp_rotation_error_deg': _finite_summary([r.get('pnp_error', {}).get('rotation_error_deg') for r in rows]),
              'pnp_translation_error_mm': _finite_summary([r.get('pnp_error', {}).get('translation_error_mm') for r in rows]),
              'reprojection_rms_px': _finite_summary([r.get('reprojection_rms_px') for r in rows]),
              'relative_angle_difference_deg': _finite_summary([r['abs_angle_difference_deg'] for r in pair_result]),
              'pair_count': len(pair_result),
              'valid_pair_count': sum(p['status'] == 'ok' for p in pair_result),
              'pairs': pair_result,
              'held_out_position_error_mm': _finite_summary([p['error_mm'] for p in calibration['held_out_points']]),
              'actual_mount_angles_deg': [r['perturbation']['applied_angle_deg'] for r in rows],
              'camera_fixed': True, 'board_truth_never_used_for_pnp': True}
    write_document(output/'samples.json', rows)
    write_document(output/'result.json', result)
    write_rows(output/'per_frame.csv', metrics)
    write_rows(output/'pairs.csv', pair_result)
    write_rows(output/'held_out_points.csv', calibration['held_out_points'])
    return result


def _baseline_checks(control, projection, thresholds):
    calibration = control['calibration']
    error = calibration['external_error']
    points = control['held_out_position_error_mm']
    pairs = control['relative_angle_difference_deg']
    return {'camera_projection_passed': bool(projection['passed']),
            'production_solver_accepted': bool(calibration['status'] == 'solved' and calibration['solution']['accepted']),
            'external_rotation_close_to_truth': bool(error and error['rotation_error_deg'] <= thresholds['external_rotation_deg']),
            'external_translation_close_to_truth': bool(error and error['translation_error_mm'] <= thresholds['external_translation_mm']),
            'held_out_points_close_to_truth': bool(points and points['max'] <= thresholds['held_out_max_mm']),
            'relative_rotation_consistent': bool(pairs and pairs['max'] <= thresholds['relative_angle_max_deg'])}


def run_image_diagnosis(output, config, root, *, include_mount_cases=True):
    from .sim_rendering import NeroCalibrationRenderer
    import imageio.v2 as imageio

    output, root = Path(output), Path(root)
    session_path = (root/config['source']['real_session']).resolve()
    source = json.loads(session_path.read_text())
    board = Chessboard(**source['board'])
    if not config['render']['use_focal_from_real_intrinsics'] or np.any(config['render']['distortion']):
        raise ValueError('this initial diagnostic supports real focal lengths with undistorted rendering and D=0 only')
    expected_model = (root/config['source']['robot_model']).resolve()
    actual_model = Path(__file__).resolve().parents[1]/'models/robot.xml'
    if expected_model != actual_model:
        raise ValueError('the configured robot model differs from the audited NERO robot.xml')
    original_intrinsics = json.loads((root/config['source']['real_intrinsics']).resolve().read_text())
    fx, fy = original_intrinsics['camera_matrix'][0][0], original_intrinsics['camera_matrix'][1][1]
    replay = []
    missing = []
    for sample in source['samples']:
        q = sample.get('robot_diagnostics', {}).get('joints_rad')
        if q is None or len(q) != 7:
            missing.append(sample['sample_id'])
            continue
        replay.append({'sample_id': sample['sample_id'], 'q7_rad': q, 'origin': 'real_joint_replay'})
    control = [{'sample_id': f'control_{i:03d}', 'q7_rad': q, 'origin': 'synthetic_control'}
               for i, q in enumerate(config['control']['joint_positions_rad'])]
    if len(control) != config['control']['samples']:
        raise ValueError('control sample count disagrees with the configured joint positions')
    replay_pairs = [(a['sample_id'], b['sample_id']) for i, a in enumerate(replay) for b in replay[i+1:]]
    X, Y = np.asarray(config['simulation_scene']['X_gt']), np.asarray(config['simulation_scene']['Y_gt'])
    baseline_case = {'name': 'C0_rigid', 'kind': 'normal', 'amplitude_deg': 0.,
                     'axis_flange': config['mount_cases']['flex_axis_flange']}
    with NeroCalibrationRenderer(board, config['render']['width'], config['render']['height'], fx=fx, fy=fy) as renderer:
        renderer.set_joints(replay[0]['q7_rad'] if replay else control[0]['q7_rad'])
        renderer.set_board_mount(Y)
        renderer.set_camera(X)  # Once for the complete experiment, including C.
        projection = renderer.camera_projection_check()
        write_document(output/'camera_projection_check.json', projection)
        cv2.imwrite(str(output/'camera_projection_features.png'), cv2.cvtColor(renderer.projection_check_image, cv2.COLOR_RGB2BGR))
        if not projection['passed']:
            raise RuntimeError('render camera projection self-check failed; no image calibration run')
        metadata = renderer.metadata()
        metadata.update(source_session=str(session_path), missing_real_joint_samples=missing,
                        real_robot_feedback_not_used_in_rigid_baseline=True,
                        original_real_intrinsics_background_only=original_intrinsics,
                        simulated_X_gt=X, simulated_Y_gt=Y,
                        source_requested_path='/data/机械臂分拣/mujoco_conveyor_sorting',
                        actual_project_path=str(root),
                        video_is='sequence of kinematic samples, not actuator trajectory tracking',
                        marker_note='red physical origin marker added only for image-based numbering; ordinary checker pattern unchanged')
        write_document(output/'parameters.json', metadata)
        with imageio.get_writer(str(output/'sampling_demo.mp4'), fps=config['render']['video_fps'],
                                codec='libx264', macro_block_size=1, quality=7) as video:
            replay_result = _image_case(renderer, board, replay, replay_pairs, X, Y, baseline_case,
                                         config, output/'phase_b/real_joint_replay', video) if replay else None
            control_result = _image_case(renderer, board, control, None, X, Y, baseline_case,
                                          config, output/'phase_b/synthetic_control', video)
            checks = _baseline_checks(control_result, projection, config['diagnostic_thresholds'])
            baseline_passed = all(checks.values())
            result = {'phase_b': {'executed': True, 'passed': baseline_passed, 'real_replay': replay_result,
                                  'synthetic_control': control_result, 'camera_projection': projection,
                                  'checks': checks, 'diagnostic_thresholds': config['diagnostic_thresholds']},
                      'phase_c': {'executed': False, 'reason': 'not requested or rigid baseline failed'}}
            # These historical matrices are read only for this side-by-side
            # comparison, never passed to the rendered baseline's solver.
            real_rows = [{'sample_id': s['sample_id'], 'F': s['gripper_to_base'],
                          'C_est': s['target_to_camera']} for s in source['samples']]
            real_pairs = pair_rows(real_rows, replay_pairs)
            result['historical_real_pairs'] = real_pairs
            write_document(output/'image_summary.json', result)
            print(f"阶段 B：{'通过' if baseline_passed else '需检查'}；"
                  f"控制组外参误差 {control_result['calibration']['external_error']}；"
                  f"检测 {control_result['detected_count']}/{control_result['samples_total']}", flush=True)
            if not baseline_passed or not include_mount_cases:
                return result
            cases = [{'name': f"C1_constant_{config['mount_cases']['constant_tilt_deg']:g}deg", 'kind': 'constant',
                       'amplitude_deg': config['mount_cases']['constant_tilt_deg'],
                       'axis_flange': config['mount_cases']['flex_axis_flange']}]
            cases += [{'name': f'C2_flex_{amplitude:g}deg', 'kind': 'flex', 'amplitude_deg': amplitude,
                       'axis_flange': config['mount_cases']['flex_axis_flange']}
                      for amplitude in config['mount_cases']['flex_amplitudes_deg']]
            case_results = {'C0_rigid': {'real_replay': replay_result, 'synthetic_control': control_result,
                                        'source': 'exactly the stage B outputs; not separately resampled'}}
            for case in cases:
                folder = output/'phase_c'/case['name']
                rr = _image_case(renderer, board, replay, replay_pairs, X, Y, case, config,
                                  folder/'real_joint_replay', video) if replay else None
                cc = _image_case(renderer, board, control, None, X, Y, case, config,
                                  folder/'synthetic_control', video)
                case_results[case['name']] = {'real_replay': rr, 'synthetic_control': cc}
                write_document(output/'phase_c/results.json', case_results)
            result['phase_c'] = {'executed': True, 'cases': case_results,
                                 'interpretation': 'sensitivity settings are not measured real mounting deformation'}
            write_document(output/'image_summary.json', result)
            return result
