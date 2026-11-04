"""Display-only TCP paths computed from recorded certified joint samples."""
import json
from pathlib import Path
import numpy as np

PHASE_COLORS = {
    'approach': (0.1, 1.0, 0.2, 1.0),
    'descent': (1.0, 0.9, 0.1, 1.0),
    'closing': (1.0, 0.1, 0.8, 1.0),
    'lift': (0.1, 0.55, 1.0, 1.0),
    'follow': (0.0, 0.9, 1.0, 1.0),
    'brake': (1.0, 0.65, 0.0, 1.0),
}


def historical_samples(path):
    """Load recorded certificate samples as display data, never as a command."""
    record = json.loads(Path(path).read_text())
    if record.get('complete_path_certified') is not True:
        raise ValueError('historical record has no complete certified path')
    request_id = record.get('plan_request_id')
    plans = [r for r in record.get('planner_results', [])
             if r.get('full_pick_schedule') and (request_id is None or r.get('request_id') == request_id)]
    if len(plans) != 1:
        raise ValueError('historical path must identify exactly one certified schedule')
    samples = plans[0]['full_pick_schedule'].get('certified_samples', [])
    if len(samples) < 2:
        raise ValueError('historical certificate lacks joint samples')
    return samples


def tcp_path_from_samples(samples, robot, base_to_world, *, max_points=180):
    """Forward-kinematics on a separate model; preserve phase boundaries."""
    transform = np.asarray(base_to_world, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or max_points < 2:
        raise ValueError('invalid world transform or point count')
    if len(samples) < 2:
        raise ValueError('trajectory requires at least two samples')
    indexes = set(np.linspace(0, len(samples)-1, min(max_points, len(samples)), dtype=int))
    indexes.update(i for i in range(1, len(samples)) if samples[i][0] != samples[i-1][0])
    points = []
    previous_time = -np.inf
    for i in sorted(indexes):
        phase, timestamp, joints = samples[i]
        q = np.asarray(joints, dtype=float)
        if phase not in PHASE_COLORS or q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(timestamp) or timestamp < previous_time:
            raise ValueError('invalid certified trajectory sample')
        previous_time = timestamp
        robot.set_state(q, .08)
        position = (transform @ np.r_[robot.tcp()[:3, 3], 1.])[:3]
        points.append((phase, float(timestamp), position))
    return points


def draw_tcp_path(scene, points):
    """Append colored segments without changing physics, ownership, or commands."""
    import mujoco
    count = 0
    for previous, current in zip(points, points[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        a, b = previous[2], current[2]
        if np.linalg.norm(b-a) < 1e-8:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
            np.asarray(PHASE_COLORS[current[0]], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, a, b)
        scene.ngeom += 1
        count += 1
    return count
