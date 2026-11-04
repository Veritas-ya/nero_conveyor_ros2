"""Reproducible simulation inputs; never supply speed/pose to visual control."""
import numpy as np


def bounds(value, name, *, positive=False):
    if value is None:
        return None
    pair = np.asarray(value, float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
        raise ValueError(name+' must be two finite increasing values')
    if positive and (pair[0] <= 0 or pair[1] > 1):
        raise ValueError(name+' must be within (0, 1] m/s')
    return pair


def make_trials(categories, seed, fixed_speed, *, speed_range=None, y_range=None, yaw_range=None):
    speeds = bounds(speed_range, 'belt-speed-range', positive=True)
    lateral = bounds(y_range, 'spawn-y-range')
    yaw = bounds(yaw_range, 'spawn-yaw-range')
    if not np.isfinite(fixed_speed) or not 0 < fixed_speed <= 1:
        raise ValueError('belt speed must be within (0, 1] m/s')
    # Separate streams keep lateral positions independent of speed settings.
    streams = np.random.SeedSequence([int(seed), 20261030]).spawn(3)
    speed_rng, lateral_rng, yaw_rng = (np.random.default_rng(s) for s in streams)
    trials = [{'index':i+1, 'category_id':int(category),
             'belt_set_m_s':float(fixed_speed if speeds is None else speed_rng.uniform(*speeds)),
             'spawn_y_world_m':None if lateral is None else float(lateral_rng.uniform(*lateral))}
            for i, category in enumerate(categories)]
    if yaw is not None:
        for trial in trials:trial['spawn_yaw_rad']=float(yaw_rng.uniform(*yaw))
    return trials


def spawn_trial(env, trial):
    """Only normal environment spawning and belt command; no mid-run teleports."""
    applied = float(env.set_belt_speed(trial['belt_set_m_s']))
    if not np.isclose(applied, trial['belt_set_m_s'], rtol=0, atol=1e-12):
        raise ValueError('random trial belt setting was clipped by the environment')
    kwargs = {'category_id':trial['category_id']}
    if trial['spawn_y_world_m'] is not None:
        kwargs['y'] = trial['spawn_y_world_m']
    yaw = trial.get('spawn_yaw_rad')
    if yaw is not None:
        if not np.isfinite(yaw):
            raise ValueError('spawn yaw must be finite')
        kwargs['yaw'] = float(yaw)
    logical_id = env.spawn_object(**kwargs)
    if logical_id is None:
        raise RuntimeError('random trial could not spawn its requested T-LESS object')
    runtime = next(r for r in env._objects if r.logical_id == logical_id)
    position=np.asarray(env.data.qpos[runtime.qpos_adr:runtime.qpos_adr+3]).copy()
    quat=np.asarray(env.data.qpos[runtime.qpos_adr+3:runtime.qpos_adr+7]).copy()
    observed_yaw=(float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),
                       1-2*(quat[2]**2+quat[3]**2))) if quat.shape == (4,) else None)
    return logical_id, {**trial, 'logical_id':int(logical_id),
        'requested_y_world_m':trial['spawn_y_world_m'], 'spawn_y_world_m':float(position[1]),
        'spawn_sim_s':float(env.data.time), 'belt_at_spawn_m_s':float(env.belt_speed),
        'spawn_position_world_m':position,
        'spawn_orientation_yaw_rad':observed_yaw,
        'source':'simulation_scenario_and_independent_evaluation_only'}
