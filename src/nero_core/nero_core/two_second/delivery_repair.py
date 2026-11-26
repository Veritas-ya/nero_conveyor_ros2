"""Repair an expired, unsent intercept using its scene and joint-curve seeds.

Only a future schedule is produced. No device calls, historical samples, or
cross-object cache reuse occur here. The executor still owns final admission.
"""
from dataclasses import replace
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .precise_ik import PrecisePoseSolver
from .synchronized_reference import SynchronizedIntercept
from .trajectory import certify_curve, sampling_times, extrema
from .phase_timing import require_reference, verify_plan, polynomial_range

KEY = 'delivery_repair'
DEFAULTS = dict(enabled=False, initial_compute_s=.10, send_guard_s=.02, max_attempts=2)


def validate(raw):
    if KEY not in raw:
        return None
    value = raw[KEY]
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('delivery_repair 配置字段无效')
    spec = {**DEFAULTS, **value}
    if not isinstance(spec['enabled'], bool):
        raise ValueError('delivery_repair.enabled 必须为布尔值')
    for name in ('initial_compute_s', 'send_guard_s'):
        x = spec[name]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0:
            raise ValueError('delivery_repair.'+name+' 必须为正有限数')
    if spec['send_guard_s'] < .02:
        raise ValueError('轨迹修正须保留至少20ms交付余量')
    n = spec['max_attempts']
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 2:
        raise ValueError('每个工件最多尝试1至2次轨迹修正')
    if spec['enabled']:
        if (raw.get('trajectory_profile') != 'smooth' or
                raw.get('intercept_strategy') != 'synchronized'):
            raise ValueError('轨迹修正需要平滑同步抓取')
        if raw.get('hardware', {}).get('joint_command_mode') != 'js':
            raise ValueError('轨迹修正需要JS连续发送模式')
        if (raw.get('hover_descent', {}).get('enabled', False) or
                'timing_experiment' in raw or 'execution_delay_window' in raw):
            raise ValueError('轨迹修正不可叠加其他时序或分段接续实验')
        if not raw.get('grasp_workpiece_collision_check_enabled', True):
            raise ValueError('轨迹修正必须检查运动工件碰撞')
    raw[KEY] = spec
    return spec


def enabled(config):
    return config.raw.get(KEY, {}).get('enabled', False)


def rank_recovery_branch(planner, old, state):
    """Route using only the frozen candidates and the current orientation reference."""
    from .smooth_planning import grasp_orientation_reference, order_obb_grasps

    planner.model.check_feedback(state)
    start = planner.model.tcp()[:3, :3].copy()
    reference, policy = grasp_orientation_reference(planner.config, planner.model, state, start)
    ranked = order_obb_grasps(old.target.grasps, start, synchronized=True, reference_rotation=reference)
    if not ranked:
        raise PickRejected('补救目标缺少候选抓姿')
    previous = next((i for i, pose in ranked
                     if np.allclose(pose, old.selected_grasp, rtol=0., atol=1e-8)), None)
    first = ranked[0][0]
    return dict(branch='repair' if first == previous else 'replan',
                first_grasp_index=first, old_grasp_index=previous, orientation_policy=policy['policy'])


def require_repair_reference(cfg, ref):
    """Use declared phase bounds, or the ordinary cell's fixed timing/bounds.

    Ordinary repair adds no timing grid or unmeasured delay/closure assumptions.
    Its stop must remain in the existing grasp interval; normal planning stays
    unchanged. Cartesian polynomial extrema cover the complete shifted join.
    """
    if 'phase_timing' in cfg.raw:
        return require_reference(cfg, ref)
    if not isinstance(ref, SynchronizedIntercept) or float(ref.velocity_world@cfg.direction) <= 0:
        raise PickRejected('轨迹修正需要同向同步参考')
    join = ref.prealign_s-ref.rotate_first_s
    durations = cfg.raw.get('synchronized_join_candidates_s', [cfg.raw['prealign_orientation_s']])
    if (not any(abs(join-x) < 1e-9 for x in durations)
            or abs(ref.rotate_first_s-cfg.raw.get('rotate_first_s', 0.)) > 1e-9
            or abs(ref.duration_s-ref.prealign_s-cfg.raw['synchronized_duration_s']) > 1e-9
            or abs(ref.follow_s-cfg.settings.full_plan_close_window_s) > 1e-9
            or abs(ref.brake_s-cfg.raw['brake_duration_s']) > 1e-9):
        raise PickRejected('轨迹修正不得改变原固定阶段时长')
    sync, follow, stop = [ref.position_derivatives(t)[0] for t in
                         (ref.prealign_s, ref.follow_end_s, ref.full_duration_s)]
    ranges = np.stack([polynomial_range(ref._join), polynomial_range(ref._braking),
        np.column_stack((np.minimum(sync, follow), np.maximum(sync, follow)))])
    margins = {
        'follow_interval_m': min(float(sync@cfg.direction)-cfg.follow_bounds[0],
                                 cfg.follow_bounds[1]-float(follow@cfg.direction)),
        'stop_interval_m': min(float(follow@cfg.direction)-cfg.follow_bounds[0],
                               cfg.follow_bounds[1]-float(stop@cfg.direction)),
        'workspace_m': float(min(np.min(ranges[:, :, 0]-cfg.workspace[:, 0]),
                                  np.min(cfg.workspace[:, 1]-ranges[:, :, 1])))}
    report = dict(policy='fixed_timing_existing_grasp_interval', margins=margins,
        join_s=join, follow_s=ref.follow_s, brake_s=ref.brake_s,
        stop_coordinate_m=float(stop@cfg.direction), stop_interval_base_m=cfg.follow_bounds.tolist(),
        braking_distance_m=float(np.linalg.norm(ref.velocity_world))*ref.brake_s/2,
        physical_grasp_guaranteed=False)
    if not np.isfinite(list(margins.values())).all() or min(margins.values()) < -1e-9:
        raise PickRejected('修正轨迹超出抓取区间、停止空间或工作区', details=report)
    fields = ('capture_s', 'start_s', 'duration_s', 'capture_tcp_world', 'velocity_world',
              'start_tcp_world', 'start_quaternion_wxyz', 'target_quaternion_wxyz',
              'follow_s', 'brake_s', 'prealign_s', 'rotate_first_s')
    if getattr(ref,'relative_descent_s',0.):
        fields += ('relative_descent_s','relative_sync_s','approach_clearance_m','object_axes_world',
                   'height_lead_s','height_normal_world','brake_rise_m')
    binding = dict(reference={k: getattr(ref, k) for k in fields}, workspace=cfg.workspace,
        follow_bounds=cfg.follow_bounds, direction=cfg.direction,
        velocity_limits=cfg.velocity_limits, acceleration_limits=cfg.acceleration_limits,
        advance=cfg.raw['command_advance_s'])
    report['binding_sha256'] = hashlib.sha256(json.dumps(binding, sort_keys=True,
        default=lambda v: np.asarray(v).tolist(), allow_nan=False).encode()).hexdigest()
    return report


def verify_repaired_plan(cfg, path, ref):
    report = require_repair_reference(cfg, ref)
    saved = path.diagnostics.get('delivery_repair_bounds')
    if saved is None or saved.get('binding_sha256') != report['binding_sha256']:
        raise PickRejected('补救轨迹的空间认证缺失或已过期')
    return report


def repair_intercept(planner, old, state, *, deadline, cancelled=lambda: False,
                     new_start_s=None):
    """Keep the grasp/durations, reconnect from fresh measured rest joints.

    new_start_s is an optional prospective schedule for controlled offline
    replay. Production chooses it from measured repair and delivery costs.
    """
    from .planning import InterceptPlan
    cfg, model, clock = planner.config, planner.model, planner.clock
    spec = cfg.raw[KEY]
    begun = clock()
    if not enabled(cfg):
        raise PickRejected('轨迹修正未开启')
    if not np.isfinite(deadline) or begun >= deadline or cancelled():
        raise PickRejected('轨迹修正总预算耗尽或任务取消')
    if (planner.target is not old.target or planner.identity != old.target.identity or
            old.path.identity != old.target.identity or old.path.version != planner.version or
            old.scene_revision != planner.scene_revision):
        raise PickRejected('轨迹修正的目标、场景或版本已改变')
    ref0 = old.reference
    if (not isinstance(ref0, SynchronizedIntercept) or
            ref0.capture_s != old.target.capture_s or
            not np.array_equal(ref0.velocity_world, old.target.velocity)):
        raise PickRejected('轨迹修正的原预测模型不一致')
    if 'phase_timing' in cfg.raw:
        verify_plan(cfg, old.path, ref0)
    q = np.asarray(state['joints'], float)
    if (q.shape != (7,) or not np.isfinite(q).all() or
            np.max(abs(q-old.path.q0)) > .005 or
            not np.isfinite([state['width_m'], state['force_n']]).all() or
            abs(state['width_m']-cfg.settings.open_width_m) > .002 or
            state['force_n'] >= cfg.settings.min_force_n):
        raise PickRejected('轨迹修正要求机械臂仍在原起点附近且空爪张开')
    model.check_feedback(state)
    tcp = model.tcp().copy()
    estimate = max(spec['initial_compute_s'], getattr(planner, 'repair_compute_s', 0.))
    post = max(0., getattr(planner, 'measured_post_planning_s', 0.))
    advance = cfg.raw['command_advance_s']
    start = (begun+estimate+post+spec['send_guard_s']+advance
             if new_start_s is None else float(new_start_s))
    if not np.isfinite(start) or start <= ref0.start_s:
        raise PickRejected('轨迹修正必须安排新的未来执行时刻')
    send_deadline = start-advance
    stop_at = min(deadline, send_deadline-spec['send_guard_s'])
    scene_revision, target = planner.scene_revision, old.target
    version = planner.version
    failures, timings = [], dict(scene_prepare=0., candidate_selection=0.)

    def abort():
        return (cancelled() or clock() >= stop_at or planner.target is not target or
                planner.identity != target.identity or planner.scene_revision != scene_revision or
                planner.version != version)

    def check():
        if abort():
            raise PickRejected('轨迹修正已取消、过期或场景版本改变', details={
                'reason':'repair_deadline_or_invalidated', 'send_deadline_s':send_deadline,
                'compute_deadline_s':stop_at, 'wall_s':clock()-begun})

    try:
        check()
        ref = replace(ref0, start_s=start, start_tcp_world=tcp[:3, 3].copy(),
            start_quaternion_wxyz=Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3, 0, 1, 2]])
        bounds = require_repair_reference(cfg, ref)  # Reject missing stopping space before IK.
        planner.version += 1
        version = planner.version
        offset = q-old.path.q0
        for rate in dict.fromkeys([cfg.raw['planning_knot_hz'], cfg.raw['planning_knot_hz_fallback']]):
            check()
            stage, started = 'ik', clock()
            try:
                times = sampling_times(ref.full_duration_s, 1./rate, ref.rotate_first_s,
                    ref.prealign_s, ref.duration_s, ref.follow_end_s,
                    *(ref.approach_knots if ref.relative_descent_s else ()))
                solver, joints = PrecisePoseSolver(model), [q.copy()]
                for elapsed in times[1:]:
                    check()
                    # Small measured start correction fades into the old IK branch.
                    u = min(1., elapsed/ref.prealign_s)
                    fade = 1.-(10*u**3-15*u**4+6*u**5)
                    result = solver.solve(*ref.sample(elapsed),
                        old.path.position(elapsed)+fade*offset, cancelled=abort)
                    if not result.success:
                        raise PickRejected('修正轨迹逆解未收敛', details={'elapsed_s':float(elapsed)})
                    joints.append(result.joint_positions.copy())
                timings['ik'] = clock()-started
                check()
                stage, started = 'polynomial_limits', clock()
                path = certify_curve(times, joints, cfg, model.command_lower, model.command_upper,
                    identity=target.identity, version=version, stage=old.path.stage,
                    zero_end_acceleration=True)
                timings[stage] = clock()-started
                check()
                stage, started = 'sampled_geometry', clock()
                geometry = planner.check_smooth_intercept(path, ref, cancelled=abort)
                timings[stage] = clock()-started
                check()
                detail = dict(policy='same_target_scene_and_old_curve_seed',
                    original_start_s=ref0.start_s, new_start_s=start,
                    total_shift_s=start-ref0.start_s,
                    contact_shift_m=(ref.contact_xyz-ref0.contact_xyz).tolist(),
                    compute_estimate_s=estimate, post_estimate_s=post,
                    completed_monotonic_s=clock(), wall_s=clock()-begun,
                    send_deadline_s=send_deadline, send_margin_s=send_deadline-clock(),
                    scene_revision=scene_revision, original_version=old.path.version,
                    start_correction_rad=(q-old.path.q0).tolist())
                diagnostics = dict(profile='smooth', intercept_strategy='synchronized',
                    delivery_repair_bounds=bounds, reference_check=geometry, delivery_repair=detail,
                    prealign_s=ref.prealign_s, rotate_first_s=ref.rotate_first_s,
                    join_s=ref.prealign_s-ref.rotate_first_s,
                    follow_s=ref.follow_s, brake_s=ref.brake_s, rest_qva_endpoints=True,
                    ik_knot_hz=rate, ik_solve_count=len(times)-1,
                    peak_velocity_rad_s=np.max(abs(np.asarray(extrema(path.curve, 1))), axis=0).tolist(),
                    peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0).tolist())
                if 'phase_timing' in cfg.raw:
                    diagnostics['phase_timing'] = bounds
                path = replace(path, planning_s=clock()-begun, diagnostics=diagnostics)
                check()
                planner.contact_reference = ref
                return InterceptPlan(path, ref, target, scene_revision, old.selected_grasp.copy(),
                    tuple(failures), {**timings, 'delivery_repair':detail})
            except PickRejected as error:
                failures.append(dict(ik_knot_hz=rate, failed_phase=stage, reason=str(error),
                    details=error.details, partial_phase_wall_s=clock()-started))
                check()
        raise PickRejected('修正轨迹未通过原运动或几何约束', details={'failures':failures})
    finally:
        # Include failed work; a cheap bounds rejection never lowers the estimate.
        if not cancelled():
            planner.repair_compute_s = max(estimate, clock()-begun)


def record(executor, name, data, plan=None):
    """Small cycle events plus loss-accounted curve archives for every repair."""
    metadata = {'identity':executor.owner, **data}
    archive = getattr(executor.planner, 'experiment_archive', None)
    if archive is not None:
        arrays = {}
        if plan is not None:
            metadata.update(version=plan.path.version, scene_revision=plan.scene_revision,
                reference={k:getattr(plan.reference, k) for k in
                    ('capture_s','start_s','duration_s','follow_s','brake_s','prealign_s',
                     'rotate_first_s','bin_descent_s','bin_clearance_m','brake_rise_m','height_lead_s','height_normal_world',
                     'relative_descent_s','relative_sync_s','approach_clearance_m','object_axes_world','capture_tcp_world','velocity_world','start_tcp_world',
                     'start_quaternion_wxyz','target_quaternion_wxyz') if hasattr(plan.reference,k)},
                diagnostics=plan.path.diagnostics)
            arrays = dict(curve_c=plan.path.curve.c, curve_x=plan.path.curve.x,
                          selected_grasp=plan.selected_grasp)
        archive._offer(name, metadata, arrays)
    executor._emit(name, {'identity':executor.owner, **data})
