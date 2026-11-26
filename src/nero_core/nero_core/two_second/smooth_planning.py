"""Duration/candidate scheduling for measured smooth interception."""
from dataclasses import replace
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import BPoly, CubicSpline
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .precise_ik import PrecisePoseSolver
from .smooth_reference import SmoothIntercept
from .synchronized_reference import (SynchronizedIntercept, schedule_forward_join,
                                      admit_synchronized_observation, freeze_window_for_speed)
from .reference import predict_entry_time, DEFAULT_FREEZE_UPSTREAM_DISTANCE_M, EntryNotReady
from .trajectory import sampling_times, certify_curve, extrema


def order_obb_grasps(grasps, start_rotation, *, synchronized, reference_rotation=None):
    """Rank the two short-axis poses by full TCP rotation to a reference."""
    indexed = list(enumerate(grasps))[:2]
    reference = start_rotation if reference_rotation is None else reference_rotation
    indexed.sort(key=lambda item: Rotation.from_matrix(reference.T@item[1][:3, :3]).magnitude())
    return indexed


def bin_approach_samples(solver, reference, times, start_joints, config, cancelled, *, poses=None):
    """Joint transit from the bin, then Cartesian tracking from above the grasp."""
    join = reference.relative_alignment_s if getattr(reference,'relative_descent_s',0.) else reference.prealign_s - reference.bin_descent_s
    i = np.searchsorted(times, join)
    if len(times)-i < 4 or not np.isclose(times[i], join, atol=1e-9, rtol=0):
        raise PickRejected('料框接入点缺少连续的抓取段采样')
    # Standby seeds only the IK branch on the belt. It is never a waypoint.
    seed = config.standby
    suffix = []
    for j in range(i, len(times)):
        t = times[j]
        pose = reference.sample(t) if poses is None else poses[j]
        result = solver.solve(*pose, seed, cancelled=cancelled)
        if not result.success:
            raise PickRejected('离框后的抓取段逆解未收敛', details={'time_s':float(t)})
        seed = result.joint_positions.copy()
        suffix.append(seed)
    blend = CubicSpline(times[i:i+4], suffix[:4])
    lift_height = max(config.settings.lift_m, config.raw['maximum_target_height_m'])
    towards_belt = reference.contact_xyz[:2] - reference.start_tcp_world[:2]
    horizontal = towards_belt * min(1., config.settings.lift_m/max(np.linalg.norm(towards_belt), 1e-9))
    lift = solver.solve(reference.start_tcp_world + np.r_[horizontal, lift_height],
        reference.start_quaternion_wxyz, start_joints, cancelled=cancelled)
    if not lift.success:
        raise PickRejected('料框上方的空爪抬升位置不可达')
    # Allocate the first leg using the exact rest-to-rest quintic velocity
    # and acceleration bounds. The fitted full curve is certified afterwards.
    dq = abs(lift.joint_positions-start_joints)
    lift_s = max(float(np.max(1.875*dq/config.velocity_limits)),
        float(np.max(np.sqrt(10/np.sqrt(3)*dq/config.acceleration_limits))))
    lift_s += 1./config.settings.control_hz
    if not 0 < lift_s < join:
        raise PickRejected('空爪离框抬升时间超过可用接入时间')
    transit = BPoly.from_derivatives([0., lift_s, join], [[start_joints, np.zeros(7), np.zeros(7)],
        [lift.joint_positions, np.zeros(7), np.zeros(7)],
        [suffix[0], blend(join, 1), blend(join, 2)]])
    return np.vstack((transit(times[:i]), suffix)), float(join)


def grasp_orientation_reference(cfg, model, state, start_rotation):
    """Use the calibrated TCP frame at the active transfer's first joint sample."""
    policy = cfg.raw.get('obb_grasp_orientation_reference', 'current_tcp')
    if policy == 'current_tcp':
        return np.asarray(start_rotation).copy(), {'policy': policy}
    if policy != 'taught_transfer_start' or not cfg.raw.get('taught_playback_enabled', True):
        raise PickRejected('OBB 抓姿方向参考配置无效')
    segment = cfg.raw.get('taught_segments', {}).get('transfer')
    if segment is None:
        raise PickRejected('OBB 抓姿排序缺少已加载的搬运示教轨迹')
    # Use the same calibrated physical TCP as the candidates; restore the
    # newest measured robot state before any scheduling or IK work.
    try:
        model.set_state(segment.joints[0], state['width_m'])
        reference = model.tcp()[:3, :3].copy()
    finally:
        model.set_state(state['joints'], state['width_m'])
    return reference, {'policy': policy, 'taught_segment_path': segment.path,
                       'taught_segment_sha256': segment.sha256,
                       'taught_joint_sample_index': 0}


def plan_smooth_intercept(planner, target, state, *, cancelled=lambda: False,
                          reservation_s=None, measure_late=False, global_deadline=None,
                          scene_cache=None, recovery_replan=False):
    options = dict(cancelled=cancelled, reservation_s=reservation_s, measure_late=measure_late,
        global_deadline=global_deadline, scene_cache=scene_cache, recovery_replan=recovery_replan)
    raw = planner.config.raw
    if (raw.get('parallel_belt_scene_enabled', False)
            and raw.get('environment_collision_scope') == 'measured_belt_surface_only'):
        from .parallel_scene import ParallelBeltScene
        with ParallelBeltScene(planner, target) as preparation:
            return _plan_smooth_intercept(planner, target, state,
                                         scene_preparation=preparation, **options)
    return _plan_smooth_intercept(planner, target, state, **options)


def _plan_smooth_intercept(planner, target, state, *, cancelled=lambda: False,
                          reservation_s=None, measure_late=False, global_deadline=None,
                          scene_cache=None, recovery_replan=False, scene_preparation=None):
    from .planning import InterceptPlan
    cfg, model, clock = planner.config, planner.model, planner.clock
    from .delivery_repair import enabled as repair_enabled
    retain_expired = repair_enabled(cfg)
    measure_late = measure_late or retain_expired
    retry_policy = cfg.raw.get('candidate_retry_policy', 'per_grasp')
    if retry_policy not in ('per_grasp', 'coarse_first'):
        raise ValueError('unknown candidate retry policy')
    coarse_first = retry_policy == 'coarse_first'
    begun = clock()
    planner.attempt_diagnostics = []
    planner._progress('smooth_entry_selection', identity=target.identity)
    entry_position = getattr(target, 'entry_position', target.center)
    prealign_s = cfg.raw.get('prealign_orientation_s', 0.)
    synchronized = cfg.raw.get('intercept_strategy', 'classic') == 'synchronized'
    rotate_first_s=cfg.raw.get('rotate_first_s',0.) if synchronized else 0.
    synchronized_s = cfg.raw.get('synchronized_duration_s', .2)
    reservation_s = cfg.raw['planning_reserve_s'] if reservation_s is None else reservation_s
    upstream_distance = cfg.raw.get('freeze_upstream_distance_m', DEFAULT_FREEZE_UPSTREAM_DISTANCE_M)
    freeze_window = None
    try:
        if synchronized and getattr(target,'detection',{}).get('native_observation_version') == 4:
            from .synchronized_reference import native_observation_entry
            entry=native_observation_entry(target.capture_s,entry_position@cfg.direction,
                target.velocity@cfg.direction,cfg.follow_bounds,begun,cfg.raw['maximum_observation_age_s'])
        elif synchronized:
            freeze_window = freeze_window_for_speed(cfg, target.velocity@cfg.direction,
                belt_region=getattr(target, 'detection', {}).get('belt_region'))
            upstream_distance = freeze_window['distance_m']
            entry = admit_synchronized_observation(target.capture_s, entry_position@cfg.direction,
                target.velocity@cfg.direction, cfg.follow_bounds, begun, upstream_distance,
                minimum_upstream_distance_m=freeze_window.get('minimum_distance_m'))
        else:
            entry = predict_entry_time(target.capture_s, entry_position@cfg.direction,
                target.velocity@cfg.direction, cfg.follow_bounds[0], begun, cfg.raw['planning_reserve_s']+prealign_s,
                maximum_upstream_distance_m=upstream_distance)
    except EntryNotReady as error:
        # A normal target admission rejection must use the same lifecycle as
        # other planning rejections; it must not terminate return lookahead.
        raise PickRejected('冻结观测未满足规划入口条件：'+str(error), details={
            'reason':error.reason, 'stage':'smooth_entry_selection',
            'identity':target.identity, 'capture_s':target.capture_s,
            'observation_age_s':begun-target.capture_s,
            'coordinate_m':float(entry_position@cfg.direction),
            'maximum_upstream_distance_m':upstream_distance, 'freeze_window':freeze_window}) from error
    model.check_feedback(state)
    start = model.tcp().copy()
    solver = PrecisePoseSolver(model)
    send_deadline = float('inf') if synchronized else entry-prealign_s-cfg.raw['command_advance_s']
    planning_deadline = (begun+cfg.raw['planning_budget_s'] if global_deadline is None else global_deadline)
    deadline = min(planning_deadline, send_deadline)
    latest_send_deadline = float('inf')
    failures, scene_prepared = [], False
    timings = {'scene_prepare': 0.}
    def scene_key():
        return (id(target), id(state), id(cfg), id(model), planner.version, planner.scene_revision)
    if scene_cache is not None and scene_cache.get('key') == scene_key():
        scene_prepared = True
        scene_cache['reused_wall_s'] = scene_cache['prepare_wall_s']
        timings['scene_reused_saved_s'] = scene_cache['prepare_wall_s']
    def stop_reason(candidate_deadline=None):
        now = clock()
        if cancelled():
            return 'cancelled'
        if not measure_late and now >= send_deadline:
            return 'send_deadline'
        if now >= planning_deadline:
            return 'global_planning_budget'
        if now >= latest_send_deadline:
            return 'target_window'
        if candidate_deadline is not None and now >= candidate_deadline:
            return 'candidate_budget'
        return None

    def stopped():
        return PickRejected('平滑抓取规划总预算耗尽或任务取消', details={
            'failures': failures, 'stop_reason': stop_reason(),
            'planning_wall_s': clock()-begun,
            'available_planning_s': deadline-begun,
            'send_deadline_monotonic_s': send_deadline,
            'scene_prepared': scene_prepared})
    # Only the two finger-symmetric short-axis poses may be selected.
    reference_rotation, orientation_reference = grasp_orientation_reference(cfg, model, state, start[:3, :3])
    indexed_grasps = order_obb_grasps(target.grasps, start[:3, :3], synchronized=synchronized,
                                     reference_rotation=reference_rotation)
    orientation_reference['candidate_angle_deg'] = [
        {'input_grasp_index': index, 'angle_deg': float(np.degrees(
            Rotation.from_matrix(reference_rotation.T @ grasp[:3, :3]).magnitude()))}
        for index, grasp in indexed_grasps]
    grasps = [grasp for _, grasp in indexed_grasps]
    durations=(cfg.raw.get('synchronized_join_candidates_s', [prealign_s]) if synchronized
               else cfg.raw['approach_duration_candidates_s'])
    from_bin = getattr(planner, 'bin_departure_tree', None) is not None
    if from_bin:
        approach = sorted(cfg.raw['approach_duration_candidates_s'])
        preferred = np.mean(approach[:2])
        durations = sorted(set(durations).union(approach))
        durations = sorted(set(durations).union((a+b)/2 for a,b in zip(durations, durations[1:])),
            key=lambda duration: (abs(duration-preferred), duration))
    approach = cfg.raw.get('object_relative_approach')
    if approach and not cfg.raw.get('grasp_workpiece_collision_check_enabled', True):
        raise PickRejected('相对接近必须复核运动工件点云')
    from .brake_height import reference_options
    height_options = reference_options(cfg,target) if synchronized else {}
    if approach and not from_bin:
        from .object_approach import order_joins, wrist_time
        angle = min(Rotation.from_matrix(start[:3,:3].T @ g[:3,:3]).magnitude() for g in grasps)
        durations = order_joins(durations,approach,angle,cfg,height_options.get('height_lead_s',0.),
            earliest=max(clock()+reservation_s,getattr(planner,'intercept_not_before_s',None) or begun)
                     +cfg.raw['command_advance_s'], target=target, grasp=grasps[0][:3,3])
    timing_search = None
    timing_trials = [(duration, None) for duration in durations]
    if 'phase_timing' in cfg.raw:
        from .phase_timing import candidate_options, validate_configuration
        validate_configuration(cfg.raw, cfg.workspace, cfg.direction)
        planner._progress('phase_timing_search', identity=target.identity)
        options, timing_search = candidate_options(cfg, target, start, grasps,
            now_s=clock(), reservation_s=reservation_s,
            cancelled=lambda: cancelled() or clock() >= planning_deadline)
        if not options:
            raise PickRejected('没有满足联合阶段约束的时长组合',
                details={'reason': 'phase_timing_no_candidate', 'phase_timing_search': timing_search})
        timing_trials = [(option['join_s'], option) for option in options]
    rotation_cache = {}
    first_hz = cfg.raw.get('planning_knot_hz', min(50., cfg.settings.control_hz))
    fallback_hz = cfg.raw.get('planning_knot_hz_fallback', first_hz)
    all_knot_rates = [first_hz] if fallback_hz == first_hz else [first_hz, fallback_hz]
    clearance_cache = {}
    trials = deque((duration, timing_option, descent, None) for duration, timing_option in timing_trials
                   for descent in (approach['descent_candidates_s'] if approach else [0.]))
    while trials:
        duration, timing_option, descent, deferred = trials.popleft()
        follow_s = (cfg.settings.full_plan_close_window_s if timing_option is None else timing_option['follow_s'])
        brake_s = cfg.raw['brake_duration_s'] if timing_option is None else timing_option['brake_s']
        trial_grasps = (list(enumerate(grasps)) if timing_option is None else
                       [(timing_option['grasp_index'], grasps[timing_option['grasp_index']])])
        if approach and not from_bin:
            high_time = duration - descent - approach['sync_s'] - height_options.get('height_lead_s',0.)
            def alignment_overrun(item):
                angle = Rotation.from_matrix(start[:3,:3].T@item[1][:3,:3]).magnitude()
                return max(0.,wrist_time(angle,cfg)-high_time)
            trial_grasps.sort(key=alignment_overrun)  # stable: keep taught preference when both fit
        # Legacy entries finish both rates for A before trying B. The separate
        # coarse-first entry defers only eligible dense retries behind all
        # remaining coarse candidates; a fully certified coarse path returns
        # immediately. Every deferred trial gets a fresh moving-target schedule.
        work_items = deque((index, grasp, [first_hz] if coarse_first else all_knot_rates, 0.)
                           for index, grasp in trial_grasps)
        budget_ends = {}
        if deferred is not None:
            item, end = deferred
            work_items = deque([item])
            budget_ends[item[0]] = end
        while work_items:
            index, grasp, knot_rates, extra_height = work_items.popleft()
            if coarse_first and (cancelled() or clock() >= planning_deadline):
                raise stopped()
            if coarse_first and index in budget_ends and clock() >= budget_ends[index]:
                record = {'candidate_index':index, 'approach_s':duration,
                    'ik_knot_hz':knot_rates[0], 'ik_solve_count':0,
                    'candidate_retry_policy':retry_policy, 'reason':'加密重算前原候选预算已耗尽',
                    'details':{'reason':'deferred_candidate_budget_expired'},
                    'failed_phase':'deferred_retry_budget', 'candidate_budget_exhausted':True,
                    'stop_reason':'candidate_budget'}
                failures.append(record)
                planner.attempt_diagnostics.append({**record, 'status':'rejected', 'wall_s':0.})
                continue
            rotate_first_s = cfg.raw.get('rotate_first_s',0.) if synchronized else 0.
            rotation_timing = None
            if rotate_first_s and cfg.raw.get('rotate_first_retime_enabled',False):
                if index not in rotation_cache:
                    from .rotation_lead import estimate_rotation_lead
                    planner._progress('rotation_timing', candidate_index=index)
                    try:
                        rotation_cache[index] = estimate_rotation_lead(cfg, model, solver, state, start, grasp,
                            cancelled=lambda: cancelled() or clock() >= planning_deadline)
                    except PickRejected as error:
                        rotation_cache[index] = error
                estimate = rotation_cache[index]
                if isinstance(estimate, PickRejected):
                    failures.append({'candidate_index':index, 'approach_s':duration,
                        'failed_phase':'rotation_preparation', 'reason':str(estimate), 'details':estimate.details})
                    if cancelled() or clock() >= planning_deadline:
                        raise stopped()
                    continue
                rotation_timing = estimate
                rotate_first_s = estimate['duration_s']
            if synchronized:
                # Each selected grasp owns its predicted contact and schedule.
                # Delay the join so it does not move against belt travel.
                try:
                    follow_bounds = cfg.follow_bounds
                    if 'execution_delay_window' in cfg.raw:
                        from .execution_delay_window import planning_bounds
                        follow_bounds = planning_bounds(cfg, target.velocity)
                    alignment_lead = 0.
                    if approach:
                        height_lead = height_options.get('height_lead_s',0.)
                        alignment_lead = descent+approach['sync_s']+max(0.,height_lead-synchronized_s)
                        if duration <= alignment_lead:
                            raise ValueError('高位对齐和相对下降超过接入时长')
                    schedule = schedule_forward_join(capture_s=target.capture_s,
                        from_bin=from_bin, forward_ramp_s=approach['forward_ramp_s'] if approach else None,
                        grasp_position=grasp[:3,3], velocity=target.velocity,
                        standby_position=start[:3,3], direction=cfg.direction,
                        now_s=clock(), planning_reserve_s=reservation_s,
                        advance_s=cfg.raw['command_advance_s'], join_s=duration-alignment_lead, rotate_first_s=rotate_first_s,
                        synchronized_s=synchronized_s+alignment_lead, follow_s=follow_s,
                        follow_bounds=follow_bounds,
                        earliest_dispatch_s=getattr(planner, 'intercept_not_before_s', None))
                except ValueError as error:
                    failures.append({'candidate_index': index, 'approach_s':duration, 'reason': str(error),
                                     'failed_phase': 'synchronized_schedule',
                                     **({'details': error.details} if hasattr(error, 'details') else {})})
                    continue
                send_deadline = schedule['join_start_s']-cfg.raw['command_advance_s']
                deadline = planning_deadline if measure_late else min(planning_deadline, send_deadline)
            if cancelled() or clock() >= deadline:
                raise stopped()
            attempt_started = clock()
            if coarse_first and synchronized:
                speed = float(target.velocity@cfg.direction)
                latest_send_deadline = (target.capture_s +
                    (follow_bounds[1]-float(grasp[:3,3]@cfg.direction))/speed -
                    duration-rotate_first_s-synchronized_s-follow_s-cfg.raw['command_advance_s'])
            if coarse_first:
                coarse_remaining = 1+sum(rates == [first_hz] for _,_,rates,_ in work_items)
                share = max(0., latest_send_deadline-attempt_started)/coarse_remaining
                budget_end = budget_ends.setdefault(index, attempt_started+
                    min(cfg.raw['candidate_planning_budget_s'], share))
                candidate_deadline = min(deadline, budget_end, latest_send_deadline)
            else:
                candidate_deadline = min(deadline, attempt_started+cfg.raw['candidate_planning_budget_s'])
            for knot_hz in knot_rates:
                attempt_timings = {}
                ik_solve_count = 0
                phase, phase_started = 'bounds', clock()
                def enter_phase(name):
                    nonlocal phase, phase_started
                    now = clock()
                    attempt_timings[phase] = attempt_timings.get(phase, 0.)+now-phase_started
                    phase, phase_started = name, now
                def abort():
                    return cancelled() or clock() >= candidate_deadline
                preparing_scene = False
                if not synchronized and entry-prealign_s < target.capture_s:
                    raise PickRejected('抓姿预转起点早于观测，不能采用过期轨迹')
                reference_type = SynchronizedIntercept if synchronized else SmoothIntercept
                start_time = schedule['join_start_s'] if synchronized else entry-prealign_s
                reference_duration = synchronized_s if synchronized else duration
                effective_prealign = duration+rotate_first_s if synchronized else prealign_s
                # Approach the moving grasp from 8 cm above contact, allowing
                # up to 0.6 s for descent before the normal synchronized close.
                bin_descent = min(.6, duration/2) if from_bin else 0.
                try:
                    relative_options = {}
                    if approach:
                        from .object_approach import required_clearance
                        if index not in clearance_cache:
                            clearance_cache[index] = required_clearance(model, target, grasp, state['joints'],
                                state['width_m'], cfg.raw.get('workpiece_collision_margin_m', .004)
                                + model.ik.config.position_tolerance)
                        relative_options = dict(relative_descent_s=descent, relative_sync_s=approach['sync_s'],
                            approach_clearance_m=clearance_cache[index]+extra_height,
                            object_axes_world=tuple(map(tuple,target.object_pose[:3,:3])))
                    ref = reference_type(target.capture_s, start_time, reference_duration+effective_prealign, grasp[:3, 3], target.velocity,
                        start[:3, 3], Rotation.from_matrix(start[:3, :3]).as_quat()[[3, 0, 1, 2]],
                        Rotation.from_matrix(grasp[:3, :3]).as_quat()[[3, 0, 1, 2]],
                        follow_s, brake_s, effective_prealign,
                        **({'rotate_first_s':rotate_first_s, 'bin_descent_s':bin_descent,
                            'bin_clearance_m':.08 if from_bin else 0., **height_options}
                            if synchronized else {}), **relative_options)
                    planner._progress('smooth_candidate_bounds', candidate_index=index, approach_s=duration)
                    phase_timing_report = None
                    if timing_option is not None:
                        from .phase_timing import require_reference
                        phase_timing_report = require_reference(cfg, ref)
                    if (recovery_replan or from_bin) and timing_option is None:
                        # Evaluate this candidate's own future stop. A rejection leaves
                        # the remaining grasps and duration choices in the search.
                        enter_phase('stop_space')
                        stop = float(ref.position_derivatives(ref.full_duration_s)[0] @ cfg.direction)
                        margin = float(cfg.follow_bounds[1] - stop)
                        if not np.isfinite(margin) or margin < -1e-9:
                            raise PickRejected('重规划候选的停止终点超出抓取区间', details={
                                'reason': 'replan_stop_space', 'stop_coordinate_m': stop,
                                'stop_margin_m': margin})
                        enter_phase('bounds')
                    contact = ref.contact_xyz@cfg.direction
                    follow_end = (ref.contact_xyz+target.velocity*ref.follow_s)@cfg.direction
                    if min(contact, follow_end) < cfg.follow_bounds[0] or max(contact, follow_end) > cfg.follow_bounds[1]:
                        raise PickRejected('平滑接触和跟随段超出抓取区间')
                    delay_window = None
                    if 'execution_delay_window' in cfg.raw:
                        from .execution_delay_window import require_reference
                        delay_window = require_reference(cfg, ref)
                    boundaries = ref.approach_knots if approach else (
                        rotate_first_s, ref.prealign_s-bin_descent, ref.prealign_s,
                        min(ref.prealign_s,ref.duration_s-ref.height_lead_s) if synchronized and ref.brake_rise_m else 0.)
                    times = sampling_times(ref.full_duration_s, 1./knot_hz, *boundaries, ref.duration_s, ref.follow_end_s)
                    poses = [ref.sample(t) for t in times]
                    xyzs = np.array([xyz for xyz, _ in poses])
                    if np.any(xyzs < cfg.workspace[:, 0]) or np.any(xyzs > cfg.workspace[:, 1]):
                        raise PickRejected('平滑接近/跟随/减速超出工作区')
                    if scene_preparation is not None and not scene_prepared:
                        scene_preparation.start()
                    enter_phase('ik')
                    joints = [np.asarray(state['joints']).copy()]
                    joint_transit_s = 0.
                    remaining = times[1:]
                    if from_bin:
                        joints, joint_transit_s = bin_approach_samples(solver, ref, times, joints[0], cfg, abort, poses=poses)
                        ik_solve_count = int(np.sum(times >= joint_transit_s))
                        remaining = ()
                    for i, t in enumerate(remaining, 1):
                        planner._progress('smooth_intercept_ik', candidate_index=index, approach_s=duration,
                            waypoint_index=i, waypoint_count=len(times)-1, trajectory_time_s=float(t),
                            candidate_wall_s=clock()-attempt_started,
                            remaining_budget_s=max(0., candidate_deadline-clock()))
                        ik_solve_count += 1
                        solved = solver.solve(*poses[i], joints[-1], cancelled=abort)
                        if not solved.success:
                            raise PickRejected('平滑路径逆解未收敛', details={'time_s': float(t),
                                'position_error_m': solved.position_error, 'orientation_error_rad': solved.orientation_error,
                                'message': solved.message})
                        joints.append(solved.joint_positions.copy())
                    timings['selected_candidate_ik'] = clock()-phase_started
                    enter_phase('polynomial_limits')
                    check_started = clock()
                    path = certify_curve(times, joints, cfg, model.command_lower, model.command_upper,
                        identity=target.identity, version=planner.version, stage='smooth_intercept_follow_brake',
                        zero_end_acceleration=True)
                    timings['polynomial_limits'] = clock()-check_started
                    # Common scene preparation consumes the global deadline, not
                    # one candidate's private IK/geometry budget.
                    if not scene_prepared:
                        preparing_scene = True
                        enter_phase('scene_prepare')
                        check_started = clock()
                        planner._progress('scene_prepare', identity=target.identity)
                        if scene_preparation is None:
                            planner.install(target, state)
                        else:
                            scene_preparation.install(target, state, cancelled=lambda:
                                cancelled() or clock() >= min(deadline, latest_send_deadline))
                            timings.update(scene_preparation.timings)
                        timings['scene_prepare'] = clock()-check_started
                        if scene_cache is not None:
                            scene_cache.update(key=scene_key(), prepare_wall_s=timings['scene_prepare'])
                        if coarse_first:
                            for pending_index in budget_ends:
                                budget_ends[pending_index] += timings['scene_prepare']
                        candidate_deadline = min(deadline, latest_send_deadline,
                            candidate_deadline+timings['scene_prepare'])
                        scene_prepared = True
                        preparing_scene = False
                    path = replace(path, version=planner.version)
                    if from_bin:
                        path.diagnostics['bin_joint_transit_s'] = joint_transit_s
                    enter_phase('sampled_geometry')
                    planner.candidate_context = {'candidate_index':index, 'approach_s':duration,
                        'input_grasp_index':next((i for i,g in enumerate(target.grasps) if g is grasp),None),
                        'selected_grasp_tcp_base':grasp.tolist()}
                    check_started = clock()
                    reference_check = planner.check_smooth_intercept(path, ref, cancelled=abort)
                    timings['sampled_geometry'] = clock()-check_started
                    if abort():
                        raise PickRejected('平滑轨迹未在实际发送截止前完成认证')
                    expired = clock() >= send_deadline
                    if measure_late and expired and not retain_expired:
                        raise PickRejected('规划完成时已错过预约发送，必须重新规划', details={
                            'reason': 'measured_deadline_missed',
                            'measured_candidate_wall_s': clock()-attempt_started,
                            'missed_by_s': clock()-send_deadline})
                    diagnostics = {'profile': 'smooth', 'ik_knot_hz': knot_hz, 'ik_solve_count': ik_solve_count,
                        'candidate_retry_policy':retry_policy,
                        'reference_check': reference_check, 'approach_s': duration, 'prealign_s': effective_prealign,
                        'rotate_first_s':rotate_first_s if synchronized else prealign_s,
                        'grasp_axis_policy':'obb_short_axis_only',
                        'grasp_orientation_reference':orientation_reference,
                        'translation_start_elapsed_s':rotate_first_s if synchronized else prealign_s,
                        'follow_s': ref.follow_s,
                        'brake_s': ref.brake_s, 'rest_qva_endpoints': True,
                        'peak_velocity_rad_s': np.max(abs(np.asarray(extrema(path.curve, 1))), axis=0).tolist(),
                        'peak_acceleration_rad_s2': np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0).tolist(),
                        'brake_geometry': 'visual_attachment_precheck_and_live_measured_payload_edges'}
                    if expired:
                        # Fully certified geometry is useful as a repair seed.
                        # The executor still rejects its expired absolute schedule.
                        diagnostics['expired_delivery_candidate'] = {
                            'requires_repair_before_dispatch':True,
                            'send_deadline_s':send_deadline, 'missed_by_s':clock()-send_deadline}
                    if synchronized:
                        if from_bin:
                            diagnostics['bin_joint_transit_s'] = joint_transit_s
                        diagnostics.update(intercept_strategy='synchronized', synchronized_schedule=schedule,
                                           synchronized_s=synchronized_s, join_s=duration,
                                           rotation_timing=rotation_timing,
                                           freeze_window=freeze_window,
                                           selected_input_grasp_index=indexed_grasps[index][0])
                    if delay_window is not None:
                        diagnostics['execution_delay_window'] = delay_window
                    if approach:
                        diagnostics['object_relative_approach'] = dict(alignment_s=ref.relative_alignment_s,
                            descent_start_s=ref.relative_alignment_s+ref.relative_sync_s,
                            descent_s=descent, clearance_m=ref.approach_clearance_m,
                            close_s=ref.duration_s, object_axes_world=ref.object_axes_world)
                    if phase_timing_report is not None:
                        diagnostics['phase_timing'] = phase_timing_report
                        diagnostics['phase_timing_search'] = timing_search
                    timings['full_planning_delivery'] = clock()-begun
                    path = replace(path, planning_s=timings['full_planning_delivery'], diagnostics=diagnostics)
                    planner.contact_reference = ref
                    enter_phase('delivered')
                    planner.attempt_diagnostics.append({'candidate_index': index, 'approach_s': duration,
                        'status': 'delivered', 'wall_s': clock()-attempt_started, 'diagnostics': diagnostics,
                        'phase_timings_s': dict(attempt_timings)})
                    planner._progress('delivered', candidate_index=index, approach_s=duration, timings_s=dict(timings))
                    return InterceptPlan(path, ref, target, planner.scene_revision, grasp.copy(), tuple(failures), timings)
                except PickRejected as error:
                    if preparing_scene:
                        raise
                    failed_phase = phase
                    enter_phase('rejected')
                    record = {'candidate_index': index, 'approach_s': duration, 'ik_knot_hz': knot_hz,
                        'candidate_retry_policy':retry_policy,
                        'ik_solve_count': ik_solve_count, 'reason': str(error),
                        'details': error.details, 'candidate_budget_exhausted': clock() >= candidate_deadline,
                        'stop_reason': stop_reason(candidate_deadline), 'failed_phase': failed_phase,
                        'phase_timings_s': dict(attempt_timings),
                        'last_progress': dict(planner.progress),
                        'remaining_send_lead_s': send_deadline-clock()}
                    if coarse_first:
                        record['latest_feasible_send_monotonic_s'] = latest_send_deadline
                        record['candidate_deadline_monotonic_s'] = candidate_deadline
                    if approach:
                        record['relative_descent_s'] = descent
                        record['additional_clearance_m'] = extra_height
                    if timing_option is not None:
                        record['phase_timing_candidate'] = {'join_s': duration, 'follow_s': follow_s,
                                                          'brake_s': brake_s}
                    failures.append(record)
                    planner.attempt_diagnostics.append({**record, 'status': 'rejected', 'wall_s': clock()-attempt_started})
                    if measure_late and error.details.get('reason') == 'measured_deadline_missed':
                        raise PickRejected(str(error), details={'failures': failures,
                            'scene_prepared': scene_prepared}) from error
                    if cancelled() or clock() >= planning_deadline:
                        raise stopped()
                    # Density can help IK, polynomial limits or interpolation
                    # error, but cannot cure a workspace/actual collision reject.
                    retryable = (failed_phase in ('ik', 'polynomial_limits') or
                        (failed_phase == 'sampled_geometry' and str(error) == '关节曲线偏离指定 TCP 参考/方向'))
                    if (approach and extra_height == 0. and failed_phase == 'sampled_geometry'
                            and error.details.get('kind') in ('robot_workpiece_point_cloud','insufficient_alignment_clearance')
                            and not error.details.get('contact_allowed') and not error.details.get('payload')
                            and not abort()):
                        # One higher clearance for a pre-close workpiece hit, after the untouched alternative grasp.
                        work_items.append((index, grasp, [knot_hz], .25*clearance_cache[index]))
                    if coarse_first:
                        if knot_hz == first_hz and fallback_hz != first_hz and retryable and not abort():
                            dense = (index, grasp, [fallback_hz], extra_height)
                            if approach:
                                trials.append((duration,timing_option,descent,(dense,budget_ends[index])))
                            else:
                                work_items.append(dense)
                        break
                    if not retryable or abort():
                        break
    planner._progress('all_smooth_candidates_rejected', planning_wall_s=clock()-begun)
    raise PickRejected('平滑抓姿和持续时间组合均未交付认证轨迹',
                       details={'failures': failures, 'scene_prepared': scene_prepared,
                                **({'phase_timing_search': timing_search} if timing_search is not None else {})})
