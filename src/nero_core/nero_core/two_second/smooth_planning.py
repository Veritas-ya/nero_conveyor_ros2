"""Duration/candidate scheduling for measured smooth interception."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .precise_ik import PrecisePoseSolver
from .smooth_reference import SmoothIntercept
from .synchronized_reference import (SynchronizedIntercept, schedule_forward_join,
                                      admit_synchronized_observation, freeze_window_for_speed)
from .reference import predict_entry_time, DEFAULT_FREEZE_UPSTREAM_DISTANCE_M
from .trajectory import sampling_times, certify_curve, extrema


def order_obb_grasps(grasps, start_rotation, *, synchronized):
    """Only short-axis poses, including legacy packets with long-axis extras."""
    indexed = list(enumerate(grasps))[:2]
    indexed.sort(key=lambda item: Rotation.from_matrix(start_rotation.T@item[1][:3, :3]).magnitude())
    return indexed


def plan_smooth_intercept(planner, target, state, *, cancelled=lambda: False,
                          reservation_s=None, measure_late=False, global_deadline=None):
    from .planning import InterceptPlan
    cfg, model, clock = planner.config, planner.model, planner.clock
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
    if synchronized:
        freeze_window = freeze_window_for_speed(cfg, target.velocity@cfg.direction)
        upstream_distance = freeze_window['distance_m']
        entry = admit_synchronized_observation(target.capture_s, entry_position@cfg.direction,
            target.velocity@cfg.direction, cfg.follow_bounds, begun, upstream_distance)
    else:
        entry = predict_entry_time(target.capture_s, entry_position@cfg.direction,
            target.velocity@cfg.direction, cfg.follow_bounds[0], begun, cfg.raw['planning_reserve_s']+prealign_s,
            maximum_upstream_distance_m=upstream_distance)
    model.check_feedback(state)
    start = model.tcp().copy()
    solver = PrecisePoseSolver(model)
    send_deadline = float('inf') if synchronized else entry-prealign_s-cfg.raw['command_advance_s']
    planning_deadline = (begun+cfg.raw['planning_budget_s'] if global_deadline is None else global_deadline)
    deadline = min(planning_deadline, send_deadline)
    failures, scene_prepared = [], False
    timings = {'scene_prepare': 0.}
    def stop_reason(candidate_deadline=None):
        now = clock()
        if cancelled():
            return 'cancelled'
        if not measure_late and now >= send_deadline:
            return 'send_deadline'
        if now >= planning_deadline:
            return 'global_planning_budget'
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
    indexed_grasps = order_obb_grasps(target.grasps, start[:3, :3], synchronized=synchronized)
    grasps = [grasp for _, grasp in indexed_grasps]
    durations=(cfg.raw.get('synchronized_join_candidates_s', [prealign_s]) if synchronized
               else cfg.raw['approach_duration_candidates_s'])
    rotation_cache = {}
    for duration in durations:
        for index, grasp in enumerate(grasps):
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
                    schedule = schedule_forward_join(capture_s=target.capture_s,
                        grasp_position=grasp[:3,3], velocity=target.velocity,
                        standby_position=start[:3,3], direction=cfg.direction,
                        now_s=clock(), planning_reserve_s=reservation_s,
                        advance_s=cfg.raw['command_advance_s'], join_s=duration, rotate_first_s=rotate_first_s,
                        synchronized_s=synchronized_s, follow_s=cfg.settings.full_plan_close_window_s,
                        follow_bounds=cfg.follow_bounds)
                except ValueError as error:
                    failures.append({'candidate_index': index, 'approach_s':duration, 'reason': str(error),
                                     'failed_phase': 'synchronized_schedule'})
                    continue
                send_deadline = schedule['join_start_s']-cfg.raw['command_advance_s']
                deadline = planning_deadline if measure_late else min(planning_deadline, send_deadline)
            if cancelled() or clock() >= deadline:
                raise stopped()
            attempt_started = clock()
            candidate_deadline = min(deadline, attempt_started+cfg.raw['candidate_planning_budget_s'])
            first_hz = cfg.raw.get('planning_knot_hz', min(50., cfg.settings.control_hz))
            fallback_hz = cfg.raw.get('planning_knot_hz_fallback', first_hz)
            knot_rates = [first_hz] if fallback_hz == first_hz else [first_hz, fallback_hz]
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
                ref = reference_type(target.capture_s, start_time, reference_duration+effective_prealign, grasp[:3, 3], target.velocity,
                    start[:3, 3], Rotation.from_matrix(start[:3, :3]).as_quat()[[3, 0, 1, 2]],
                    Rotation.from_matrix(grasp[:3, :3]).as_quat()[[3, 0, 1, 2]],
                    cfg.settings.full_plan_close_window_s, cfg.raw['brake_duration_s'], effective_prealign,
                    **({'rotate_first_s':rotate_first_s} if synchronized else {}))
                try:
                    planner._progress('smooth_candidate_bounds', candidate_index=index, approach_s=duration)
                    contact = ref.contact_xyz@cfg.direction
                    follow_end = (ref.contact_xyz+target.velocity*ref.follow_s)@cfg.direction
                    if min(contact, follow_end) < cfg.follow_bounds[0] or max(contact, follow_end) > cfg.follow_bounds[1]:
                        raise PickRejected('平滑接触和跟随段超出抓取区间')
                    times = sampling_times(ref.full_duration_s, 1./knot_hz,
                                           rotate_first_s, ref.prealign_s, ref.duration_s, ref.follow_end_s)
                    xyzs = np.array([ref.sample(t)[0] for t in times])
                    if np.any(xyzs < cfg.workspace[:, 0]) or np.any(xyzs > cfg.workspace[:, 1]):
                        raise PickRejected('平滑接近/跟随/减速超出工作区')
                    enter_phase('ik')
                    joints = [np.asarray(state['joints']).copy()]
                    for i, t in enumerate(times[1:], 1):
                        planner._progress('smooth_intercept_ik', candidate_index=index, approach_s=duration,
                            waypoint_index=i, waypoint_count=len(times)-1, trajectory_time_s=float(t),
                            candidate_wall_s=clock()-attempt_started,
                            remaining_budget_s=max(0., candidate_deadline-clock()))
                        ik_solve_count += 1
                        solved = solver.solve(*ref.sample(t), joints[-1], cancelled=abort)
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
                        planner.install(target, state)
                        timings['scene_prepare'] = clock()-check_started
                        candidate_deadline = min(deadline, candidate_deadline+timings['scene_prepare'])
                        scene_prepared = True
                        preparing_scene = False
                    path = replace(path, version=planner.version)
                    enter_phase('sampled_geometry')
                    planner.candidate_context = {'candidate_index':index, 'approach_s':duration,
                        'input_grasp_index':next((i for i,g in enumerate(target.grasps) if g is grasp),None),
                        'selected_grasp_tcp_base':grasp.tolist()}
                    check_started = clock()
                    reference_check = planner.check_smooth_intercept(path, ref, cancelled=abort)
                    timings['sampled_geometry'] = clock()-check_started
                    if abort():
                        raise PickRejected('平滑轨迹未在实际发送截止前完成认证')
                    if measure_late and clock() >= send_deadline:
                        raise PickRejected('规划完成时已错过预约发送，必须重新规划', details={
                            'reason': 'measured_deadline_missed',
                            'measured_candidate_wall_s': clock()-attempt_started,
                            'missed_by_s': clock()-send_deadline})
                    diagnostics = {'profile': 'smooth', 'ik_knot_hz': knot_hz, 'ik_solve_count': ik_solve_count,
                        'reference_check': reference_check, 'approach_s': duration, 'prealign_s': effective_prealign,
                        'rotate_first_s':rotate_first_s if synchronized else prealign_s,
                        'grasp_axis_policy':'obb_short_axis_only',
                        'translation_start_elapsed_s':rotate_first_s if synchronized else prealign_s,
                        'follow_s': ref.follow_s,
                        'brake_s': ref.brake_s, 'rest_qva_endpoints': True,
                        'peak_velocity_rad_s': np.max(abs(np.asarray(extrema(path.curve, 1))), axis=0).tolist(),
                        'peak_acceleration_rad_s2': np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0).tolist(),
                        'brake_geometry': 'visual_attachment_precheck_and_live_measured_payload_edges'}
                    if synchronized:
                        diagnostics.update(intercept_strategy='synchronized', synchronized_schedule=schedule,
                                           synchronized_s=synchronized_s, join_s=duration,
                                           rotation_timing=rotation_timing,
                                           freeze_window=freeze_window,
                                           selected_input_grasp_index=indexed_grasps[index][0])
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
                        'ik_solve_count': ik_solve_count, 'reason': str(error),
                        'details': error.details, 'candidate_budget_exhausted': clock() >= candidate_deadline,
                        'stop_reason': stop_reason(candidate_deadline), 'failed_phase': failed_phase,
                        'phase_timings_s': dict(attempt_timings),
                        'last_progress': dict(planner.progress),
                        'remaining_send_lead_s': send_deadline-clock()}
                    failures.append(record)
                    planner.attempt_diagnostics.append({**record, 'status': 'rejected', 'wall_s': clock()-attempt_started})
                    if measure_late and error.details.get('reason') == 'measured_deadline_missed':
                        raise PickRejected(str(error), details={'failures': failures,
                            'scene_prepared': scene_prepared}) from error
                    if cancelled() or clock() >= planning_deadline:
                        raise stopped()
                    # A sparse curve can deviate between IK nodes or violate q/v/a;
                    # retry the SAME candidate at the configured dense grid while
                    # retaining its original budget and the global send deadline.
                    retryable = (failed_phase in ('ik', 'polynomial_limits') or
                        (failed_phase == 'sampled_geometry' and str(error) == '关节曲线偏离指定 TCP 参考/方向'))
                    if not retryable or abort():
                        break
    planner._progress('all_smooth_candidates_rejected', planning_wall_s=clock()-begun)
    raise PickRejected('平滑抓姿和持续时间组合均未交付认证轨迹',
                       details={'failures': failures, 'scene_prepared': scene_prepared})
