"""Single-owner cycles with confirmed return or empty-gripper bin handoff."""
import time


def run_pick_cycles(config, planner, device, vision, *, count, cancelled, result, reports,
                    executors, set_phase, journal, executor_factory, offer_target=None,
                    clock=time.monotonic, finish_attempt=None, lookahead=None, buffered_freeze=False,
                    direct_bin_pick=False, next_plans=None, save_data=True):
    if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count<1):
        raise ValueError('count must be positive, or None for continuous operation')
    if count is None and finish_attempt is None and save_data:
        raise ValueError('continuous operation requires per-attempt trace persistence')
    attempt = 0
    handoff = None
    handoff_enabled = direct_bin_pick or next_plans is not None
    result['recovered_attempts'] = 0
    while count is None or result['completed_feedback_cycles'] < count:
        if cancelled.is_set():
            result['status'] = 'cancelled'
            return
        attempt += 1
        result['attempt_count'] = attempt
        if attempt > 1 and handoff is None:
            vision.resume_targets()
        set_phase('waiting_for_qualified_target', cycle_index=attempt)
        last_not_accepted = None
        prefetched = None
        frozen_entry = None
        while not cancelled.is_set():
            if handoff is not None:
                frozen_entry = handoff.entry
                target = frozen_entry.target
                break
            elif buffered_freeze:
                frozen_entry = vision.claim_frozen()
                if frozen_entry is not None:
                    target = frozen_entry.target
                    break
                target = None
            elif lookahead is not None and attempt > 1:
                # Read-only mailbox publication on the owning thread. A busy
                # parser supplies no update; cached timestamps still expire.
                snapshot = getattr(device, 'snapshot_nowait', device.snapshot)()
                if snapshot is not None:
                    lookahead.update_wait_state(snapshot)
                prefetched = lookahead.poll_ready()
                target = None if prefetched is None else prefetched.target
            else:
                target = vision.poll()
            if target is not None and (prefetched is not None or
                    0 <= clock()-target.capture_s <= config.raw['maximum_observation_age_s']):
                if prefetched is not None:
                    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
                    try:
                        lookahead.claim(prefetched, planner)
                    except PickRejected as error:
                        lookahead.discard_ready(prefetched, error)
                        continue
                    planner = prefetched.planner
                break
            if target is not None and target.sequence != last_not_accepted:
                journal.emit('ready_target_not_accepted', cycle_index=attempt,
                    identity=target.identity, observation_age_s=clock()-target.capture_s,
                    maximum_age_s=config.raw['maximum_observation_age_s'])
                last_not_accepted = target.sequence
            cancelled.wait(.01)
        else:
            result['status'] = 'cancelled'
            return
        if cancelled.is_set():
            result['status']='cancelled'
            return
        if frozen_entry is None:
            vision.consume(target.identity)
        if lookahead is not None:
            lookahead.begin_cycle(target.identity)
        if offer_target is not None:
            offer_target(target)
        set_phase('planning_and_execution', cycle_index=attempt, target_identity=target.identity)
        capture_position = getattr(target, 'entry_position', None)
        capture_upstream = (None if capture_position is None else
            float(config.follow_bounds[0]-sum(p*d for p,d in zip(capture_position, config.direction))))
        journal.emit('qualified_target', cycle_index=attempt, identity=target.identity,
            capture_monotonic_s=target.capture_s, received_monotonic_s=target.received_s,
            source_device_s=target.device_s, sequence=target.sequence,
            speed_report=target.speed_report, grasp_candidate_count=len(target.grasps),
            freeze_capture_upstream_m=capture_upstream,
            freeze_position_upstream_m=config.raw.get('freeze_position_upstream_m'),
            freeze_position_tolerance_m=config.raw.get('freeze_position_tolerance_m'))
        executor = executor_factory(config, planner, device, epoch=lambda: vision.clock_epoch,
            cancelled=cancelled,
            event_sink=lambda row, index=attempt: journal.emit('execution_event', cycle_index=index, record=row))
        executor.cycle_index=attempt
        if frozen_entry is not None:
            executor.frozen_entry = frozen_entry
        if handoff is not None:
            executor.incoming_handoff = handoff
            if handoff.origin == 'bin':
                executor.bin_start = handoff
        handoff = None
        if handoff_enabled and (count is None or result['completed_feedback_cycles'] + 1 < count):
            executor.claim_after_release = vision.claim_frozen
            if next_plans is not None:
                executor.next_plans = next_plans
                executor.pending_targets = vision.pending_candidates
        if lookahead is not None:
            executor.return_started = lookahead.return_started
            executor.prefetched_intercept = prefetched
        executors.append(executor)
        cycle_begin = clock()
        report = executor.run(target)
        if (lookahead is not None and report.get('returned_to_wait') and
                report['status'] in ('attempt_cancelled_returned_to_wait', 'cycle_completed_from_feedback')):
            lookahead.confirm_return(device.snapshot())
        report['cycle_wall_s'] = clock()-cycle_begin
        reports.append(report)
        if finish_attempt is not None:
            finish_attempt(executor,attempt,report)
        if count is None or not save_data:
            # Only the last sender and recent cycle outcomes are needed for live progress.
            del executors[:-1]
            del reports[:-20]
        if config.raw.get('hover_descent', {}).get('enabled', False) and count is not None:
            # Finite staged tests stop after ONE exposure, including a rejected
            # descent. Explicit continuous runs use the common return/rearm
            # checks below and persist every attempt before accepting the next.
            if not report.get('returned_to_wait'):
                raise RuntimeError('分段测试未确认回等待位')
            if report['status'] == 'cycle_completed_from_feedback':
                result['completed_feedback_cycles'] += 1
            elif report['status'] == 'attempt_cancelled_returned_to_wait':
                result['recovered_attempts'] += 1
            else:
                raise RuntimeError('分段测试返回未知状态：'+str(report['status']))
            result['status'] = 'hover_test_attempt_completed'
            journal.emit('hover_test_attempt_completed', cycle_index=attempt, report=report)
            return
        if report['status'] == 'attempt_cancelled_returned_to_wait':
            if not report.get('returned_to_wait'):
                raise RuntimeError('恢复未确认回到等待位，禁止重新等待抓取')
            result['recovered_attempts'] += 1
            journal.emit('attempt_recovered', cycle_index=attempt, report=report)
            continue
        if report['status'] != 'cycle_completed_from_feedback':
            raise RuntimeError('抓取未完成且没有恢复等待位：'+str(report['status']))
        if not report.get('returned_to_wait') or report.get('next_target_identity') is not None:
            handoff = getattr(executor, 'next_bin_pick', None)
            if (not handoff_enabled or handoff is None or handoff.previous_identity != target.identity
                    or handoff.entry.target.identity != report.get('next_target_identity')
                    or handoff.origin not in ('bin', 'wait')
                    or (handoff.origin == 'bin' and not direct_bin_pick)
                    or (handoff.origin == 'wait') != bool(report.get('returned_to_wait'))):
                raise RuntimeError('本轮未确认回到等待位或交接下一件，禁止继续抓取')
        result['completed_feedback_cycles'] += 1
        journal.emit('cycle_finished', cycle_index=attempt,
            completed_feedback_cycles=result['completed_feedback_cycles'], report=report)
    result['status'] = 'requested_feedback_cycles_completed'
