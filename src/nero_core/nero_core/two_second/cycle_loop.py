"""Finite or continuous cycles; only rearm after a confirmed return."""
import time


def run_pick_cycles(config, planner, device, vision, *, count, cancelled, result, reports,
                    executors, set_phase, journal, executor_factory, offer_target=None,
                    clock=time.monotonic, finish_attempt=None):
    if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count<1):
        raise ValueError('count must be positive, or None for continuous operation')
    if count is None and finish_attempt is None:
        raise ValueError('continuous operation requires per-attempt trace persistence')
    attempt = 0
    result['recovered_attempts'] = 0
    while count is None or result['completed_feedback_cycles'] < count:
        if cancelled.is_set():
            result['status'] = 'cancelled'
            return
        attempt += 1
        result['attempt_count'] = attempt
        if attempt > 1:
            vision.resume_targets()
        set_phase('waiting_for_qualified_target', cycle_index=attempt)
        last_not_accepted = None
        while not cancelled.is_set():
            target = vision.poll()
            if target is not None and 0 <= clock()-target.capture_s <= config.raw['maximum_observation_age_s']:
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
        vision.consume(target.identity)
        if offer_target is not None:
            offer_target(target)
        set_phase('planning_and_execution', cycle_index=attempt, target_identity=target.identity)
        journal.emit('qualified_target', cycle_index=attempt, identity=target.identity,
            capture_monotonic_s=target.capture_s, received_monotonic_s=target.received_s,
            source_device_s=target.device_s, sequence=target.sequence,
            speed_report=target.speed_report, grasp_candidate_count=len(target.grasps))
        executor = executor_factory(config, planner, device, epoch=lambda: vision.clock_epoch,
            cancelled=cancelled,
            event_sink=lambda row, index=attempt: journal.emit('execution_event', cycle_index=index, record=row))
        executor.cycle_index=attempt
        executors.append(executor)
        cycle_begin = clock()
        report = executor.run(target)
        report['cycle_wall_s'] = clock()-cycle_begin
        reports.append(report)
        if finish_attempt is not None:
            finish_attempt(executor,attempt,report)
        if count is None:
            # Completed traces are persisted by finish_attempt. Retain only
            # the last sender for progress and a bounded summary window.
            del executors[:-1]
            del reports[:-20]
        if report['status'] == 'attempt_cancelled_returned_to_wait':
            if not report.get('returned_to_wait'):
                raise RuntimeError('恢复未确认回到等待位，禁止重新等待抓取')
            result['recovered_attempts'] += 1
            journal.emit('attempt_recovered', cycle_index=attempt, report=report)
            continue
        if report['status'] != 'cycle_completed_from_feedback':
            raise RuntimeError('抓取未完成且没有恢复等待位：'+str(report['status']))
        if not report.get('returned_to_wait'):
            raise RuntimeError('本轮未确认回到等待位，禁止继续下一件')
        result['completed_feedback_cycles'] += 1
        journal.emit('cycle_finished', cycle_index=attempt,
            completed_feedback_cycles=result['completed_feedback_cycles'], report=report)
    result['status'] = 'requested_feedback_cycles_completed'
