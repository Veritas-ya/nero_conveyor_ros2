"""Concise terminal steps, rendered by the journal writer, never a sender."""
import math
from pathlib import Path


def short(value, limit=160):
    text = ' '.join(str(value).split())
    return text if len(text) <= limit else text[:limit]+'…'


def step_name(stage):
    return {
        'loading_dependencies': '加载依赖', 'model_startup': '加载模型',
        'can_startup': '连接机械臂', 'startup_readiness': '等待位与夹爪检查',
        'camera_startup': '启动相机', 'waiting_for_qualified_target': '检测与测速',
        'checking_start_state': '抓取前状态检查', 'planning_intercept': '抓取轨迹规划',
        'smooth_intercept_follow_brake': '接近、跟随与减速',
        'intercept_follow': '接近与跟随', 'planning_lift': '抬升轨迹规划',
        'planning_transfer': '搬运轨迹规划', 'planning_return': '返回轨迹规划',
        'lift': '抬升', 'transfer': '搬运', 'release': '张爪投放',
        'transfer_connect': '接入搬运示教起点', 'taught_transfer': '示教搬运至投放位',
        'return_connect': '接入返回示教起点', 'taught_return': '示教返回',
        'return': '返回等待位', 'waiting': '等待下一件',
        'recovery_checking': '恢复状态检查', 'recovery_settling': '恢复前位置保持',
        'planning_recovery_lift': '恢复抬升规划', 'recovery_lift': '恢复抬升',
        'planning_recovery_transfer': '恢复入框规划', 'recovery_transfer': '恢复入框',
        'recovery_release': '恢复投放', 'planning_recovery_return': '恢复回位规划',
        'recovery_return': '恢复回等待位', 'recovery_open_empty': '回位后张爪',
    }.get(stage, short(stage or '启动'))


class ConsoleProgress:
    """Only milestones and low-rate waiting status; full evidence stays on disk.

    observe() runs in RunJournal's background writer. It has no device access,
    no motion decisions, and never runs on the 100 Hz command thread.
    """
    def __init__(self, *, direction=(1., 0., 0.), write=None):
        self.direction = tuple(float(x) for x in direction)
        self.write = write or (lambda line: print(line, flush=True))
        self.total = self.cycle = 1
        self.last_wait = None
        self.last_wait_s = -math.inf
        self.pending_wait, self.pending_wait_s = None, -math.inf
        self.failed = self.grip_reported = False
        self.fixed_speed = None
        self.planning_start_rotation = None

    def observe(self, row):
        event, data = row['event'], row['data']
        now = row['monotonic_s']
        if event == 'run_started':
            self.total = data.get('requested_count', 1)
            velocity_policy = data.get('velocity_policy') or {}
            self.fixed_speed = (velocity_policy.get('fixed_belt_speed_m_s')
                if velocity_policy.get('source')=='operator_fixed_belt_speed' else None)
            start = ('[启动] 连续真机抓取，每轮回等待位；Ctrl+C 停止' if data.get('continuous') else
                     f'[启动] 真机抓取，共 {self.total} 件；Ctrl+C 停止')
            if data.get('taught_arguments_ignored'):
                start += '；示教暂停，忽略示教文件，自动规划投放'
            elif data.get('taught_segments'):
                start += '；使用示教投放'
            elif 'taught_playback_enabled' in data:
                start += '；自动规划投放'
            self.write(start)
            if data.get('environment_collision_scope') == 'measured_belt_surface_only':
                self.write('[避障] 环境仅检查实测带面点云；保留自碰撞')
            if data.get('grasp_workpiece_collision_check_enabled', True):
                self.write('[工件] 抓取阶段检查机器人与目标工件点云碰撞')
            else:
                self.write('[工件] 抓取阶段不检查机器人与目标工件点云碰撞')
        elif event == 'phase_started':
            phase = data['phase']
            if phase == 'loading_dependencies':
                self.write('[启动] 加载算法与模型')
            elif phase == 'can_startup':
                self.write(f"[连接] 机械臂 {data.get('can_channel', '')}")
            elif phase == 'startup_readiness':
                self.write('[检查] 等待位、夹爪与反馈')
            elif phase == 'camera_startup':
                self.write('[相机] 启动 RGB-D 观测')
            elif phase == 'waiting_for_qualified_target':
                self.cycle = data.get('cycle_index', 1)
                self.last_wait, self.last_wait_s = 'waiting', now
                self.pending_wait, self.pending_wait_s = None, now
                self.grip_reported = False
                action = '开始检测与定位' if self.fixed_speed is not None else '开始检测与视觉测速'
                self.write(f'[等待] 第 {self.cycle} 次抓取尝试，{action}')
        elif event == 'startup_return_wait_started':
            self.write(f"[回位] 平滑移动到等待位，速度 {data['speed_percent']}%")
        elif event in ('startup_return_wait_passed', 'startup_return_wait_skipped'):
            self.write('[回位] 等待位已确认')
        elif event == 'target_sent':  # standalone move_to_wait
            self.write(f"[回位] 正在前往等待位，速度 {data['speed_percent']}%")
        elif event == 'runtime_progress':
            self._waiting(data, now)
        elif event == 'collision_detected':
            elapsed=data.get('trajectory_elapsed_s')
            when='' if elapsed is None else f'，轨迹 {elapsed:.3f} s'
            self.write(f'[碰撞] {data["geometry"]}{when}；记录 {data["collision_id"]}')
        elif event == 'preview_requested':
            self.write('[画面] 打开同源相机与碰撞预览')
        elif event == 'qualified_target':
            speed = data.get('speed_report') or {}
            velocity = speed.get('measured_velocity_base_m_s')
            if speed.get('source')=='operator_fixed_belt_speed':
                self.write(f'[带速] 用户设定 {speed["configured_speed_m_s"]*100:.2f} cm/s，本次预测不使用视觉测速')
            elif velocity is not None and len(velocity) == 3:
                along = sum(float(v)*d for v, d in zip(velocity, self.direction))
                samples, span = speed.get('sample_count', '?'), speed.get('span_s')
                span_text = f'/{span:.2f} s' if isinstance(span, (float, int)) else ''
                self.write(f'[测速] 已冻结 {along*100:.2f} cm/s（卡尔曼，{samples} 帧{span_text}）')
            else:
                self.write('[测速] 目标测速已就绪')
            count = data.get('grasp_candidate_count', '?')
            self.write(f'[抓姿] OBB 已生成 {count} 个候选')
        elif event == 'execution_event':
            self._execution(data['record'])
        elif event == 'cycle_finished':
            completed = data.get('completed_feedback_cycles', data['cycle_index'])
            progress=str(completed) if self.total is None else f'{completed}/{self.total}'
            self.write(f'[完成] 第 {progress} 件动作与反馈周期结束，已回等待位')
        elif event == 'attempt_recovered':
            self.write('[恢复] 本次抓取已取消，已回等待位，等待下一件')
        elif event == 'run_failed':
            self.failed = True
            stage = step_name(data.get('execution_stage') or data.get('phase'))
            self.write(f'[停止] {stage}：{short(data.get("error", "未知原因"))}')

    def _waiting(self, data, now):
        if data.get('phase') != 'waiting_for_qualified_target':
            return
        vision = data.get('vision') or {}
        stage = vision.get('stage', '')
        entry = vision.get('entry_gate') or {}
        last_blocker = vision.get('last_blocker') or {}
        geometry = vision.get('grasp_geometry') or {}
        geometry_blocker = geometry.get('last_blocker') or {}
        calibration = (vision.get('perception') or {}).get('plane_calibration') or {}
        # Vision processing changes stage many times between one-second
        # snapshots. A recent blocker is more informative than the transient
        # waiting_rgbd stage, but only while a target is still tracked.
        current_time = vision.get('snapshot_monotonic_s', now)
        def recent(blocker):
            stamp = blocker.get('monotonic_s')
            return isinstance(stamp, (int,float)) and 0 <= current_time-stamp <= 2.
        entry_reason = entry.get('reason')
        if (vision.get('tracks') and last_blocker.get('stage') == 'waiting_upstream_window'
                and recent(last_blocker)):
            entry_reason = last_blocker.get('reason')
        if calibration and calibration.get('ready') is False:
            key='belt_calibration'
            message=f'[相机] 实测带面初始化 {calibration.get("frames",0)}/{calibration.get("required_frames","?")} 帧，完成后检测工件'
        elif last_blocker.get('reason')=='invalid_capture_time' and recent(last_blocker):
            key, message='camera_timestamp','[相机] 本帧时间戳无效，已丢弃并等待有效新画面'
        elif stage == 'waiting_upstream_window' or (vision.get('tracks') and entry_reason in ('too_far_upstream','already_at_entry','insufficient_planning_time')):
            reason = entry_reason
            key, message = reason, {
                'too_far_upstream': '[等待] 目标仍在上游，等待进入规划触发范围',
                'already_at_entry': '[等待] 当前目标已过抓取入口，等待下一件',
                'insufficient_planning_time': '[等待] 当前目标入区前的规划时间不足，等待下一件',
            }.get(reason, '[等待] 等待合适的抓取触发时刻')
        elif (vision.get('tracks') and recent(geometry_blocker) and
              geometry_blocker.get('reason') in ('observation_stale_after_detection','observation_stale_at_delivery')):
            key, message = 'geometry_stale', '[等待] 已检测工件，但点云处理后观测过期，等待新画面'
        elif geometry.get('stage') in ('detecting_target', 'generating_obb'):
            key, message = 'grasp', '[抓姿] 提取目标点云并计算 OBB 抓姿'
        elif stage in ('detecting_target', 'generating_obb', 'waiting_target_geometry',
                       'waiting_target_association'):
            key, message = 'grasp', '[抓姿] 提取目标点云并计算 OBB 抓姿'
        elif stage == 'fitting_belt_surface':
            key, message = 'belt', '[相机] 检测并拟合传送带面'
        elif stage == 'waiting_feedback':
            key, message = 'camera', '[相机] 等待新画面与配对反馈'
        elif not vision.get('frames_received'):
            key, message = 'camera', '[相机] 等待首帧与配对反馈'
        elif not vision.get('tracks'):
            key, message = 'waiting', '[等待] 等待检测到工件'
        else:
            key, message = ('position', '[定位] 等待确认独立工件位置与身份') if self.fixed_speed is not None else (
                'speed', '[测速] 正在积累视觉观测，等待速度有效')
        # Stage snapshots may alternate every frame. Require a persistent
        # change; during long waits give one short update every ten seconds.
        if key != self.pending_wait:
            self.pending_wait, self.pending_wait_s = key, now
        if (key != self.last_wait and now-self.pending_wait_s >= 1. and
                now-self.last_wait_s >= 2. or
                now-self.last_wait_s >= 10.):
            self.write(message)
            self.last_wait, self.last_wait_s = key, now

    def _execution(self, row):
        event, data = row['event'], row['data']
        if event == 'attempt_cancelled':
            self.write('[取消] '+short(data['error'])+'；正在检查恢复路径')
            route = data.get('details', {}).get('route_diagnostics', {})
            examples = route.get('rejection_examples', [])
            if examples:
                first = examples[0]
                goal_label = '示教起点' if route.get('stage') in ('transfer_connect','return_connect') else '目标点'
                role = {'start':'连接起点', 'goal':goal_label, 'path':'搜索路径'}.get(first.get('role'), '轨迹')
                self.write(f'[诊断] {role}检查未通过：{short(first.get("reason", "未知原因"))}')
            elif route:
                outcome = {'no_ik_solution':'目标逆解未收敛', 'planning_timeout':'规划超时',
                    'cancelled':'规划已取消', 'no_certified_route':'路径或轨迹认证未通过',
                    'payload_estimate_missing':'缺少载物点云估计',
                    'start_feedback_mismatch':'起点与反馈不一致'}.get(route.get('outcome'), '未交付路径')
                self.write(f'[诊断] 连接规划结果：{outcome}')
        elif event == 'recovery_workpiece_contact_policy' and data.get('allow_cancelled_workpiece_contact'):
            self.write('[恢复] 允许接触本次已取消的轻质工件，准备回等待位')
        elif event == 'recovery_feedback_wait':
            self.write('[恢复] 等待新反馈与最后目标到位，暂不发送回位指令')
        elif event == 'recovery_feedback_rechecked':
            self.write(f'[恢复] 新反馈复核通过，等待 {data["waited_s"]:.3f}s')
        elif event == 'recovery_started':
            self.write('[恢复] 先放入配置料框，再回等待位' if data['carrying'] else '[恢复] 规划空爪返回等待位')
        elif event == 'recovery_finished':
            self.write('[恢复] 已回等待位并张爪')
        elif event == 'planning_submitted':
            pose = data.get('start_tcp_base')
            self.planning_start_rotation = None if pose is None else [row[:3] for row in pose[:3]]
            self.write('[轨迹] 正在求解并检查抓取轨迹')
        elif event == 'planning_delivery':
            pose = data.get('selected_grasp_tcp_base')
            if pose is not None:
                # XYZ extrinsic roll/pitch/yaw, same convention as the existing
                # geometry helper; the complete matrix remains in events.jsonl.
                pitch = math.atan2(-pose[2][0], math.hypot(pose[0][0], pose[1][0]))
                if abs(math.cos(pitch)) > 1e-8:
                    roll, yaw = math.atan2(pose[2][1], pose[2][2]), math.atan2(pose[1][0], pose[0][0])
                else:
                    roll, yaw = math.atan2(-pose[1][2], pose[1][1]), 0.
                rpy = ', '.join(f'{math.degrees(a):.1f}' for a in (roll, pitch, yaw))
                rotation_text = ''
                if self.planning_start_rotation is not None:
                    product = sum(self.planning_start_rotation[i][j]*pose[i][j] for i in range(3) for j in range(3))
                    angle = math.degrees(math.acos(max(-1.,min(1.,(product-1.)/2.))))
                    diagnostics=data.get('trajectory_diagnostics', {})
                    rotation_s=diagnostics.get('rotate_first_s',0. if diagnostics.get('intercept_strategy')=='synchronized' else diagnostics.get('prealign_s',0.))
                    if rotation_s > 0:
                        rotation_text = f'；接近前计划原位旋转 {angle:.1f}°（{rotation_s:.2f} s）'
                    else:
                        rotation_text = f'；接近过程中计划旋转 {angle:.1f}°'
                self.write(f'[抓姿] 已选定 OBB 姿态，基座系 RPY=({rpy})°{rotation_text}')
            point = data.get('predicted_contact_tcp_base_m')
            point_text = '' if point is None else '；预计接触 TCP=('+', '.join(f'{float(x)*1000:.1f}' for x in point)+') mm（基座系）'
            self.write(f'[轨迹] 已得到，规划 {data["planning_wall_s"]:.2f} s{point_text}')
        elif event == 'planning_delivery_missed':
            self.write('[轨迹] 交付时已错过发送起点，根据实测耗时重新规划')
        elif event == 'stage_started':
            stage = data['stage']
            if stage in ('smooth_intercept_follow_brake', 'intercept_follow'):
                message = ('[抓取] 等待预约起点，先转向再接近与跟随'
                           if data.get('rotate_first_s', data.get('prealign_s',0.)) > 0 else
                           '[抓取] 等待预约起点，执行接近与跟随')
                self.write(message)
            else:
                label = {'lift': '抬升', 'transfer': '搬运', 'return': '返回'}.get(stage, '执行')
                self.write(f'[{label}] {step_name(stage)}轨迹开始')
        elif event == 'close_during_follow':
            self.write('[夹爪] 按计划时刻闭合并继续跟随' if data.get('closure_policy') == 'scheduled_attempt' else '[夹爪] 跟随闭合，等待夹持反馈')
        elif event in ('payload_estimate_attached', 'grip_confirmed_from_width_force'):
            if not self.grip_reported:
                self.write('[夹持] 宽度与受力确认通过')
                self.grip_reported = True
        elif event == 'route_submitted':
            self.write(f'[轨迹] 正在规划{step_name(data["stage"])}路径')
        elif event == 'route_delivered':
            self.write(f'[轨迹] {step_name(data["stage"])}路径已得到，规划 {data["planning_wall_s"]:.2f} s')
        elif event == 'release_open_requested':
            self.write('[投放] 已发送张爪，等待开口反馈')

    def finish(self, result, output):
        if result['status'] == 'requested_feedback_cycles_completed':
            self.write(f'[结束] 完成 {result["completed_feedback_cycles"]}/{result["requested_count"]} 个反馈周期；实物入框需现场确认')
        elif result['status'] == 'attempt_completed_grip_unconfirmed':
            self.write('[结束] 已完成抓取尝试和减速；夹持未确认，停在轨迹终点')
        elif not self.failed:
            if result['status'] == 'cancelled':
                self.write(f'[停止] 已取消；已完成 {result["completed_feedback_cycles"]} 轮，'
                    f'失败后回位 {result.get("recovered_attempts",0)} 次')
            else:
                runtime = result.get('last_runtime') or {}
                stage = step_name(runtime.get('execution_stage') or runtime.get('phase'))
                self.write(f'[停止] {stage}：{short(result.get("error", result["status"]))}')
        if (result.get('log_save_errors') or result.get('execution_event_logging_errors') or
                any(result.get('visualization',{}).get(key) for key in ('errors','dropped','writer_alive')) or
                result.get('collision_diagnostic_errors') or
                any(result.get('journal', {}).get(key) for key in ('errors', 'dropped', 'writer_alive', 'observer_errors'))):
            self.write('[日志] 部分记录保存或终端显示异常，请检查输出目录与写盘状态')
        if (result.get('preview_cleanup_error') or
                result.get('preview_process_exit_code') not in (None,0) and not result.get('preview_closed_by_parent')):
            self.write('[画面] 预览进程异常退出，详见 preview/viewer.log')
        self.write(f'[日志] {Path(output).resolve()}')
