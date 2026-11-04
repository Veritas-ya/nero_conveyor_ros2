"""Human-readable CLI progress and wall-clock timing; never controls the robot."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading
import time


STAGES = {
    "idle": "等待可用抓取目标", "selecting": "选择抓取目标",
    "pregrasp": "接近工件上方（含规划等待）",
    "tracking_descent": "下降跟随", "closing": "闭合夹爪",
    "lifting": "夹住后抬升", "transferring": "原位放下",
    "releasing": "松开夹爪", "returning": "退到工件上方",
    "recovering": "流程未完成", "complete": "流程完成",
}
REASONS = {
    "awaiting_grasp_geometry": "等待工件识别和目标关联",
    "no_observation": "还没有目标观测", "target_not_observed": "暂未检测到目标",
    "velocity_not_observed": "等待有效速度样本",
    "velocity_observation_expired": "目标速度观测过期",
    "observation_expired": "目标位置观测过期",
    "lifting_timeout": "夹住后抬升超时", "pregrasp_timeout": "接近工件上方超时",
    "grasp_not_confirmed": "尚未确认夹住", "empty_grasp": "闭合后未夹到工件",
    "object_dropped": "夹持反馈丢失",
}


class RunProgress:
    """Small independent timeline. Concurrent durations must not be summed."""

    def __init__(self, *, started=None, clock=time.monotonic, writer=None, verbose=False):
        self.clock = clock
        self.started = clock() if started is None else float(started)
        self.writer = writer or (lambda line: print(line, flush=True))
        self.verbose = verbose
        self.directory = None
        self.phases = []
        self.milestones = {}
        self.planning = []
        self.stages = []
        self.components = []
        self._active = {}
        self._stage = None
        self._stage_name = None
        self._last_wait = -math.inf
        self._last_visual_error = -math.inf
        self._last_error = None
        self._preview_shown = False
        self._terminal = False
        self._standby = False
        self._lock = threading.RLock()

    def elapsed(self):
        return max(0., self.clock() - self.started)

    def say(self, message):
        with self._lock:
            self.writer(f"[启动+{self.elapsed():7.2f}s] {message}")

    def begin(self, key, label, *, lane="main", announce=True, at_monotonic=None):
        with self._lock:
            start = self.elapsed() if at_monotonic is None else max(0., at_monotonic-self.started)
            row = {"name": key, "label": label, "lane": lane, "start_s": start,
                   "end_s": None, "duration_s": None, "status": "running"}
            self.phases.append(row)
            self._active[key] = row
            if announce:
                self.say(f"{label}…")
            return row

    def end(self, key, *, status="completed", announce=True, at_monotonic=None):
        with self._lock:
            row = self._active.pop(key, None)
            if row is None:
                return
            end = self.elapsed() if at_monotonic is None else max(0., at_monotonic-self.started)
            row.update(end_s=end, status=status)
            row["duration_s"] = max(0., row["end_s"] - row["start_s"])
            if announce:
                suffix = "完成" if status == "completed" else "中断"
                self.say(f'{row["label"]}{suffix}，用时 {row["duration_s"]:.2f}s')

    @contextmanager
    def phase(self, key, label, *, lane="main", announce=True):
        self.begin(key, label, lane=lane, announce=announce)
        try:
            yield
        except BaseException:
            self.end(key, status="interrupted", announce=announce)
            raise
        else:
            self.end(key, announce=announce)

    def milestone(self, key, label=None, *, data=None):
        with self._lock:
            if key in self.milestones:
                return
            self.milestones[key] = {"elapsed_s": self.elapsed(), **(data or {})}
            if label:
                self.say(label)

    def interrupt_active_phases(self):
        with self._lock:
            for key in list(self._active):
                self.end(key, status="interrupted", announce=False)

    @staticmethod
    def _speed(value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            speed = math.hypot(float(value[0]), float(value[1])) * 1000
        except (TypeError, ValueError):
            return None
        return speed if math.isfinite(speed) else None

    def _end_stage(self, status):
        if self._stage is not None:
            row = self._stage
            row.update(end_s=self.elapsed(), status=status)
            row["duration_s"] = max(0., row["end_s"] - row["start_s"])
            self._stage = None

    def event(self, event, data):
        # Hot feedback/command events do not format text or acquire the lock.
        if event in {"feedback", "joint_command", "target_state", "high_joint_stream_review"}:
            return
        with self._lock:
            self._event(event, data)

    def _event(self, event, data):
        if event == 'belt_surface_wait':
            details = data.get('details', {})
            count = details.get('valid_green_depth_points')
            pixels = '' if count is None else f'，本帧有效点 {count}'
            self.say(f'等待看清传送带：{data["elapsed_s"]:.1f}s{pixels}；继续读取新帧，按 Q 退出')
        elif event == 'belt_surface_ready':
            if data.get('attempts', 1) > 1:
                self.say(f'带面定位成功，已读取 {data["attempts"]} 帧；开始后台测速')
        elif event == "standby_started":
            self._standby = True
            self.begin("standby", "待命等工件", announce=False,
                       at_monotonic=data["started_monotonic_s"])
            self.say("模型与设备已就绪，等待工件；空带会继续等待，按 Q 退出")
        elif event == "standby_wait":
            reason = data.get('message') or '尚未检测到可抓工件'
            self.say(f'等待工件：已等待 {data["elapsed_s"]:.1f}s；{reason}；继续等待')
        elif event == "standby_target":
            self._standby = False
            self.end("standby", at_monotonic=data["recognition_started_monotonic_s"])
            self.begin("recognition", "采图与工件识别", announce=False,
                       at_monotonic=data["recognition_started_monotonic_s"])
            self.end("recognition", at_monotonic=data["recognition_finished_monotonic_s"])
            self.say("已检测到工件，开始处理本次抓取")
        elif event in ("standby_cancelled", "standby_error"):
            self._standby = False
            self.end("standby", status="cancelled" if event == "standby_cancelled" else "interrupted")
        elif event == "visual_velocity_source":
            self.milestone("velocity_reader_started", "启动后台视觉测速（与识别并行）")
        elif event == "visual_velocity_measurement":
            self.milestone("velocity_first_report", "测速收到首帧处理结果")
            if data.get("current", True):
                for track in data.get("track_measurements", []):
                    speed = self._speed(track.get("velocity_base_m_s"))
                    if track.get("speed_valid") and speed is not None:
                        self.milestone("velocity_first_valid_any_track",
                            "首次获得有效速度（尚未代表抓取目标已关联）",
                            data={"track_id": track.get("track_id"),
                                  "sample_count": track.get("sample_count"),
                                  "speed_mm_s": speed})
                        break
        elif event == "velocity_process":
            if data.get("status") == "starting":
                self.begin("velocity_process", "测速子进程初始化", lane="background")
            elif data.get("status") == "ready":
                self.end("velocity_process")
        elif event == "visual_target_update":
            measured = data.get("velocity_measurement", {})
            speed = self._speed(measured.get("velocity_base_m_s"))
            if measured.get("speed_valid") and speed is not None:
                self.milestone("target_velocity_ready",
                    f"抓取目标速度可用：{speed:.1f} mm/s",
                    data={"track_id": data.get("track_id"), "speed_mm_s": speed,
                          "sample_count": measured.get("sample_count"),
                          "capture_host_monotonic_s": measured.get("host_capture_timestamp")})
        elif event == "moving_vision_status":
            # Healthy continuous tracking is saved in events, not spammed.
            if (not self._terminal and not self._standby and data.get("velocity_m_s") is None
                    and self.elapsed() - self._last_wait >= 5.):
                self._last_wait = self.elapsed()
                reason = str(data.get("reason", "等待目标"))
                text = REASONS.get(reason, reason)
                rate = data.get("effective_rate_hz")
                suffix = f"，处理 {rate:.1f} 帧/s" if isinstance(rate, (int, float)) else ""
                count = data.get("bound_track_sample_count")
                if count is not None:
                    suffix += f"，目标样本 {count}"
                self.say(f"视觉：{text}{suffix}")
        elif event == "policy_stage":
            name = data.get("name")
            if name == self._stage_name:
                return
            self._end_stage("incomplete" if name == "recovering" else "ended")
            self._stage_name = name
            if name not in ("recovering", "complete"):
                self._stage = {"name": name, "label": STAGES.get(name, name),
                    "start_s": self.elapsed(), "end_s": None, "duration_s": None}
                self.stages.append(self._stage)
                self.say(STAGES.get(name, name))
        elif event == "policy_event":
            name = data.get("name")
            if name == "target_selected":
                self.milestone("target_selected", data={"policy_time_s": data.get("timestamp")})
            elif name == "lift_started" and data.get("reason") == "grasp_confirmed":
                self.milestone("grasp_confirmed", "夹持反馈已确认",
                               data={"policy_time_s": data.get("timestamp")})
            elif name == "gripper_close_started":
                self.milestone("gripper_close_started", data={"policy_time_s": data.get("timestamp")})
        elif event == "planning_progress":
            row = {**data, "elapsed_s": self.elapsed()}
            self.planning.append(row)
            if data.get("kind") == "descent_check":
                return
            number = data.get("request_id")
            if data.get("status") == "requested":
                self.say(f"路线 #{number} 开始计算")
            else:
                seconds = data.get("planning_time_s")
                duration = f"{seconds:.2f}s" if isinstance(seconds, (int, float)) else "未知"
                capture = data.get("planned_target_time_s")
                received = data.get("received_policy_time_s")
                if data.get("cancelled"):
                    state = "已取消"
                elif data.get("error") == "full_pick_delivery_missed_start":
                    state = "完整路线检查已通过，但交付迟到，已错过启动时刻"
                elif data.get("error"):
                    state = "未找到可用路线"
                elif capture is not None and received is not None and capture < received:
                    state = f"路线已算出，但预约抓取时间已过 {received-capture:.2f}s"
                else:
                    state = "路线已算出，等待策略接纳"
                self.say(f"路线 #{number}：计算 {duration}；{state}")
        elif event == "high_joint_stream_started":
            self.milestone("first_high_motion_stream", "开始执行高位接近路线")
        elif event in ("task_failed", "stopped"):
            self._terminal = True
            self._end_stage("incomplete")
            reason = str(data.get("failure_reason") or data.get("error") or "已停止")
            self.milestone("task_stopped", data={"reason": reason})
            if reason != self._last_error:
                self._last_error = reason
                detail = REASONS.get(reason, reason)
                self.say(f"流程未完成：{detail}" + (f"（{reason}）" if detail != reason else ""))
        elif event == "motion_paused":
            self._terminal = True
            self._end_stage("paused")
            self.milestone("hold_started",
                "保持当前位置与使能；按 Q 或再次 Ctrl+C 退出")
        elif event == "exit_holding":
            self.milestone("hold_ended", "已退出保持循环，保留最后位置目标")
        elif event == "network_grasp_retry":
            self.say(f'抓姿未通过，重新采图选姿（已尝试 {data.get("attempt")} 次）')
        elif event == "gripper_nearby_scene_ignore":
            self.say(f'临时实验：忽略采图时夹爪模型外扩{data["padding_m"]*1000:.0f} mm'
                     f'范围内的{data["removed_points"]}个环境点（范围固定于本次采图位置）')
        elif event == "scene_depth_recheck":
            saturation = data.get("saturation") or {}
            elapsed = saturation.get("processing_s")
            removed = int(saturation.get("removed_points", 0))
            if saturation.get("reason") == "evaluated" or removed:
                duration = f"，{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else ""
                self.say(f"环境深度复核：剔除 {removed} 个红外饱和且匹配不唯一的测量点{duration}")
            background = data.get("background") or {}
            if background.get("reason") == "evaluated":
                elapsed = background.get("processing_s")
                duration = f"，{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else ""
                self.say(f'背景遮挡复核：剔除 {int(background.get("removed_points", 0))} 个异常深度点{duration}')
            if data.get("stereo_requested"):
                stereo = data.get("stereo") or {}
                self.say(f'环境双向双目复核：剔除 {int(stereo.get("removed_points", 0))} 点，'
                         f'状态 {stereo.get("reason", "不可用")}')
        elif event == "network_grasp_selected":
            prepared = data.get("prepared_inference") or {}
            selection = data.get("selection_timings_s") or {}
            parts = []
            for key, label, value in (
                ("network_inference_job", "网络推理任务（与场景处理并行）",
                 prepared.get("inference_job_s")),
                ("network_selection_wait", "选姿时额外等待网络",
                 prepared.get("selection_wait_s")),
                ("network_candidate_checks", "候选抓姿检查",
                 selection.get("original_candidate_checks_s")),
            ):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.components.append({"name": key, "label": label,
                                            "duration_s": value, "reported_at_s": self.elapsed()})
                    parts.append(f"{label} {value:.2f}s")
            if parts:
                self.say("；".join(parts))
        elif event == "grasp_pose" and not self._preview_shown:
            self._preview_shown = True
            for label, prefix in (("抓取TCP", "tcp"), ("对应法兰", "flange")):
                xyz, rpy = data.get(prefix+"_xyz_mm"), data.get(prefix+"_rpy_xyz_deg")
                if xyz is not None and rpy is not None:
                    self.say(f"{label} XYZ(mm)={[round(x, 2) for x in xyz]} "
                             f"RPY(°)={[round(x, 2) for x in rpy]}")
            html = data.get("files", {}).get("html")
            if html and self.directory:
                self.say(f"抓姿预览：{Path(self.directory)/html}")
        elif event == "visual_velocity_error":
            if not self._terminal and self.elapsed() - self._last_visual_error >= 5.:
                self._last_visual_error = self.elapsed()
                self.say(f'测速提示：{data.get("error")}')
        elif event in ("planning_backend_fallback", "grasp_depth_backend_fallback",
                       "grasp_depth_camera_fallback", "cleanup_error",
                       "preview_error", "evidence_save_error"):
            reason = str(data.get("reason") or data.get("error") or data.get("errors"))
            if reason != self._last_error:
                self._last_error = reason
                self.say(f"提示：{reason}")
        if self.verbose and event not in {
                "visual_velocity_measurement", "visual_target_update", "grasp_pose"}:
            self.say(event + " " + json.dumps(data, ensure_ascii=False, allow_nan=False))

    def finish_policy(self, result):
        with self._lock:
            complete = result.get("status") in ("cycle_completed_from_feedback",
                                               "offline_full_policy_completed", "prepared_high_only")
            self._end_stage("completed" if complete else "incomplete")
            self.milestone("policy_returned")
            self._terminal = True
            if complete:
                self.say("本轮流程完成" if result["status"] != "prepared_high_only" else "已到高位并保持")
            elif result.get("error") and "task_stopped" not in self.milestones:
                self._event("task_failed", result)
            if self.verbose:
                self.say("result " + json.dumps(result, ensure_ascii=False))

    def report(self, result):
        with self._lock:
            return {"clock": "host_monotonic", "origin_monotonic_s": self.started,
                "scope": "Python entry to final report; includes imports, warmup, holds and cleanup",
                "duration_note": "Parallel phases overlap; do not add durations. Milestones are host receipt times, not camera capture times.",
                "elapsed_s": self.elapsed(), "phases": self.phases,
                "milestones": self.milestones, "policy_stages_wall": self.stages,
                "planning": self.planning, "components": self.components,
                "result_status": result.get("status")}

    def summary(self, result):
        with self._lock:
            self.interrupt_active_phases()
            self._end_stage("incomplete")
            self.say("耗时汇总（秒；后台测速与识别并行，各项不能直接相加）")
            for row in self.phases:
                if row["lane"] in ("main", "background"):
                    self.writer(f'  {row["label"]}：{row["duration_s"]:.2f}'
                                + ("（后台并行）" if row["lane"] == "background" else "")
                                + ("（中断）" if row["status"] != "completed" else ""))
            speed_start = self.milestones.get("velocity_reader_started", {}).get("elapsed_s")
            if speed_start is not None:
                for key, label in (
                    ("velocity_first_report", "测速启动→首帧处理结果"),
                    ("velocity_first_valid_any_track", "测速启动→任意目标有效速度"),
                    ("target_velocity_ready", "测速启动→抓取目标速度可用"),
                ):
                    stamp = self.milestones.get(key, {}).get("elapsed_s")
                    if key == "target_velocity_ready" and any(row["name"] == "standby" for row in self.phases):
                        label += "（含待命等工件）"
                    self.writer(f"  {label}：" + ("未获得" if stamp is None else f"{stamp-speed_start:.2f}"))
            for row in self.stages:
                self.writer(f'  {row["label"]}：{row["duration_s"]:.2f}'
                            + ("（未完成）" if row["status"] in ("incomplete", "paused") else ""))
            hold = self.milestones.get("hold_started", {}).get("elapsed_s")
            if hold is not None:
                end = self.milestones.get("hold_ended", {}).get("elapsed_s", self.elapsed())
                self.writer(f"  退出前原位保持：{end-hold:.2f}")
            self.writer(f"  启动至收尾总用时：{self.elapsed():.2f}")
