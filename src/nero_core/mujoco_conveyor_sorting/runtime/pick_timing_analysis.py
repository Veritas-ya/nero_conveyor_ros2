"""Read command/measurement timing without inferring unmeasured robot delays.

Event ``time`` values share the host log clock. Device timestamps are retained
as independent columns; they are never subtracted from policy or host times.
Width and motor force are telemetry, not ground-truth contact measurements.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _statistics(values):
    values = [v for value in values if (v := _number(value)) is not None]
    if not values:
        return {"n": 0, "unit": "s", "median": "unknown", "p95": "unknown", "maximum": "unknown"}
    a = np.asarray(values)
    return {"n": len(values), "unit": "s", "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "maximum": float(a.max())}


def analyze_events(events: Iterable[Mapping]) -> dict:
    """Align each sent command to preceding records, preserving missing values."""
    commands, gripper, feedback = [], [], []
    last_feedback = last_target = None
    stage_times = {}
    last_send = None
    for event in events:
        name = event.get("event")
        stamp = _number(event.get("time"))
        if name == "feedback":
            last_feedback = event
            feedback.append({"host_log_s": stamp,
                "joint_device_stamp": event.get("joint_stamp", "unknown"),
                "gripper_device_stamp": event.get("gripper_stamp", "unknown"),
                "width_m": event.get("width_m", "unknown"),
                "motor_force_n": event.get("force_n", "unknown"),
                "signed_motor_force_n": event.get("force_signed_n", "unknown"),
                "read_wait_s": event.get("feedback_timing", {}).get("read_wait_s", "unknown")})
        elif name == "target_state":
            last_target = event
        elif name == "policy_stage":
            stage_times.setdefault(str(event.get("name", "unknown")), stamp)
        elif name == "gripper_command":
            gripper.append({"host_log_s": stamp, "width_m": event.get("width_m", "unknown"),
                **{key: event.get(key, "unknown") for key in (
                    "first_close_committed_at_dispatch", "gate_policy_time_s",
                    "jaw_send_policy_time_s", "gate_to_jaw_send_s", "gate_feedback_age_s",
                    "closed_sweep_compute_s", "jaw_send_call_s")}})
        elif name == "joint_command":
            timing = event.get("dispatch_timing") or {}
            target = last_target or {}
            observed_age = _number(target.get("observation_age_s"))
            target_log = _number(target.get("time"))
            target_age = (None if stamp is None or target_log is None or observed_age is None
                          or stamp < target_log else observed_age + stamp-target_log)
            fb = last_feedback or {}
            fb_log = _number(fb.get("time"))
            packet_age = _number(fb.get("feedback_timing", {}).get("oldest_packet_age_s"))
            feedback_age = (None if stamp is None or fb_log is None or packet_age is None
                            or stamp < fb_log else packet_age + stamp-fb_log)
            gap = (None if last_send is None or stamp is None or stamp < last_send else stamp-last_send)
            row = {"host_log_s": stamp, "stage": event.get("stage", "unknown"),
                "policy_time_s": event.get("policy_time_s", "unknown"),
                "command_gap_s": gap,
                "target_age_at_send_from_log_s": target_age,
                "feedback_age_at_send_from_log_s": feedback_age,
                "compute_s": timing.get("compute_s", "unknown"),
                "lateness_s": timing.get("lateness_s", "unknown"),
                "control_dt_s": timing.get("control_dt_s", "unknown"),
                "decision_to_send_s": timing.get("decision_to_send_s", "unknown"),
                "motion_finalize_s": timing.get("motion_finalize_s", "unknown"),
                "target_observation_age_at_dispatch_s": timing.get("target_observation_age_at_dispatch_s", "unknown"),
                "feedback_oldest_packet_age_at_dispatch_s": timing.get("feedback_oldest_packet_age_at_dispatch_s", "unknown"),
                "velocity_source": target.get("velocity_source", "unknown"),
                "feedback_read_wait_s": fb.get("feedback_timing", {}).get("read_wait_s", "unknown")}
            commands.append(row)
            last_send = stamp
    metrics = {key: _statistics(row[key] for row in commands) for key in (
        "command_gap_s", "target_age_at_send_from_log_s", "feedback_age_at_send_from_log_s",
        "compute_s", "lateness_s", "decision_to_send_s", "motion_finalize_s",
        "target_observation_age_at_dispatch_s", "feedback_oldest_packet_age_at_dispatch_s")}
    return {"joint_commands": commands, "feedback_samples": feedback, "gripper_commands": gripper,
        "stage_first_host_log_s": stage_times, "metrics": metrics,
        "gripper_metrics": {key: _statistics(row[key] for row in gripper) for key in (
            "gate_to_jaw_send_s", "gate_feedback_age_s", "closed_sweep_compute_s", "jaw_send_call_s")},
        "first_close_dispatch_count": sum(
            row["first_close_committed_at_dispatch"] is True for row in gripper),
        "hardware_actions_performed_by_analysis": False,
        "unmeasured": {"robot_response_delay_s": "unknown", "physical_contact_time_s": "unknown",
                       "actual_grasp_success": "unknown"},
        "interpretation": [
            "Target and feedback send ages use the preceding host log timestamps and are logging-based diagnostics.",
            "Device timestamps are retained independently, never compared numerically with host/policy clocks.",
            "Motor force and width do not establish actual contact or grasp success.",
            "No measured robot/closure response parameter is invented from command dispatch gaps."]}


def analyze_recording(recording: Path, output: Path) -> dict:
    recording, output = Path(recording), Path(output)
    source = recording / "events.jsonl" if recording.is_dir() else recording
    with source.open(encoding="utf-8") as f:
        result = analyze_events(json.loads(line) for line in f if line.strip())
    output.mkdir(parents=True, exist_ok=True)
    for key in ("joint_commands", "feedback_samples", "gripper_commands"):
        rows = result[key]
        with (output / f"{key}.csv").open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {k: v for k, v in result.items() if k not in ("joint_commands", "feedback_samples", "gripper_commands")}
    summary.update(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   counts={key: len(result[key]) for key in ("joint_commands", "feedback_samples", "gripper_commands")})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary
