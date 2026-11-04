"""Fit a host-visible position response from a completed execution trace only.

No camera, CAN, robot SDK, firmware or motion interface is opened.  Historical
events lack exact dispatch timestamps: their default output is a sufficiency
report; --allow-proxy-times enables explicitly ineligible diagnostic fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from mujoco_conveyor_sorting.controllers.joint_response import fit_joint_response


def read_trace(path: Path, *, stage: str | None = "pregrasp") -> tuple[dict, dict]:
    folder = path if path.is_dir() else path.parent
    npz_path = folder / "execution_trace.npz" if path.is_dir() else path
    if npz_path.suffix == ".npz" and npz_path.exists():
        summary_path = folder / "execution_trace_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        with np.load(npz_path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        selected = np.asarray(a.get("dispatch_succeeded", np.ones(len(a["dispatched_q_rad"]), bool)), bool)
        feedback_selected = np.ones(len(a["feedback_q_rad"]), bool)
        if stage is not None:
            if "sample_stage" in a:
                selected &= a["sample_stage"] == stage
            if "feedback_sample_stage" in a:
                feedback_selected &= a["feedback_sample_stage"] == stage
        feedback_time_key = next((key for key in (
            "feedback_joint_axis_sample_monotonic_s", "feedback_joint_axis_receive_monotonic_s",
            "feedback_joint_sample_monotonic_s") if key in a), None)
        precise = (feedback_time_key is not None and summary.get("clock_domain") == "host_monotonic"
                   and summary.get("stamp_clock_domain") == "host_wall"
                   and "SDK" in summary.get("feedback_timestamp_source", ""))
        if feedback_time_key is None:
            feedback_time_key = "feedback_read_monotonic_s"
        samples = {"command_times_s": a["dispatch_end_monotonic_s"][selected],
                   "commands_rad": a["dispatched_q_rad"][selected],
                   "feedback_times_s": a[feedback_time_key][feedback_selected],
                   "feedback_rad": a["feedback_q_rad"][feedback_selected]}
        evidence = {"input": str(npz_path), "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                    "stage": stage, "runtime_eligible": bool(precise),
                    "joint_command_mode": summary.get("joint_command_mode", "unknown"),
                    "exact_host_dispatch_times_available": True,
                    "feedback_time_field": feedback_time_key,
                    "feedback_timestamp_source": summary.get("feedback_timestamp_source", "unknown"),
                    "clock_domain": summary.get("clock_domain", "unknown"),
                    "time_proxy_used": not precise,
                    "encoder_internal_sample_time_available": False,
                    "feedback_time_limitation": summary.get("feedback_time_limitation", "unknown")}
        return samples, evidence

    events_path = folder / "events.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    commands = [r for r in rows if r.get("event") == "joint_command"
                and (stage is None or r.get("stage") == stage)]
    feedback = [r for r in rows if r.get("event") == "feedback" and r.get("joints")
                and r.get("joint_stamp") is not None]
    # Historical feedback rows have no stage field.  Use the selected command
    # interval, then the fitter discards the maximum-delay initial window.
    ct = np.asarray([r["time"] for r in commands], float)
    selected_feedback = [r for r in feedback if len(ct) and ct.min() <= r["joint_stamp"] <= ct.max()]
    samples = {"command_times_s": ct,
               "commands_rad": np.asarray([r["joints_rad"] for r in commands], float).reshape(-1, 7),
               "feedback_times_s": np.asarray([r["joint_stamp"] for r in selected_feedback], float),
               "feedback_rad": np.asarray([r["joints"] for r in selected_feedback], float).reshape(-1, 7)}
    evidence = {"input": str(events_path), "sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                "stage": stage, "runtime_eligible": False,
                "exact_host_dispatch_times_available": False,
                "time_proxy_used": True, "clock_domain": "host_wall",
                "command_timestamp_source": "joint_command event write time after SDK call",
                "feedback_timestamp_source": "SDK joint_stamp oldest host CAN receive packet time",
                "encoder_internal_sample_time_available": False,
                "missing_for_calibration": ["actual SDK dispatch end monotonic time",
                    "per-axis feedback packet receive time on the same monotonic clock"],
                "limitation": "Proxy fit mixes SDK call, event write, CAN transport, packet grouping and servo response."}
    return samples, evidence


def run(args: argparse.Namespace) -> dict:
    samples, evidence = read_trace(args.recording, stage=None if args.stage == "all" else args.stage)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "hardware_connections": 0,
              "scope": "offline host-visible joint response identification, not physical execution validation",
              "evidence": evidence,
              "command_samples": len(samples["commands_rad"]),
              "feedback_samples": len(samples["feedback_rad"])}
    if evidence["time_proxy_used"] and not args.allow_proxy_times:
        report["status"] = "insufficient_precise_timestamps_report_only"
        report["model_written"] = False
    else:
        model, fit = fit_joint_response(**samples, maximum_delay_s=args.maximum_delay_s,
            maximum_time_constant_s=args.maximum_time_constant_s, train_fraction=args.train_fraction,
            metadata=dict(evidence, runtime_eligible=bool(evidence["runtime_eligible"] and not args.diagnostic_only)))
        report.update(status="fit_completed", fit=fit, model_written=True)
        model_path = args.output / "joint_response_model.json"
        model_path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        report["model_path"] = str(model_path)
        report["runtime_eligible"] = model.metadata["runtime_eligible"]
    (args.output / "joint_response_fit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="completed recording folder, execution_trace.npz, or events.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="pregrasp", help="command/feedback stage, or all; default pregrasp")
    parser.add_argument("--maximum-delay-s", type=float, default=.5, help="offline search range, not a motion gate")
    parser.add_argument("--maximum-time-constant-s", type=float, default=2.)
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--allow-proxy-times", action="store_true", help="diagnostic fit for old logs; never runtime eligible")
    parser.add_argument("--diagnostic-only", action="store_true", help="write a model that leaves runtime commands unchanged")
    report = run(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("status", "command_samples", "feedback_samples", "model_written")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
