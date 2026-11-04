#!/usr/bin/env python3
"""Real GraspGen-X inference and continuous NERO MuJoCo sorting benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import yaml

from experiments.graspgenx_sim.camera import FixedGeminiSimulation, cloud_for_track, projection_selfcheck
from experiments.graspgenx_sim.continuous import ContinuousGraspAdapter, ContinuousSortingEvaluator
from experiments.graspgenx_sim.inference_worker import GraspGenXBackend
from experiments.graspgenx_sim.report import write_report
from experiments.graspgenx_sim.candidates import select_candidates, snapshot_env_for_candidates
from experiments.graspgenx_sim.visuals import save_input_visuals, draw_grasp
from experiments.graspgenx_sim.sensor_control import SensorControlEnvironment, SensorControlConfig
from experiments.graspgenx_sim.input_integrity import build_input_provenance, select_captured_request, validate_candidate_admission
from experiments.graspgenx_sim.diagnostic_templates import DiagnosticTemplateEnvironment


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def infer_and_select(backend, cloud, seed, generated, returned, snapshot, position, cfg,
                     *, grasp_position_mode='network'):
    """Keep CUDA and IK work off the thread advancing conveyor physics."""
    result = backend.infer(cloud, seed, generated, returned)
    if grasp_position_mode in ('visible_bbox_center', 'visible_obb_center'):
        from .bbox_center_grasp import center_network_tcp_on_visible_box, center_network_tcp_on_visible_obb
        original=np.asarray(result['poses']).copy()
        modified,info=(center_network_tcp_on_visible_obb if grasp_position_mode == 'visible_obb_center' else center_network_tcp_on_visible_box)(cloud,original)
        result['original_network_poses']=original
        result['poses']=modified
        result['center_grasp_info']=info
    elif grasp_position_mode != 'network':
        raise ValueError('unknown grasp position mode')
    started = time.perf_counter()
    accepted, rejected = select_candidates(snapshot, result["poses"], result["scores"], position, cfg,
                                           observed_cloud_world=cloud)
    result["accepted_candidates"] = accepted
    result["rejected_candidates"] = rejected
    if grasp_position_mode in ('visible_bbox_center', 'visible_obb_center'):
        from .bbox_center_grasp import annotate_center_candidates
        annotate_center_candidates(accepted,result['original_network_poses'],result['center_grasp_info'])
    result["metadata"]["candidate_filter_wall_s"] = time.perf_counter() - started
    result["metadata"]["candidate_snapshot_sim_s"] = snapshot.source_timestamp
    return result


def warmup_model(backend, seed, generated, returned):
    """Warm the model before the episode; synthetic points never enter control."""
    # Fixed finite cloud is only used to initialize kernels. The backend resets
    # its RNG on every infer call, including each subsequent real observation.
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-0.02, 0.02, (512, 3)).astype(np.float32)
    started = time.perf_counter()
    result = backend.infer(cloud, int(seed) + 999999, generated, returned)
    return {"source": "synthetic_kernel_warmup",
            "point_count": len(cloud), "wall_s": time.perf_counter() - started,
            "metadata": result["metadata"], "used_for_control_or_success": False,
            "before_benchmark_wall_clock": True}


def continuous_reference(policy, timestamp):
    """Record the certified q/v/a separately from action and plant feedback."""
    plan = getattr(policy, "_continuous_plan", None)
    if plan is None:
        return None
    q, velocity, acceleration = plan.sample(min(float(timestamp), plan.end))
    return {"sample_sim_s": float(timestamp), "request_id": plan.request.request_id,
            "start_sim_s": plan.start, "capture_sim_s": plan.end,
            "joint_pos_rad": q, "joint_vel_rad_s": velocity,
            "joint_acc_rad_s2": acceleration,
            "is_sent_or_encoder_sample": False}


def config_overrides(cfg):
    exp, cam = cfg["experiment"], cfg["camera"]
    tracking = cfg.get("tracking", {})
    control = cfg.get("control", {})
    count, interval = int(exp["objects"]), float(exp["feed_interval_s"])
    overrides = {
        "robot": {"robot": {"trajectory": {
            "release_retreat_distance": float(control.get("release_retreat_distance_m", .020)),
            "prefetch_handoff_retreat_distance": float(control.get("release_retreat_distance_m", .020)),
        }}},
        "scene": {
            "task": {"active_category_ids": exp["active_category_ids"]},
            **({"workspace": {"dynamic_intercept": tracking["dynamic_intercept"]}}
               if "dynamic_intercept" in tracking else {}),
            "conveyor": {"default_speed": exp["belt_speed_m_s"], "speed_jitter_std": 0.,
                         "speed_range": [0., 1.]},
            "spawning": {"interval_seconds": interval,
                         **({"x": .08} if exp.get("test_mode") == "stopped" else {})},
            "episode": {"object_count": count, "initial_objects": 1,
                        "max_time": count * interval + float(exp["max_tail_s"])},
            "debug": {"enabled": False},
            "randomization": {"belt_speed": {"enabled": False},
                              "belt_speed_jitter": {"enabled": False}},
        },
        "cameras": {
            "cameras": {"capture_hz": cam["capture_hz"], "top": {"resolution": cam["resolution"]}},
            "perception": {"max_track_age_s": tracking.get("maximum_track_age_s", 4.8),
                           "coast_confidence_half_life_s": tracking.get("coast_confidence_half_life_s", 2.0),
                           **({"tracker": {"new_track_max_x": .24, "trusted_measurement_max_x": .24}}
                              if exp.get("test_mode") == "stopped" else {})},
        },
        # Three 3 Hz observations span .668 s. The stock .2/.35 s windows
        # otherwise keep prediction_ready false for every frame.
        "latency": {"latency": {"prediction": {
            "history_window": tracking.get("history_window_s", .8),
            "velocity_window_s": tracking.get("velocity_window_s", .8)}}},
    }
    if control.get("moving_approach_mode", "high") == "continuous":
        overrides["scene"].setdefault("workspace", {}).setdefault("dynamic_intercept", {})["enabled"] = True
    return overrides


def main(argv=None, **_kwargs):
    # This historical runner creates the legacy controller. Keep reusable
    # inference functions, but never launch that controller as the current
    # project entry; fail before model loading, rendering or plant movement.
    from experiments.graspgenx_sim.full_preplanned_entry import main as current_entry
    return current_entry(argv)


def _historical_main_not_a_current_entry(argv=None, *, environment_type=FixedGeminiSimulation,
         control_environment_type=SensorControlEnvironment, adapter_type=ContinuousGraspAdapter):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--objects", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--belt-speed", type=float)
    parser.add_argument("--feed-interval", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fast", action="store_true", help="Unthrottled offline replay; timing is not a real-time result")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo scene viewer")
    parser.add_argument("--moving-approach", choices=("high", "continuous"), default=None,
                        help="high保留原高位接近；continuous从当前关节直接规划到未来抓取位置")
    parser.add_argument("--max-sim-time", type=float)
    parser.add_argument("--test-mode", choices=("moving", "stopped"), default=None,
                        help="Stopped is a diagnostic fixture at x=.08; it is excluded from moving acceptance")
    parser.add_argument("--grasp-source", choices=("graspgenx", "baseline_diagnostic"), default="graspgenx",
                        help="The legacy catalog templates are a separate diagnostic comparison only")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    control = cfg.setdefault("control", {})
    control["moving_approach_mode"] = args.moving_approach or control.get("moving_approach_mode", "high")
    exp["test_mode"] = args.test_mode or exp.get("test_mode", "moving")
    exp["grasp_source"] = args.grasp_source
    for arg, key in ((args.objects, "objects"), (args.seed, "seed"),
                     (args.belt_speed, "belt_speed_m_s"), (args.feed_interval, "feed_interval_s")):
        if arg is not None:
            exp[key] = arg
    if exp["test_mode"] == "stopped":
        exp["belt_speed_m_s"] = 0.
        cfg["graspgenx"]["inference_trigger_x"] = [-.3, .25]
    if int(exp["objects"]) < 1 or not 0 <= float(exp["belt_speed_m_s"]) <= 1 or float(exp["feed_interval_s"]) <= 0:
        parser.error("objects, belt speed and feed interval must be positive")
    if exp["test_mode"] == "moving" and float(exp["belt_speed_m_s"]) <= 0:
        parser.error("zero belt speed belongs to the stopped diagnostic, not moving acceptance")
    exp["real_time"] = not args.fast
    output = (args.output or ROOT / "recordings/graspgenx_sim" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "inference").mkdir()
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    state = {"status": "starting", "output": str(output), "hardware_access": False,
             "config": cfg, "network_executed": False, "target_met": False,
             "acceptance_eligible": exp["test_mode"] == "moving" and args.grasp_source == "graspgenx"}
    sources = ["camera.py", "candidates.py", "continuous.py", "policy.py",
               "physical_containment.py", "visual_clearance.py", "reachability.py", "inference_worker.py", "run.py",
               "sensor_control.py", "input_integrity.py", "diagnostic_templates.py", "motor_effort_contract_audit.py", "servo_response.py"]
    state["implementation"] = {str((HERE / source).relative_to(ROOT)):
                               hashlib.sha256((HERE / source).read_bytes()).hexdigest()
                               for source in sources}
    for source in ("baseline_sorting_policy.py", "continuous_intercept.py",
                   "continuous_joint_trajectory.py", "planning_worker.py"):
        path = ROOT / "mujoco_conveyor_sorting/controllers" / source
        state["implementation"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(output / "result.json", state)
    print(f"独立仿真输出：{output}", flush=True)
    print("加载官方 GraspGen-X 权重；此入口不导入 CAN、相机 SDK 或真机控制后端。", flush=True)
    print("接近模式：" + ("整条连续抓取路径（continuous）" if control["moving_approach_mode"] == "continuous"
                         else "原高位接近（high）"), flush=True)
    env = control_env = adapter = evaluator = video = viewer = control_stream = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graspgenx")
    pending = None
    inference_log, policy_log, policy_events, slow_control_calls, input_deferrals = [], [], [], [], []
    try:
        backend = None
        if args.grasp_source == "graspgenx":
            backend = GraspGenXBackend(seed=int(exp["seed"]), max_points=cfg["graspgenx"]["maximum_input_points"])
            backend.torch.set_num_threads(2)
            state["model"] = backend.provenance
            state["model_warmup"] = warmup_model(backend, int(exp["seed"]),
                cfg["graspgenx"]["generated_candidates"], cfg["graspgenx"]["returned_candidates"])
        else:
            state["model"] = {"source": "legacy_catalog_diagnostic_only", "neural_network": False}
        overrides = config_overrides(cfg)
        prototype = HERE / "oblique_prototypes.yaml"
        if prototype.is_file():
            overrides["cameras"]["perception"]["prototype_catalog_path"] = str(prototype)
        state["runtime_overrides"] = overrides
        env = environment_type(
            capture_path=ROOT / cfg["camera"]["capture_reference"],
            config_overrides=overrides, grasp_assist=False,
            perception_mode="rendered_rgbd", grasp_event_source="physical_contacts",
            seed=int(exp["seed"]),
        )
        state["camera"] = env.camera_record()
        model_fields = ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range",
                        "geom_size", "geom_pos", "geom_quat", "geom_friction", "mesh_vert", "mesh_face",
                        "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "qpos0")
        state["physics_signature"] = {
            "model_arrays_sha256": {field: hashlib.sha256(np.ascontiguousarray(getattr(env.model, field)).tobytes()).hexdigest()
                                     for field in model_fields},
            "timestep_s": float(env.model.opt.timestep), "gravity": env.model.opt.gravity.copy(),
            "grasp_assist": bool(env.grasp_assist), "purpose": "comparison_and_evaluation_only"}
        state["spawn_manifest"] = []
        original_spawn = env.spawn_object
        def recorded_spawn(*a, **kw):
            identifier = original_spawn(*a, **kw)
            if identifier is not None:
                runtime = next(x for x in env._objects if x.logical_id == identifier)
                state["spawn_manifest"].append({
                    "source": "evaluation_only_spawn_configuration",
                    "object_id": int(identifier), "category_id": int(runtime.category_id),
                    "spawned_at_s": float(runtime.spawned_at),
                    "pose_world_wxyz": env.data.qpos[runtime.qpos_adr:runtime.qpos_adr + 7].copy(),
                    "mass_kg": float(env.model.body_mass[runtime.body_id]),
                    "geom_friction": env.model.geom_friction[list(runtime.geom_ids)].copy()})
            return identifier
        env.spawn_object = recorded_spawn
        state["projection_selfcheck"] = projection_selfcheck(env)
        if not state["projection_selfcheck"]["passed"]:
            raise RuntimeError("Rendered projection selfcheck failed")
        if env.robot_controller is None or env.gripper_controller is None:
            raise RuntimeError("NERO/AGX controllers unexpectedly unavailable")
        control_type = control_environment_type if args.grasp_source == "graspgenx" else DiagnosticTemplateEnvironment
        control_env = control_type(env, SensorControlConfig(
            maximum_track_age_s=cfg.get("tracking", {}).get("maximum_track_age_s", 4.8),
            coast_confidence_half_life_s=cfg.get("tracking", {}).get("coast_confidence_half_life_s", 2.)))
        state["control_boundary"] = control_env.provenance()
        adapter = adapter_type(control_env, compression_m=cfg["graspgenx"]["closing_preload_m"],
                                        require_network_grasp=args.grasp_source == "graspgenx",
                                        moving_approach_mode=control["moving_approach_mode"],
                                        enable_speculative_prefetch=cfg.get("control", {}).get("speculative_prefetch", False),
                                        acquisition_translation_step_m=cfg.get("control", {}).get("acquisition_translation_step_m", .010),
                                        acquisition_command_integrator=cfg.get("control", {}).get("acquisition_command_integrator", False),
                                        visual_acquisition_tracking=cfg.get("control", {}).get("visual_acquisition_tracking", False),
                                        minimum_motor_hold_effort=cfg.get("control", {}).get("minimum_motor_hold_effort_sim_N", 0.),
                                        visual_servo_bias_compensation=cfg.get("control", {}).get("visual_servo_bias_compensation", False))
        if hasattr(adapter, 'full_observation_trace'):
            adapter.full_observation_trace.max_observation_age_s = cfg['tracking']['maximum_track_age_s']
            adapter.full_observation_trace.max_result_age_s = cfg['tracking'].get('maximum_result_age_s')
        state["policy_config"] = asdict(adapter.policy.config)
        state["effective_dynamic_intercept"] = dict(control_env.scene_config.get(
            "workspace", {}).get("dynamic_intercept", {}))
        evaluator = ContinuousSortingEvaluator(env, terminal_retention_s=exp["terminal_retention_s"])
        env.reset(seed=int(exp["seed"]), options={"belt_speed": exp["belt_speed_m_s"], "initial_objects": 1})
        obs = control_env.get_observation()
        warm_frame = env._camera_frames[-1] if env._camera_frames else None
        if warm_frame is not None and backend is not None and "model_warmup" not in state:
            warm_ids = np.unique(warm_frame["top_instance_labels"])
            for warm_id in warm_ids[warm_ids >= 0]:
                warm_cloud = cloud_for_track(warm_frame, int(warm_id))
                if len(warm_cloud) >= cfg["graspgenx"]["minimum_input_points"]:
                    warm_result = backend.infer(warm_cloud, int(exp["seed"]) + 999999,
                                               cfg["graspgenx"]["generated_candidates"],
                                               cfg["graspgenx"]["returned_candidates"])
                    state["model_warmup"] = {"source": "initial_rendered_control_frame",
                                              "capture_sim_s": float(warm_frame["timestamp"]),
                                              "track_id": int(warm_id),
                                              "metadata": warm_result["metadata"],
                                              "used_for_control_or_success": False,
                                              "before_benchmark_wall_clock": True}
                    break
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        if cfg["video"]["enabled"] and not args.no_video:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(output / "continuous_sorting.mp4"), fps=cfg["video"]["fps"],
                                       codec="libx264", quality=7, macro_block_size=1)
            # Initialise renderer/encoder before physics starts. Keep the
            # genuine t=0 image as the first frame; no moving part is frozen.
            video.append_data(env._render_camera(cfg["video"]["camera"], tuple(cfg["video"]["resolution"]), depth=False))
        attempts, last_request = {}, {}
        last_frame, next_video_time, last_progress, last_policy = -1., (1 / cfg["video"]["fps"] if video is not None else 0.), -1., None
        wall_start = time.perf_counter()
        control_stream = (output / "control_telemetry.jsonl").open("w", encoding="utf-8")
        simulation_start = float(env.data.time)
        lag_max, lag_samples = 0., []
        terminal_seen_s = None
        run_scene_epoch = int(env._scene_epoch)
        max_sim_time = args.max_sim_time or (int(exp["objects"]) * float(exp["feed_interval_s"]) + float(exp["max_tail_s"]))
        state["status"] = "running"
        save_json(output / "result.json", state)
        while float(env.data.time) < max_sim_time:
            if int(env._scene_epoch) != run_scene_epoch:
                raise RuntimeError("Scene reset during benchmark invalidates track IDs and this evaluation")
            now = float(env.data.time)
            if pending is not None and pending["future"].done():
                job = pending
                pending = None
                try:
                    result = job["future"].result()
                except Exception as error:
                    save_json(output / "inference" / f"{job['name']}.json", {
                        "track_id": job["track_id"], "capture_sim_s": job["capture_sim_s"],
                        "ready_sim_s": now, "status": "network_inference_failed", "error": str(error),
                    })
                    raise
                state["network_executed"] = True
                np.savez_compressed(output / "inference" / f"{job['name']}.npz",
                                    poses=result["poses"], scores=result["scores"])
                live = next((item for item in obs.get("objects", []) if int(item["object_id"]) == job["track_id"]), None)
                admission = validate_candidate_admission(
                    job["provenance"], live, int(env._scene_epoch), now,
                    max_observation_age_s=cfg["tracking"]["maximum_track_age_s"],
                    max_result_age_s=cfg["tracking"].get("maximum_result_age_s", 4.8))
                if not admission["allowed"] or (live is not None and float(np.asarray(live["position"])[0]) > float(env.grasp_region[1])):
                    entry = {"track_id": job["track_id"], "capture_sim_s": job["capture_sim_s"],
                             "ready_sim_s": now, "status": "stale_track_result", "metadata": result["metadata"],
                             "input_provenance": job["provenance"], "admission": admission}
                    inference_log.append(entry)
                    save_json(output / "inference" / f"{job['name']}.json", entry)
                    result = None
                if result is None:
                    continue
                accepted, rejected = result["accepted_candidates"], result["rejected_candidates"]
                entry = {"track_id": job["track_id"], "capture_sim_s": job["capture_sim_s"],
                         "ready_sim_s": now, "metadata": result["metadata"],
                         "accepted_candidates": accepted, "rejected_candidates": rejected,
                         "input_visuals": job["visuals"], "input_provenance": job["provenance"], "admission": admission}
                if accepted:
                    entry["selected_grasp_png"] = draw_grasp(
                        output / "inference" / job["name"], accepted[0]["T_world_tcp"],
                        job["visuals"], f"track {job['track_id']} | selected GraspGen-X TCP at capture")
                inference_log.append(entry)
                save_json(output / "inference" / f"{job['name']}.json", entry)
                if accepted:
                    chosen = accepted[0]
                    installed = adapter.set_grasp(job["track_id"], chosen["tcp_pose_world_wxyz"],
                                                  job["track_position"], job["capture_sim_s"],
                                                  closing_width_m=chosen["closing_width_m"], score=chosen["score"])
                    if installed and hasattr(adapter, 'on_network_result'):
                        report = next((r for r in reversed(env.nero7_reports)
                            if abs(float(r['capture_sim_s'])-float(job['capture_sim_s'])) < 1e-6), None)
                        clock_epoch = None if report is None else report['peer'].get('clock_epoch')
                        adapter.on_network_result(job['provenance'], chosen, clock_epoch=clock_epoch)
                    if installed:
                        entry["pregrasp_seed_registered"] = adapter.policy.register_pregrasp_seed(
                            job["track_id"], chosen["pregrasp_ik"]["joint_positions"],
                            scene_epoch=job["scene_epoch"])
                        save_json(output / "inference" / f"{job['name']}.json", entry)
                    print(f"t={now:.2f}s track={job['track_id']}：网络 {len(result['poses'])} 个候选，可用 {len(accepted)} 个。", flush=True)
                else:
                    print(f"t={now:.2f}s track={job['track_id']}：网络候选均不可用，原因已保存。", flush=True)
            # Network input observes the same camera delivery latency as
            # visual control, instead of reading a frame before its arrival.
            frame = next((f for f in reversed(env._camera_frames)
                          if float(f["timestamp"]) + float(env.visual_latency) <= now + 1e-9), None)
            if frame is not None and frame["timestamp"] > last_frame:
                last_frame = float(frame["timestamp"])
                if pending is None and backend is not None:
                    ids = np.unique(frame["top_instance_labels"])
                    requested = []
                    for identifier in ids[ids >= 0]:
                        identifier = int(identifier)
                        if identifier in adapter.ready_track_ids:
                            continue
                        if attempts.get(identifier, 0) >= int(cfg["graspgenx"]["maximum_inferences_per_track"]):
                            continue
                        if now - last_request.get(identifier, -100.) < cfg["graspgenx"]["minimum_retry_interval_sim_s"]:
                            continue
                        live_input = next((x for x in obs.get("objects", []) if int(x["object_id"]) == identifier), None)
                        if live_input is None or not live_input.get("prediction_ready", False):
                            input_deferrals.append({"track_id": identifier, "capture_sim_s": last_frame,
                                                    "reason": "visual_history_not_ready"})
                            continue
                        _, observed = control_env._observed_state_at(identifier, last_frame)
                        if observed is None:
                            continue
                        position = np.asarray(observed["position"], dtype=float).copy()
                        lower, upper = cfg["graspgenx"]["inference_trigger_x"]
                        if not lower <= position[0] <= upper:
                            continue
                        if cfg["camera"].get("require_mask_inside_frame", False):
                            mask_y, mask_x = np.nonzero(frame["top_instance_labels"] == identifier)
                            border = int(cfg["camera"].get("input_border_margin_pixels", 2))
                            height, width = frame["top_instance_labels"].shape
                            if (mask_x.min() < border or mask_y.min() < border
                                    or mask_x.max() >= width - border or mask_y.max() >= height - border):
                                input_deferrals.append({"track_id": identifier, "capture_sim_s": last_frame,
                                                        "reason": "visual_mask_touches_image_border",
                                                        "bbox_xyxy": [int(mask_x.min()), int(mask_y.min()),
                                                                      int(mask_x.max()) + 1, int(mask_y.max()) + 1]})
                                continue
                        requested.append((position[0], identifier, position, observed))
                    if requested:
                        _, identifier, position, observed = select_captured_request(requested)
                        cloud = cloud_for_track(frame, identifier)
                        if len(cloud) >= cfg["graspgenx"]["minimum_input_points"]:
                            provenance = build_input_provenance(frame, identifier, cloud, observed,
                                                                int(env._scene_epoch), now)
                            if not provenance["input_valid"]:
                                input_deferrals.append({"track_id": identifier, "capture_sim_s": last_frame,
                                                        "reason": "invalid_captured_input", "provenance": provenance})
                                continue
                            attempts[identifier] = attempts.get(identifier, 0) + 1
                            last_request[identifier] = now
                            name = f"track_{identifier:04d}_try_{attempts[identifier]:02d}"
                            np.savez_compressed(output / "inputs" / f"{name}.npz", points=cloud,
                                                track_position=position, capture_timestamp=last_frame)
                            visuals = save_input_visuals(output / "inputs" / name, frame, identifier)
                            save_json(output / "inputs" / f"{name}.json", provenance)
                            candidate_cfg = dict(cfg["graspgenx"])
                            candidate_cfg["preferred_intercept_x"] = float(env.scene_config.get("workspace", {}).get("preferred_grasp_x", .08))
                            snapshot = control_env.candidate_snapshot()
                            future = executor.submit(infer_and_select, backend, cloud,
                                                     int(exp["seed"]) + 100 * identifier + attempts[identifier],
                                                     cfg["graspgenx"]["generated_candidates"], cfg["graspgenx"]["returned_candidates"],
                                                     snapshot, position, candidate_cfg)
                            pending = {"future": future, "track_id": identifier,
                                       "track_position": position, "capture_sim_s": last_frame,
                                       "name": name, "visuals": visuals,
                                       "scene_epoch": int(env._scene_epoch), "provenance": provenance}
            control_started = time.perf_counter()
            action = adapter.act(obs)
            policy_info = adapter.policy.get_info()
            active_id = adapter.policy.target_id
            target_sensor = next((x for x in obs.get("objects", []) if x["object_id"] == active_id), {})
            control_stream.write(json.dumps(serializable({
                "sim_s": now, "stage": str(adapter.policy.state), "track_id": active_id,
                "ee_pose": obs["ee_pose"], "joint_pos": obs["joint_pos"],
                "jaw_motor_sensors": obs.get("gripper"),
                "visual": {k: target_sensor.get(k) for k in
                    ("position", "estimated_velocity", "capture_timestamp", "measurement_timestamp",
                     "timestamp", "observation_age", "track_observation_age", "visible",
                     "estimated_grasp_time", "grasp_window")},
                "action": action, "last_error": policy_info.get("last_error"),
                **({"continuous_intercept": policy_info["continuous_intercept"],
                    "continuous_reference": continuous_reference(adapter.policy, now)}
                   if control["moving_approach_mode"] == "continuous" else {})}),
                ensure_ascii=False) + "\n")
            diagnostics = adapter.policy.closure_diagnostics
            if diagnostics and abs(diagnostics[-1]["sim_s"] - now) < 1e-9:
                diagnostics[-1]["outgoing_action"] = action
                diagnostics[-1]["last_error_after_command"] = adapter.policy.get_info().get("last_error")
                contacts = []
                import mujoco
                for contact_index in range(env.data.ncon):
                    contact = env.data.contact[contact_index]
                    names = [env.model.geom(int(identifier)).name for identifier in (contact.geom1, contact.geom2)]
                    if any(name and (name.startswith("object_") or "gripper" in name) for name in names):
                        forces = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, contact_index, forces)
                        contacts.append({"geoms": names, "distance_m": float(contact.dist),
                                         "force_contact_frame": forces})
                diagnostics[-1]["physics_contacts_diagnostics_only"] = contacts
            control_wall_s = time.perf_counter() - control_started
            if control_wall_s > env.control_dt:
                slow_control_calls.append({"sim_s": now, "wall_s": control_wall_s,
                                           "stage": str(adapter.policy.state),
                                           "target_id": adapter.policy.target_id,
                                           "last_error": adapter.policy.get_info().get("last_error")})
            events = adapter.policy.drain_events()
            policy_events.extend(events)
            # One-way physical/evaluation bookkeeping. Its answer never reaches
            # the sensor-only ledger or the next formal policy observation.
            for event in control_env.drain_events():
                env.on_policy_event(event)
            _, _, terminated, truncated, info = env.step(action)
            obs = control_env.get_observation()
            if int(env._scene_epoch) != run_scene_epoch:
                raise RuntimeError("Scene reset during benchmark invalidates track IDs and this evaluation")
            evaluator.update()
            phase = str(getattr(adapter.policy, "state", getattr(adapter.policy, "stage", "unknown")))
            policy_key = (phase, getattr(adapter.policy, "target_id", None))
            if policy_key != last_policy:
                policy_log.append({"sim_s": now, "stage": phase, "target_id": policy_key[1],
                                   "policy_info": adapter.policy.get_info(),
                                   "joint_positions": obs["joint_pos"],
                                   "ee_pose": obs["ee_pose"]})
                last_policy = policy_key
            if viewer is not None:
                if not viewer.is_running():
                    state["interrupted"] = True
                    break
                viewer.sync()
            if video is not None and now <= cfg["video"]["maximum_seconds"] and now + 1e-9 >= next_video_time:
                rgb = env._render_camera(cfg["video"]["camera"], tuple(cfg["video"]["resolution"]), depth=False)
                label = f"GraspGen-X | belt {exp['belt_speed_m_s']:.2f} m/s | t={now:.1f}s | {phase}"
                cv2.putText(rgb, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1, cv2.LINE_AA)
                video.append_data(rgb)
                next_video_time += 1 / cfg["video"]["fps"]
            if now - last_progress >= 5.:
                summary = evaluator.summary(expected_objects=int(exp["objects"]))
                print(f"t={now:.1f}s 已来料{summary['spawned_objects']} 正确落箱{summary['successful_sorted_objects']} 阶段{phase}", flush=True)
                state["interim"] = summary
                state["policy_info"] = adapter.policy.get_info()
                save_json(output / "result.json", state)
                last_progress = now
            if terminated or truncated:
                state["environment_terminated"] = bool(terminated)
                state["environment_truncated"] = bool(truncated)
                if truncated:
                    break
                if terminal_seen_s is None:
                    terminal_seen_s = float(env.data.time)
                # Lifecycle SORTED precedes independently measured settling.
                # Continue ordinary physics; do not end on the first bin hit.
                all_stable = all(r["success"] for r in evaluator.records.values())
                if all_stable or float(env.data.time) - terminal_seen_s >= evaluator.terminal_retention_s - .1:
                    break
            elapsed_wall = time.perf_counter() - wall_start
            elapsed_sim = float(env.data.time) - simulation_start
            lag = elapsed_wall - elapsed_sim
            lag_max = max(lag_max, lag)
            lag_samples.append(lag)
            if not args.fast and lag < 0:
                time.sleep(min(-lag, .02))
        wall_seconds = time.perf_counter() - wall_start
        final_lag = wall_seconds - (float(env.data.time) - simulation_start)
        lag_max = max(lag_max, final_lag)
        lag_samples.append(final_lag)
        summary = evaluator.summary(expected_objects=int(exp["objects"]))
        real_time_factor = (float(env.data.time) - simulation_start) / max(wall_seconds, 1e-9)
        sorting_target_met = summary["success_rate"] >= exp["target_success_rate"]
        timing_target_met = bool(not args.fast and lag_max <= exp["maximum_wall_lag_s"]
                                 and real_time_factor >= exp["minimum_real_time_factor"])
        state.update({"status": "interrupted" if state.get("interrupted") else "completed", "evaluation": summary,
                      "inference_calls": len(inference_log), "policy_transitions": policy_log,
                      "network_updates": adapter.updates, "environment_metrics": env.get_metrics(),
                      "slow_control_calls": slow_control_calls,
                      "timing": {"mode": "unthrottled_offline" if args.fast else "wall_clock_paced",
                                 "wall_seconds": wall_seconds,
                                 "sim_seconds": float(env.data.time) - simulation_start,
                                 "real_time_factor": real_time_factor,
                                 "maximum_wall_lag_s": lag_max,
                                 "p95_wall_lag_s": float(np.percentile(lag_samples, 95)) if lag_samples else 0.,
                                 "inference_asynchronous": True},
                      "sorting_target_met": sorting_target_met,
                      "timing_target_met": timing_target_met,
                      "target_met": bool(sorting_target_met and timing_target_met
                                         and state["acceptance_eligible"] and not state.get("interrupted")),
                      "statistical_target_supported": summary["one_sided_95_percent_lower_bound"] >= exp["target_success_rate"],
                      "limitations": ["Collision meshes use the existing T-LESS convex approximation.",
                                      "Partial single-camera depth; hidden surfaces are not supplied to the network.",
                                      "Fixed camera matches the saved relative pose; conveyor layout remains simulated.",
                                      "Observed success rate and confidence lower bound are distinct.",
                                      "Hardware feedback is emulated; encoder and actuator-effort noise are not modeled.",
                                      "Motor obstruction does not identify the held object; only the independent evaluator scores identity.",
                                      "No simulation result guarantees real-robot success."]})
        save_json(output / "result.json", state)
        write_report(output, state)
        fields = sorted({key for row in summary["objects"] for key in row})
        with (output / "objects.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary["objects"])
        save_json(output / "inference_log.json", inference_log)
        save_json(output / "policy_events.json", policy_events)
        save_json(output / "closure_diagnostics.json", adapter.policy.closure_diagnostics)
        save_json(output / "input_deferrals.json", input_deferrals)
        print(json.dumps(serializable({"output": output, "evaluation": summary,
                                       "target_met": state["target_met"], "timing": state["timing"]}), ensure_ascii=False), flush=True)
        return 0 if state["target_met"] else 2
    except KeyboardInterrupt:
        state.update(status="interrupted", interrupted=True)
        if evaluator is not None:
            state["evaluation"] = evaluator.summary(expected_objects=int(exp["objects"]))
        save_json(output / "result.json", state)
        write_report(output, state)
        return 130
    except Exception as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        if evaluator is not None:
            state["evaluation"] = evaluator.summary(expected_objects=int(exp["objects"]))
        save_json(output / "result.json", state)
        write_report(output, state)
        print(state["traceback"], file=sys.stderr)
        return 1
    finally:
        save_json(output / "policy_events.json", policy_events)
        save_json(output / "input_deferrals.json", input_deferrals)
        executor.shutdown(wait=True, cancel_futures=True)
        if video is not None:
            video.close()
        if control_stream is not None:
            control_stream.close()
        if viewer is not None:
            viewer.close()
        if adapter is not None:
            if hasattr(adapter, "full_observation_trace"):
                save_json(output / "live_full_observation.json", adapter.full_observation_trace.events)
            save_json(output / "closure_diagnostics.json", adapter.policy.closure_diagnostics)
            adapter.close()
        if evaluator is not None:
            evaluator.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
