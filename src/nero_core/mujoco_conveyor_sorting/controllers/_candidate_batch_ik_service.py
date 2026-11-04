"""Private optional cuRobo process. stdout is line-delimited JSON only."""
from __future__ import annotations

import contextlib
import json
import sys
import time


def main():
    protocol = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.robot import RobotConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
            urdf, num_seeds, batch_size, return_seeds = sys.argv[1:]
            num_seeds, batch_size, return_seeds = map(int, (num_seeds, batch_size, return_seeds))
            args = TensorDeviceType()
            robot = RobotConfig.from_basic(urdf, "mujoco_world", "nero_tcp", args)
            config = IKSolverConfig.load_from_robot_config(robot, None,
                tensor_args=args, num_seeds=num_seeds, position_threshold=.0002,
                rotation_threshold=.0015, self_collision_check=False,
                self_collision_opt=False, use_cuda_graph=True, high_precision=True)
            solver = IKSolver(config)
            q = solver.sample_configs(batch_size)
            fk = solver.fk(q)
            solver.solve_batch(Pose(fk.ee_position, fk.ee_quaternion),
                retract_config=q, return_seeds=return_seeds)
            torch.cuda.synchronize()
            ready = {"ready": True, "batch_size": batch_size, "num_seeds": num_seeds,
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "memory_allocated_mib": torch.cuda.memory_allocated()/1024**2,
                "memory_reserved_mib": torch.cuda.memory_reserved()/1024**2,
                "collision_check": False, "joint_names": solver.joint_names}
        protocol.write(json.dumps(ready)+"\n")
        protocol.flush()
    except Exception as error:
        protocol.write(json.dumps({"error": type(error).__name__+": "+str(error)})+"\n")
        protocol.flush()
        return
    previous_bounds = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                started = time.perf_counter()
                bounds_array = np.asarray([message["lower_limits"], message["upper_limits"]])
                if previous_bounds is None or not np.array_equal(previous_bounds, bounds_array):
                    bounds = args.to_device(bounds_array)
                    for rollout in solver.get_all_rollout_instances():
                        rollout.dynamics_model.get_state_bounds().position.copy_(bounds)
                        rollout.dynamics_model.joint_limits.position.copy_(bounds)
                        for attribute in ("bound_cost", "bound_constraint"):
                            cost = getattr(rollout, attribute, None)
                            if cost is not None:
                                cost.joint_limits.position.copy_(bounds)
                    for optimizer in solver.solver.optimizers:
                        if optimizer.rollout_fn.dynamics_model.control_space.name == "POSITION":
                            repeats = optimizer.action_lows.numel()//bounds.shape[1]
                            optimizer.action_lows.copy_(bounds[0].repeat(repeats))
                            optimizer.action_highs.copy_(bounds[1].repeat(repeats))
                            if hasattr(optimizer, "action_range"):
                                optimizer.action_range.copy_(optimizer.action_highs-optimizer.action_lows)
                            if hasattr(optimizer, "action_step_max"):
                                optimizer.action_step_max.copy_(optimizer.step_scale*torch.abs(optimizer.action_range))
                    solver.q_sample_gen.low_bounds.copy_(bounds[0])
                    solver.q_sample_gen.range_b.copy_(bounds[1]-bounds[0])
                    previous_bounds = bounds_array.copy()
                targets = np.asarray(message["target_poses"])
                dof = bounds_array.shape[1]
                supplied = np.asarray(message["seeds"]).reshape(-1, dof)[:num_seeds]
                retract = args.to_device(message["start_q"]).reshape(1, dof).repeat(batch_size, 1)
                seed_tensor = args.to_device(supplied).reshape(1, -1, dof).repeat(batch_size, 1, 1)
                solutions, batch_count = [], 0
                for offset in range(0, len(targets), batch_size):
                    chunk = targets[offset:offset+batch_size]
                    count = len(chunk)
                    padded = np.concatenate((chunk, np.repeat(chunk[-1:], batch_size-count, axis=0)))
                    result = solver.solve_batch(Pose(args.to_device(padded[:, :3]),
                        args.to_device(padded[:, 3:])), retract_config=retract,
                        seed_config=seed_tensor, return_seeds=return_seeds)
                    torch.cuda.synchronize()
                    accepted = result.success.detach().cpu().numpy().reshape(batch_size, -1)
                    qs = result.solution.detach().cpu().numpy().reshape(batch_size, -1, dof)
                    for i in range(count):
                        for q, valid in zip(qs[i], accepted[i]):
                            if valid:
                                solutions.append({"pose_index": offset+i, "q": q.tolist()})
                    batch_count += 1
                response = {"solutions": solutions, "diagnostics": {
                    "solve_s": time.perf_counter()-started, "batch_count": batch_count,
                    "target_count": len(targets), "returned_solutions": len(solutions),
                    "memory_allocated_mib": torch.cuda.memory_allocated()/1024**2,
                    "memory_reserved_mib": torch.cuda.memory_reserved()/1024**2}}
        except Exception as error:
            response = {"error": type(error).__name__+": "+str(error)}
        protocol.write(json.dumps(response)+"\n")
        protocol.flush()


if __name__ == "__main__":
    main()
