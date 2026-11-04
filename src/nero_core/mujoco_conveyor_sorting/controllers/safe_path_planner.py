"""Deterministic constrained RRT-Connect for short manipulator transfers.

The dynamic conveyor interception remains a local Cartesian-servo problem.
Once an object is lifted, however, the seven-axis arm may need to move around
joint-limit and wrist-branch barriers.  This small planner supplies that global
layer without adding a ROS/MoveIt dependency.  Callers define validity, so the
same search can enforce tool orientation, soft joint limits, workspace height,
and MuJoCo collision checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np


StateValidity = Callable[[np.ndarray], bool]
StateBatchValidity = Callable[[np.ndarray], bool]
SegmentValidity = Callable[[np.ndarray, np.ndarray], bool]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class SafePathConfig:
    normalized_step_size: float = 0.10
    normalized_edge_resolution: float = 0.018
    execution_joint_step: float = 0.08
    max_iterations: int = 2400
    goal_sample_probability: float = 0.18
    shortcut_attempts: int = 120

    def __post_init__(self) -> None:
        if self.normalized_step_size <= 0.0:
            raise ValueError("normalized_step_size must be positive")
        if self.normalized_edge_resolution <= 0.0:
            raise ValueError("normalized_edge_resolution must be positive")
        if self.execution_joint_step <= 0.0:
            raise ValueError("execution_joint_step must be positive")
        if self.max_iterations < 1 or self.shortcut_attempts < 0:
            raise ValueError("planner iteration counts are invalid")
        if not 0.0 <= self.goal_sample_probability <= 1.0:
            raise ValueError("goal_sample_probability must be in [0, 1]")


class _Tree:
    def __init__(self, root: np.ndarray) -> None:
        self.nodes: list[np.ndarray] = [root.copy()]
        self.parents: list[int] = [-1]

    def add(self, state: np.ndarray, parent: int) -> int:
        self.nodes.append(state.copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def trace(self, index: int) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        while index >= 0:
            result.append(self.nodes[index].copy())
            index = self.parents[index]
        result.reverse()
        return result


class ConstrainedRRTConnect:
    """Bidirectional joint-space search with dense edge validation."""

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        state_valid: StateValidity,
        *,
        config: SafePathConfig | None = None,
        seed: int = 0,
        cancelled: CancellationCheck | None = None,
        states_valid: StateBatchValidity | None = None,
        segment_valid: SegmentValidity | None = None,
    ) -> None:
        self.lower = np.asarray(lower_limits, dtype=float).reshape(-1)
        self.upper = np.asarray(upper_limits, dtype=float).reshape(-1)
        if (
            self.lower.size == 0
            or self.lower.shape != self.upper.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.lower >= self.upper)
        ):
            raise ValueError("planner joint limits are invalid")
        if not callable(state_valid):
            raise TypeError("state_valid must be callable")
        self.span = self.upper - self.lower
        self._state_validity = state_valid
        self.config = SafePathConfig() if config is None else config
        self.rng = np.random.default_rng(seed)
        self._cancellation_check = cancelled
        self._batch_state_validity = states_valid
        self._segment_validity = segment_valid

    def _cancelled(self) -> bool:
        check = self._cancellation_check
        return bool(check is not None and check())

    def _coerce_state(self, state: np.ndarray) -> np.ndarray | None:
        """Return a flat state with the configured DoF, or ``None``."""

        try:
            candidate = np.asarray(state, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if candidate.shape != self.lower.shape:
            return None
        return candidate

    def state_valid(self, state: np.ndarray) -> bool:
        """Check dimensions, finiteness, joint limits, and caller constraints.

        Keeping the configured bounds in this public check prevents callers
        from accidentally validating a collision-free state that is outside
        the robot's usable joint range.
        """

        if self._cancelled():
            return False
        candidate = self._coerce_state(state)
        if candidate is None or not np.all(np.isfinite(candidate)):
            return False
        if np.any(candidate < self.lower) or np.any(candidate > self.upper):
            return False
        return bool(self._state_validity(candidate))

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm((right - left) / self.span))

    def _nearest(self, tree: _Tree, target: np.ndarray) -> int:
        distances = [self._distance(node, target) for node in tree.nodes]
        return int(np.argmin(distances))

    def edge_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Validate both endpoints and a resolution-bounded edge sampling."""

        start_state = self._coerce_state(start)
        end_state = self._coerce_state(end)
        if start_state is None or end_state is None:
            return False
        if (self._cancelled() or not np.isfinite([start_state, end_state]).all()
                or np.any(start_state < self.lower) or np.any(start_state > self.upper)
                or np.any(end_state < self.lower) or np.any(end_state > self.upper)):
            return False
        if self._segment_validity is not None:
            return bool(self._segment_validity(start_state, end_state)) and not self._cancelled()
        distance = self._distance(start_state, end_state)
        samples = max(
            1,
            int(np.ceil(distance / self.config.normalized_edge_resolution)),
        )
        if self._batch_state_validity is not None:
            # Exactly the original endpoints and resolution-bounded samples.
            # Only their evaluation is batched; sample spacing is unchanged.
            states = np.array([start_state, end_state, *(
                start_state+(end_state-start_state)*(index/samples)
                for index in range(1, samples))])
            if (self._cancelled() or not np.isfinite(states).all()
                    or np.any(states < self.lower) or np.any(states > self.upper)):
                return False
            return bool(self._batch_state_validity(states)) and not self._cancelled()
        if not self.state_valid(start_state) or not self.state_valid(end_state):
            return False
        for index in range(1, samples):
            if self._cancelled():
                return False
            state = start_state + (end_state - start_state) * (
                index / samples
            )
            if not self.state_valid(state):
                return False
        return True

    def _edge_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Compatibility wrapper for code using the former private helper."""

        return self.edge_valid(start, end)

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        nearest_index = self._nearest(tree, target)
        nearest = tree.nodes[nearest_index]
        distance = self._distance(nearest, target)
        if distance <= 1e-12:
            return "reached", nearest_index
        fraction = min(1.0, self.config.normalized_step_size / distance)
        # A reached state owns the exact caller-supplied endpoint, including
        # any request-scoped continuation certificate attached to its bytes.
        # Subtracting and adding at fraction=1 can change a float by one ulp.
        candidate = (
            target.copy() if fraction >= 1.0
            else nearest + fraction * (target - nearest)
        )
        candidate = np.clip(candidate, self.lower, self.upper)
        if not self.edge_valid(nearest, candidate):
            return "trapped", nearest_index
        new_index = tree.add(candidate, nearest_index)
        return ("reached" if fraction >= 1.0 else "advanced"), new_index

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        status = "advanced"
        index = 0
        while status == "advanced" and not self._cancelled():
            status, index = self._extend(tree, target)
        if self._cancelled():
            return "trapped", index
        return status, index

    def _shortcut(self, path: list[np.ndarray]) -> list[np.ndarray]:
        result = [state.copy() for state in path]
        for _ in range(self.config.shortcut_attempts):
            if self._cancelled():
                break
            if len(result) < 3:
                break
            first, second = sorted(
                self.rng.choice(len(result), size=2, replace=False).tolist()
            )
            if second <= first + 1:
                continue
            if self.edge_valid(result[first], result[second]):
                result = result[: first + 1] + result[second:]
        return result

    def densify(
        self, path: Sequence[np.ndarray]
    ) -> list[np.ndarray] | None:
        """Densify a path and validate every edge that will be executed.

        ``None`` denotes an invalid waypoint or edge.  An empty path remains
        empty, and a valid single-state path is returned as a defensive copy.
        """

        states: list[np.ndarray] = []
        for state in path:
            if self._cancelled():
                return None
            candidate = self._coerce_state(state)
            if candidate is None or not self.state_valid(candidate):
                return None
            states.append(candidate.copy())
        if not states:
            return []

        dense = [states[0].copy()]
        for start, end in zip(states, states[1:]):
            if self._cancelled():
                return None
            samples = max(
                1,
                int(
                    np.ceil(
                        np.max(np.abs(end - start))
                        / self.config.execution_joint_step
                    )
                ),
            )
            dense.extend(
                start + (end - start) * (index / samples)
                for index in range(1, samples)
            )
            # Preserve the exact already-validated waypoint rather than
            # recomputing start+(end-start)*1. Every new edge is still checked
            # below, and interior sampling stays on the original grid.
            dense.append(end.copy())
        result = [np.asarray(state, dtype=float).copy() for state in dense]
        if any(
            not self.edge_valid(start, end)
            for start, end in zip(result, result[1:])
        ):
            return None
        return result

    def _densify(
        self, path: Sequence[np.ndarray]
    ) -> list[np.ndarray] | None:
        """Compatibility wrapper for code using the former private helper."""

        return self.densify(path)

    def direct_path(
        self, start: np.ndarray, goal: np.ndarray
    ) -> list[np.ndarray] | None:
        """Return a safe executable straight-line path when one exists."""

        start_state = self._coerce_state(start)
        goal_state = self._coerce_state(goal)
        if start_state is None or goal_state is None:
            return None
        if not self.edge_valid(start_state, goal_state):
            return None
        return self.densify([start_state, goal_state])

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        check_direct: bool = True,
    ) -> list[np.ndarray] | None:
        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if start.shape != self.lower.shape or goal.shape != self.lower.shape:
            raise ValueError("start and goal must match configured DoF")
        if self._cancelled():
            return None
        if not self.state_valid(start) or not self.state_valid(goal):
            return None
        if check_direct:
            direct = self.direct_path(start, goal)
            if direct is not None:
                return direct

        start_tree = _Tree(start)
        goal_tree = _Tree(goal)
        tree_a, tree_b = start_tree, goal_tree
        a_is_start = True

        for _ in range(self.config.max_iterations):
            if self._cancelled():
                return None
            if self.rng.random() < self.config.goal_sample_probability:
                sample = tree_b.nodes[0]
            else:
                sample = self.rng.uniform(self.lower, self.upper)
            status_a, index_a = self._extend(tree_a, sample)
            if status_a != "trapped":
                status_b, index_b = self._connect(
                    tree_b, tree_a.nodes[index_a]
                )
                if status_b == "reached":
                    if a_is_start:
                        from_start = tree_a.trace(index_a)
                        from_goal = tree_b.trace(index_b)
                    else:
                        from_start = tree_b.trace(index_b)
                        from_goal = tree_a.trace(index_a)
                    path = from_start + list(reversed(from_goal))[1:]
                    return self.densify(self._shortcut(path))
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
        return None


__all__ = [
    "CancellationCheck",
    "ConstrainedRRTConnect",
    "SafePathConfig",
]
