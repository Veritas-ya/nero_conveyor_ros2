"""Prewarmed PRM route suggestions for the first newly observed workpiece.

The graph depends only on a named robot/fixed-world/gripper snapshot.  It is
built before perception, rather than from the preceding workpiece's executed
trajectory.  Queries connect measured start and newly solved goal joints to
the graph.  Returned routes are *suggestions*: the planning worker still has
to certify all request constraints, current observed points, dynamic objects,
descent and timing.  A missing route or changed signature simply lets the
existing direct/RRT planner continue.

This is a small PRM scaffold inspired by preprocessing/query planners.  It
does not implement CTMP's task-cover construction or promise a query deadline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import copy
import hashlib
import heapq
import json
import threading
import time
from typing import Any

import numpy as np


StateValidity = Callable[[np.ndarray], bool]
EdgeValidity = Callable[[np.ndarray, np.ndarray], bool]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class RoadmapSignature:
    """Inputs whose changes invalidate every preprocessed static certificate.

    The creator owns the meaning of robot_model/static_world: content hashes
    are preferable to paths.  validation_policy must change when collision
    masks, padding, native contact tolerances or edge-check semantics change.
    Current object identity and measured scene points deliberately do not
    belong here: neither is certified by the static roadmap.
    """

    robot_model: str
    static_world: str
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    gripper_snapshot: tuple[float, ...] = ()
    validation_policy: str = "native-static-v1"

    def __post_init__(self) -> None:
        for key in ("robot_model", "static_world", "validation_policy"):
            object.__setattr__(self, key, str(getattr(self, key)))
        for key in ("lower_limits", "upper_limits", "gripper_snapshot"):
            values = tuple(float(value) for value in getattr(self, key))
            if not np.isfinite(values).all():
                raise ValueError(f"{key} must be finite")
            object.__setattr__(self, key, values)
        lower = np.asarray(self.lower_limits)
        upper = np.asarray(self.upper_limits)
        if not lower.size or lower.shape != upper.shape or np.any(lower >= upper):
            raise ValueError("roadmap signature joint limits are invalid")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def structure_digest(self) -> str:
        """Reusable route topology, independent of current jaw geometry.

        Changed jaw geometry invalidates the preheated static proof but can
        still use its corners as suggestions for full current validation.
        """
        values = {key: value for key, value in self.__dict__.items()
                  if key != "gripper_snapshot"}
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WarmRoadmapConfig:
    node_count: int = 192
    neighbors: int = 10
    query_neighbors: int = 8
    max_candidates: int = 3
    # Used only when the caller does not supply its own static edge checker.
    # These sampled edges are route hints, not an adaptive safety certificate.
    normalized_edge_resolution: float = 0.025
    seed: int = 7401
    sample_attempt_multiplier: int = 8
    local_sample_fraction: float = 0.30
    local_sample_scale: float = 0.15

    def __post_init__(self) -> None:
        for name in ("node_count", "neighbors", "query_neighbors", "max_candidates",
                     "sample_attempt_multiplier"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.normalized_edge_resolution) or self.normalized_edge_resolution <= 0:
            raise ValueError("normalized_edge_resolution must be positive")
        if not 0 <= self.local_sample_fraction <= 1:
            raise ValueError("local_sample_fraction must be in [0, 1]")
        if not np.isfinite(self.local_sample_scale) or self.local_sample_scale <= 0:
            raise ValueError("local_sample_scale must be positive")


@dataclass(frozen=True)
class _Graph:
    nodes: np.ndarray
    adjacency: tuple[tuple[tuple[int, float], ...], ...]


def _halton(count: int, dimensions: int, offset: int) -> np.ndarray:
    """Deterministic low-discrepancy samples without a scipy dependency."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < dimensions:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    result = np.empty((count, dimensions), dtype=float)
    for column, base in enumerate(primes):
        indices = np.arange(offset + 1, offset + count + 1, dtype=np.int64)
        values = np.zeros(count)
        weight = 1.0 / base
        while np.any(indices):
            values += weight * (indices % base)
            indices //= base
            weight /= base
        result[:, column] = values
    return result


class WarmRoadmap:
    """Build once under static callbacks; return uncertified per-query routes."""

    def __init__(
        self,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        *,
        signature: RoadmapSignature,
        static_state_valid: StateValidity,
        static_edge_valid: EdgeValidity | None = None,
        config: WarmRoadmapConfig | None = None,
    ) -> None:
        lower = np.asarray(lower_limits, dtype=float).reshape(-1).copy()
        upper = np.asarray(upper_limits, dtype=float).reshape(-1).copy()
        if (not lower.size or lower.shape != upper.shape or not np.isfinite([lower, upper]).all()
                or np.any(lower >= upper)):
            raise ValueError("roadmap joint limits are invalid")
        if tuple(lower) != signature.lower_limits or tuple(upper) != signature.upper_limits:
            raise ValueError("roadmap limits must match its immutable signature")
        if not callable(static_state_valid):
            raise TypeError("static_state_valid must be callable")
        if static_edge_valid is not None and not callable(static_edge_valid):
            raise TypeError("static_edge_valid must be callable")
        lower.setflags(write=False)
        upper.setflags(write=False)
        self.lower, self.upper, self.span = lower, upper, upper - lower
        self.signature = signature
        self.config = config or WarmRoadmapConfig()
        self._state_check = static_state_valid
        self._edge_check = static_edge_valid
        self._lock = threading.RLock()
        self._graph: _Graph | None = None
        self._diagnostics: dict[str, Any] = {
            "backend": "prewarmed_static_prm_scaffold",
            "signature_sha256": signature.digest,
            "ready": False,
            "certification": "route_hint_requires_current_request_full_validation",
            "prepares": 0,
            "queries": 0,
            "prior_workpiece_trajectories_used": 0,
        }

    @property
    def ready(self) -> bool:
        return self._graph is not None

    @property
    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._diagnostics)

    def _state(self, value: Any) -> np.ndarray | None:
        try:
            state = np.asarray(value, dtype=float).reshape(-1)
        except (ValueError, TypeError):
            return None
        if (state.shape != self.lower.shape or not np.isfinite(state).all()
                or np.any(state < self.lower) or np.any(state > self.upper)):
            return None
        return state

    def _valid_state(self, q: np.ndarray) -> bool:
        candidate = self._state(q)
        if candidate is None:
            return False
        protected = candidate.copy()
        protected.setflags(write=False)
        return bool(self._state_check(protected))

    def _valid_edge(self, a: np.ndarray, b: np.ndarray, cancelled: CancellationCheck) -> bool:
        if cancelled():
            return False
        if self._edge_check is not None:
            # The callbacks cannot alter nodes in the published graph.
            left, right = a.copy(), b.copy()
            left.setflags(write=False)
            right.setflags(write=False)
            return bool(self._edge_check(left, right)) and not cancelled()
        steps = max(1, int(np.ceil(np.linalg.norm((b - a) / self.span)
                                   / self.config.normalized_edge_resolution)))
        return all(not cancelled() and self._valid_state(a + (b - a) * (index / steps))
                   for index in range(steps + 1))

    def prepare(
        self,
        seed_states: Sequence[Sequence[float]] = (),
        *,
        cancelled: CancellationCheck | None = None,
    ) -> bool:
        """Preprocess immutable nodes/edges before a new object is observed.

        Seed states may be nominal standby configurations or several high-pose
        IK branches for the calibrated interception region.  Random samples
        supplement rather than replace those task-specific seeds.  Exceptions
        become a diagnostic and False, allowing the caller's original planner.
        A cancelled/failed build never publishes a partial graph.
        """
        check_cancelled = cancelled or (lambda: False)
        started = time.perf_counter()
        with self._lock:
            if self._graph is not None:
                self._diagnostics["prepare_reused"] = True
                return True
            self._diagnostics["prepares"] += 1
            counts = {"sample_attempts": 0, "state_checks": 0, "edge_checks": 0,
                      "edge_count": 0, "valid_seed_count": 0}
            try:
                nodes: list[np.ndarray] = []
                seen: set[bytes] = set()

                def append(q: Any, *, seed: bool = False) -> None:
                    candidate = self._state(q)
                    if candidate is None or candidate.tobytes() in seen:
                        return
                    counts["state_checks"] += 1
                    if self._valid_state(candidate):
                        nodes.append(candidate.copy())
                        seen.add(candidate.tobytes())
                        counts["valid_seed_count"] += int(seed)

                for seed in seed_states:
                    if check_cancelled():
                        break
                    append(seed, seed=True)
                anchors = tuple(q.copy() for q in nodes)
                if len(nodes) < self.config.node_count and not check_cancelled():
                    attempts = self.config.node_count * self.config.sample_attempt_multiplier
                    samples = _halton(attempts, len(self.lower), max(0, int(self.config.seed)))
                    for index, unit in enumerate(samples):
                        if len(nodes) >= self.config.node_count or check_cancelled():
                            break
                        counts["sample_attempts"] += 1
                        if anchors and index % 100 < round(100 * self.config.local_sample_fraction):
                            anchor = anchors[index % len(anchors)]
                            q = np.clip(anchor + (2 * unit - 1) * self.span
                                        * self.config.local_sample_scale, self.lower, self.upper)
                        else:
                            q = self.lower + unit * self.span
                        append(q)
                if check_cancelled():
                    self._diagnostics.update(last_prepare_reason="cancelled", **counts)
                    return False
                if len(nodes) < 2:
                    self._diagnostics.update(last_prepare_reason="insufficient_valid_nodes", **counts)
                    return False
                array = np.asarray(nodes)
                normalized = (array - self.lower) / self.span
                distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
                adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
                tested: set[tuple[int, int]] = set()
                for first in range(len(nodes)):
                    nearest = np.argsort(distances[first], kind="stable")
                    for second in nearest[1:self.config.neighbors + 1]:
                        if check_cancelled():
                            break
                        pair = (min(first, int(second)), max(first, int(second)))
                        if pair in tested:
                            continue
                        tested.add(pair)
                        counts["edge_checks"] += 1
                        if self._valid_edge(array[first], array[second], check_cancelled):
                            cost = float(distances[first, second])
                            adjacency[first].append((int(second), cost))
                            adjacency[second].append((first, cost))
                            counts["edge_count"] += 1
                    if check_cancelled():
                        break
                if check_cancelled():
                    self._diagnostics.update(last_prepare_reason="cancelled", **counts)
                    return False
                array.setflags(write=False)
                self._graph = _Graph(array, tuple(tuple(sorted(edges)) for edges in adjacency))
                self._diagnostics.update(ready=True, node_count=len(array), **counts,
                                         last_prepare_reason="completed", prepare_reused=False)
                return True
            except Exception as error:
                self._diagnostics.update(last_prepare_reason=f"{type(error).__name__}: {error}", **counts)
                return False
            finally:
                self._diagnostics["prepare_wall_s"] = time.perf_counter() - started

    @staticmethod
    def _shortest(
        graph: _Graph,
        starts: dict[int, float],
        goals: dict[int, float],
        blocked: frozenset[tuple[int, int]] = frozenset(),
    ) -> tuple[float, list[int]] | None:
        # Virtual source and sink are represented by initial/final costs.
        queue = [(cost, node) for node, cost in starts.items()]
        heapq.heapify(queue)
        distances = dict(starts)
        parents = {node: -1 for node in starts}
        best_cost, best_goal = float("inf"), -1
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distances.get(node) or cost >= best_cost:
                continue
            if node in goals and cost + goals[node] < best_cost:
                best_cost, best_goal = cost + goals[node], node
            for other, edge_cost in graph.adjacency[node]:
                if (min(node, other), max(node, other)) in blocked:
                    continue
                candidate = cost + edge_cost
                if candidate < distances.get(other, float("inf")):
                    distances[other] = candidate
                    parents[other] = node
                    heapq.heappush(queue, (candidate, other))
        if best_goal < 0:
            return None
        path = [best_goal]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        return best_cost, path[::-1]

    def propose(
        self,
        start_q: Sequence[float],
        goal_q: Sequence[float],
        *,
        signature: RoadmapSignature,
        cancelled: CancellationCheck | None = None,
        max_candidates: int | None = None,
        runtime_geometry_key: str | None = None,
    ) -> list[np.ndarray]:
        """Connect a first-object query to prewarmed static graph routes.

        Query endpoints and connectors are checked under the same static
        callback.  All graph nodes/edges are reused without native FK here.
        Current scene/object checks remain the caller's final certification.
        No request timestamp, object ID, previous trajectory or live scene is
        stored in the graph.  Signature mismatch returns no hint.
        """
        started = time.perf_counter()
        check_cancelled = cancelled or (lambda: False)
        with self._lock:
            self._diagnostics["queries"] += 1
            self._diagnostics["last_query_candidates"] = 0
            self._diagnostics["last_query_connector_checks"] = 0
            try:
                self._diagnostics["last_runtime_geometry_key"] = runtime_geometry_key
                self._diagnostics["static_certificate_applicable"] = signature == self.signature
                if signature.structure_digest != self.signature.structure_digest:
                    self._diagnostics["last_query_reason"] = "signature_changed_use_original_planner"
                    return []
                graph = self._graph
                if graph is None:
                    self._diagnostics["last_query_reason"] = "not_prepared_use_original_planner"
                    return []
                start, goal = self._state(start_q), self._state(goal_q)
                if start is None or goal is None:
                    self._diagnostics["last_query_reason"] = "endpoints_outside_preprocessed_limits"
                    return []
                if check_cancelled() or not self._valid_state(start) or not self._valid_state(goal):
                    self._diagnostics["last_query_reason"] = "static_endpoint_rejected_or_cancelled"
                    return []

                def connectors(endpoint: np.ndarray) -> dict[int, float]:
                    distances = np.linalg.norm((graph.nodes - endpoint) / self.span, axis=1)
                    result = {}
                    for node in np.argsort(distances, kind="stable")[:self.config.query_neighbors]:
                        if check_cancelled():
                            break
                        self._diagnostics["last_query_connector_checks"] += 1
                        if self._valid_edge(endpoint, graph.nodes[node], check_cancelled):
                            result[int(node)] = float(distances[node])
                    return result

                starts, goals = connectors(start), connectors(goal)
                if check_cancelled() or not starts or not goals:
                    self._diagnostics["last_query_reason"] = "no_static_connector_or_cancelled"
                    return []
                best = self._shortest(graph, starts, goals)
                if best is None:
                    self._diagnostics["last_query_reason"] = "graph_does_not_cover_query"
                    return []
                limit = self.config.max_candidates if max_candidates is None else max(0, int(max_candidates))
                if limit == 0:
                    self._diagnostics["last_query_reason"] = "no_candidates_requested"
                    return []
                options = [best]
                seen_paths = {tuple(best[1])}
                # A small set of alternate topologies lets the full online
                # checker reject a dynamic obstruction without trusting it as
                # permanently static or immediately restarting global RRT.
                for left, right in zip(best[1], best[1][1:]):
                    if check_cancelled() or len(options) >= limit * 3:
                        break
                    pair = (min(left, right), max(left, right))
                    option = self._shortest(graph, starts, goals, frozenset((pair,)))
                    if option is not None and tuple(option[1]) not in seen_paths:
                        options.append(option)
                        seen_paths.add(tuple(option[1]))
                result = []
                for _, indices in sorted(options, key=lambda item: (item[0], item[1]))[:limit]:
                    route = [start]
                    for node in indices:
                        if not np.array_equal(route[-1], graph.nodes[node]):
                            route.append(graph.nodes[node])
                    if not np.array_equal(route[-1], goal):
                        route.append(goal)
                    result.append(np.asarray(route).copy())
                if check_cancelled():
                    self._diagnostics["last_query_reason"] = "cancelled"
                    return []
                self._diagnostics.update(last_query_reason="static_routes_require_online_certification",
                                         last_query_candidates=len(result))
                return result
            except Exception as error:
                self._diagnostics["last_query_reason"] = f"{type(error).__name__}: {error}"
                return []
            finally:
                self._diagnostics["query_wall_s"] = time.perf_counter() - started


def compiled_model_digest(model: Any) -> str:
    """Hash the complete compiled model, including mesh/masks/transforms."""
    import mujoco

    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


def runtime_geometry_signature(
    signature: RoadmapSignature,
    *,
    gripper_qpos: Sequence[float] = (),
    scene_key: Any = None,
    object_key: Any = None,
    request_constraints_key: Any = None,
) -> str:
    """Record online geometry identity without authorizing cached validity.

    Callers should pass immutable content hashes/version tokens for measured
    scene and object geometry.  This helper is only a diagnostic/cache key;
    no point cloud, target motion or constraints are certified by the PRM.
    """
    values = {"structure_sha256": signature.structure_digest,
              "gripper_qpos": tuple(float(value) for value in gripper_qpos),
              "scene_key": scene_key, "object_key": object_key,
              "request_constraints_key": request_constraints_key}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_mujoco_roadmap(
    model: Any,
    *,
    joint_names: Sequence[str],
    full_qpos: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    robot_collision_body_ids: Sequence[int],
    ee_site_name: str | None = None,
    gripper_qpos_indices: Sequence[int] = (),
    seed_states: Sequence[Sequence[float]] = (),
    fixed_world_version: str = "compiled_fixed_world",
    static_state_validator: StateValidity | None = None,
    static_validation_version: str = "native-contact",
    config: WarmRoadmapConfig | None = None,
    cancelled: CancellationCheck | None = None,
) -> WarmRoadmap:
    """Build a private native self/fixed-world roadmap during device startup.

    The fixed qpos snapshot includes the open-jaw configuration used for
    preheating.  Native ``object_*`` contacts are omitted, matching the global
    worker's static-contact semantics; dynamic objects and measured scene are
    always checked online.  Height, approach alignment, singularity, target
    and descent are request-specific and deliberately not pre-certified.
    ``ee_site_name`` is checked when supplied, making accidental model/TCP
    mismatch visible during startup; task IK branch seeds may be supplied.
    """
    import mujoco

    static_model = copy.copy(model)
    scratch = mujoco.MjData(static_model)
    if ee_site_name is not None:
        static_model.site(str(ee_site_name))
    arm_indices = np.asarray([int(static_model.jnt_qposadr[static_model.joint(str(name)).id])
                              for name in joint_names], dtype=int)
    snapshot = np.asarray(full_qpos, dtype=float).reshape(static_model.nq).copy()
    if not np.isfinite(snapshot).all():
        raise ValueError("full_qpos must be finite")
    gripper_indices = np.asarray(gripper_qpos_indices, dtype=int).reshape(-1)
    if np.any(gripper_indices < 0) or np.any(gripper_indices >= static_model.nq):
        raise ValueError("gripper qpos index is outside model")
    lower = np.asarray(lower_limits, dtype=float).reshape(len(arm_indices))
    upper = np.asarray(upper_limits, dtype=float).reshape(len(arm_indices))
    # Only non-arm/non-jaw coordinates define fixed-world transforms.  Jaw
    # changes have their own proof snapshot and do not erase reusable corners.
    dynamic_indices: list[int] = []
    for joint in range(static_model.njnt):
        body = int(static_model.jnt_bodyid[joint])
        if (static_model.body(body).name or "").startswith("object_"):
            begin = int(static_model.jnt_qposadr[joint])
            end = (int(static_model.jnt_qposadr[joint + 1])
                   if joint + 1 < static_model.njnt else static_model.nq)
            dynamic_indices.extend(range(begin, end))
    fixed_indices = np.setdiff1d(np.arange(static_model.nq),
                                np.concatenate((arm_indices, gripper_indices,
                                                np.asarray(dynamic_indices, dtype=int))))
    world = hashlib.sha256(str(fixed_world_version).encode("utf-8")
                           + fixed_indices.tobytes() + snapshot[fixed_indices].tobytes()).hexdigest()
    signature = RoadmapSignature(
        compiled_model_digest(static_model), world, tuple(lower), tuple(upper),
        tuple(snapshot[gripper_indices]),
        "mujoco-fwdPosition-robot-contact<=1e-6-except-object_prefix-v1",
    )
    robot_bodies = frozenset(int(value) for value in robot_collision_body_ids)
    # Collision body selection is part of the static certificate semantics.
    signature = RoadmapSignature(
        signature.robot_model,
        hashlib.sha256((signature.static_world + repr(sorted(robot_bodies))).encode("utf-8")).hexdigest(),
        signature.lower_limits, signature.upper_limits, signature.gripper_snapshot,
        signature.validation_policy + ":" + str(static_validation_version),
    )
    body_names = tuple(static_model.body(index).name or "" for index in range(static_model.nbody))
    checks = {"static_fk_calls": 0}

    def state_valid(q: np.ndarray) -> bool:
        scratch.qpos[:] = snapshot
        scratch.qpos[arm_indices] = q
        scratch.qvel[:] = 0.0
        # Position stage supplies native collision state without computing
        # velocities, actuation, acceleration or constraint forces.
        mujoco.mj_fwdPosition(static_model, scratch)
        checks["static_fk_calls"] += 1
        for contact in scratch.contact[:scratch.ncon]:
            first, second = (int(static_model.geom_bodyid[int(contact.geom1)]),
                             int(static_model.geom_bodyid[int(contact.geom2)]))
            if first not in robot_bodies and second not in robot_bodies:
                continue
            other = second if first in robot_bodies else first
            if body_names[other].startswith("object_"):
                continue
            if float(contact.dist) <= 1e-6:
                return False
        return static_state_validator is None or bool(static_state_validator(q))

    roadmap = WarmRoadmap(lower, upper, signature=signature,
                          static_state_valid=state_valid, config=config)
    # Keep static inputs alive and make their provenance available to the
    # startup report without exposing scratch mutable model/data to a caller.
    roadmap._mujoco_checks = checks
    roadmap._mujoco_snapshot = snapshot.copy()
    roadmap._gripper_qpos_indices = tuple(int(value) for value in gripper_indices)
    margin = np.minimum(0.02, (upper - lower) * 0.05)
    clamp = lambda q: np.clip(np.asarray(q, dtype=float), lower + margin, upper - margin)
    anchors = [clamp(snapshot[arm_indices]), clamp(static_model.qpos0[arm_indices]),
               clamp(np.zeros(len(arm_indices))), (lower + upper) * 0.5]
    # Several nominal, bounded alternatives improve coverage when the current
    # folded arm state itself is self-colliding.  Actual native checks decide
    # whether any such seed is usable; none is a robot motion command.
    base = anchors[0]
    for axis in range(len(arm_indices)):
        for sign in (-1.0, 1.0):
            offset = np.zeros(len(arm_indices))
            offset[axis] = sign * 0.15 * (upper[axis] - lower[axis])
            anchors.append(clamp(base + offset))
    roadmap.prepare((*seed_states, *anchors), cancelled=cancelled)
    # The PRM still supplies only route hints.  Separately prove convex boxes
    # for native self/fixed-world contacts; measured clouds and all request
    # conditions are deliberately absent from this reusable certificate.
    roadmap.static_native_regions = None
    if roadmap.ready and ee_site_name is not None:
        try:
            from .static_native_regions import prepare_static_native_regions
            joint_ids = [int(static_model.joint(str(name)).id) for name in joint_names]
            regions = prepare_static_native_regions(static_model,
                arm_joint_ids=joint_ids, ee_site_id=int(static_model.site(str(ee_site_name)).id),
                robot_body_ids=robot_bodies, full_qpos=snapshot,
                lower=lower, upper=upper, seeds=roadmap._graph.nodes,
                model_digest=signature.robot_model, cancelled=cancelled)
            roadmap.static_native_regions = regions
            roadmap._diagnostics['static_native_regions'] = (
                dict(regions.report) if regions is not None else
                {'enabled': False, 'reason': 'unsupported_or_cancelled_original_checks_retained'})
        except Exception as error:
            roadmap._diagnostics['static_native_regions'] = {
                'enabled': False, 'reason': f'{type(error).__name__}: {error}'}
    roadmap._diagnostics["static_fk_calls"] = checks["static_fk_calls"]
    roadmap._diagnostics["preheated_gripper_qpos"] = snapshot[gripper_indices].tolist()
    return roadmap
