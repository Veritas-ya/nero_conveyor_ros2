"""Deterministic-seed automatic workpiece generation for the body pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .object_manager import NoAvailableObjectSlot, ObjectInstance, ObjectManager
from .object_spec import ObjectSpec


@dataclass(frozen=True, slots=True)
class ObjectSpawnerConfig:
    spawn_x: float = -0.52
    lateral_range: tuple[float, float] = (-0.075, 0.075)
    surface_height: float = 0.20
    z_clearance: float = 0.006
    spacing_range: tuple[float, float] = (0.10, 0.22)
    yaw_range: tuple[float, float] = (-np.pi, np.pi)
    belt_speed_range: tuple[float, float] = (0.15, 0.25)
    velocity_jitter_std: float = 0.008
    lateral_velocity_jitter_std: float = 0.0
    mass_scale_range: tuple[float, float] = (0.90, 1.10)
    friction_scale_range: tuple[float, float] = (0.90, 1.10)
    color_jitter: float = 0.04
    randomize_category: bool = True
    randomize_belt_speed: bool = True
    randomize_lateral_position: bool = True
    randomize_yaw: bool = True
    randomize_mass: bool = True
    randomize_friction: bool = True
    randomize_color: bool = True
    minimum_spawn_interval: float = 0.10
    maximum_active_objects: int = 8

    def __post_init__(self) -> None:
        for name in (
            "lateral_range",
            "spacing_range",
            "yaw_range",
            "belt_speed_range",
            "mass_scale_range",
            "friction_scale_range",
        ):
            pair = _range_pair(getattr(self, name), name)
            object.__setattr__(self, name, pair)
        for name in (
            "spawn_x",
            "surface_height",
            "z_clearance",
            "velocity_jitter_std",
            "lateral_velocity_jitter_std",
            "color_jitter",
            "minimum_spawn_interval",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "z_clearance",
            "velocity_jitter_std",
            "lateral_velocity_jitter_std",
            "color_jitter",
            "minimum_spawn_interval",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.spacing_range[0] <= 0.0:
            raise ValueError("spacing_range must be positive")
        if self.belt_speed_range[0] < 0.0:
            raise ValueError("belt_speed_range cannot be negative")
        if self.mass_scale_range[0] <= 0.0:
            raise ValueError("mass_scale_range must be positive")
        if self.friction_scale_range[0] < 0.0:
            raise ValueError("friction_scale_range cannot be negative")
        if self.maximum_active_objects <= 0:
            raise ValueError("maximum_active_objects must be positive")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ObjectSpawnerConfig":
        """Read a minimal mapping, ``scene.yaml``, or full project config."""

        default = cls()
        root: Mapping[str, Any] = source
        if isinstance(root.get("scene"), Mapping):
            root = root["scene"]  # type: ignore[assignment]
        conveyor = root.get("conveyor", root)
        if not isinstance(conveyor, Mapping):
            raise ValueError("conveyor configuration must be a mapping")
        conveyor_spawn = conveyor.get("spawn", root.get("spawn", {}))
        if not isinstance(conveyor_spawn, Mapping):
            raise ValueError("conveyor.spawn configuration must be a mapping")
        top_level_spawn = root.get("spawning", {})
        if not isinstance(top_level_spawn, Mapping):
            top_level_spawn = {}
        spawn = dict(conveyor_spawn)
        spawn.update(top_level_spawn)
        randomization = root.get("randomization", {})
        if not isinstance(randomization, Mapping):
            randomization = {}

        def random_section(name: str) -> Mapping[str, Any]:
            value = randomization.get(name, {})
            return value if isinstance(value, Mapping) else {}

        lateral_random = random_section("object_lateral_position")
        yaw_random = random_section("object_yaw")
        mass_random = random_section("object_mass_scale")
        friction_random = random_section("object_friction_scale")
        speed_random = random_section("belt_speed")
        jitter_random = random_section("belt_speed_jitter")
        enabled = bool(randomization.get("enabled", True))
        active_bounds = conveyor.get("active_bounds", {})
        surface_height = conveyor.get("surface_height")
        if surface_height is None and isinstance(active_bounds, Mapping):
            z_bounds = active_bounds.get("z")
            surface_height = z_bounds[0] if z_bounds else None
        kwargs: dict[str, Any] = {
            "spawn_x": spawn.get("x", root.get("spawn_x", default.spawn_x)),
            "lateral_range": spawn.get(
                "y_range",
                lateral_random.get(
                    "range", root.get("lateral_range", default.lateral_range)
                ),
            ),
            "surface_height": surface_height
            if surface_height is not None
            else root.get("surface_height", default.surface_height),
            "z_clearance": spawn.get(
                "z_clearance", root.get("z_clearance", default.z_clearance)
            ),
            "spacing_range": spawn.get(
                "spacing_range",
                (
                    spawn.get("min_spacing", default.spacing_range[0]),
                    spawn.get("max_spacing", default.spacing_range[1]),
                ),
            ),
            "yaw_range": yaw_random.get(
                "range", root.get("yaw_range", default.yaw_range)
            ),
            "belt_speed_range": speed_random.get(
                "range",
                root.get(
                    "belt_speed_range",
                    (
                        conveyor.get("default_speed", default.belt_speed_range[0]),
                        conveyor.get("default_speed", default.belt_speed_range[1]),
                    ),
                ),
            ),
            "velocity_jitter_std": jitter_random.get(
                "std",
                conveyor.get(
                    "speed_jitter_std",
                    root.get(
                        "velocity_jitter_std", default.velocity_jitter_std
                    ),
                ),
            ),
            "mass_scale_range": mass_random.get(
                "range", root.get("mass_scale_range", default.mass_scale_range)
            ),
            "friction_scale_range": friction_random.get(
                "range",
                root.get("friction_scale_range", default.friction_scale_range),
            ),
            "randomize_lateral_position": enabled
            and bool(lateral_random.get("enabled", True)),
            "randomize_yaw": enabled and bool(yaw_random.get("enabled", True)),
            "randomize_mass": enabled and bool(mass_random.get("enabled", True)),
            "randomize_friction": enabled
            and bool(friction_random.get("enabled", True)),
            "randomize_category": enabled
            and bool(random_section("object_category").get("enabled", True)),
            "randomize_belt_speed": enabled
            and bool(speed_random.get("enabled", True)),
            "randomize_color": enabled
            and bool(random_section("object_color").get("enabled", True)),
        }
        if not enabled or not bool(jitter_random.get("enabled", True)):
            kwargs["velocity_jitter_std"] = 0.0
        optional = (
            "lateral_velocity_jitter_std",
            "color_jitter",
            "randomize_category",
            "randomize_belt_speed",
            "randomize_color",
            "minimum_spawn_interval",
            "maximum_active_objects",
        )
        for name in optional:
            if name in root:
                kwargs[name] = root[name]
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class SpawnRecord:
    """Randomized values used for one reproducible spawn."""

    object_id: str
    category_id: int
    timestamp: float
    position: npt.NDArray[np.float64]
    quaternion_wxyz: npt.NDArray[np.float64]
    linear_velocity: npt.NDArray[np.float64]
    spacing: float
    belt_speed: float
    mass: float
    friction: tuple[float, float, float]
    rgba: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category_id": self.category_id,
            "timestamp": self.timestamp,
            "position": self.position.copy(),
            "quaternion_wxyz": self.quaternion_wxyz.copy(),
            "linear_velocity": self.linear_velocity.copy(),
            "spacing": self.spacing,
            "belt_speed": self.belt_speed,
            "mass": self.mass,
            "friction": self.friction,
            "rgba": self.rgba,
        }


class ObjectSpawner:
    """Schedule and randomize upstream workpieces with a reproducible RNG."""

    def __init__(
        self,
        manager: ObjectManager,
        config: ObjectSpawnerConfig | Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(manager, ObjectManager):
            raise TypeError("manager must be an ObjectManager")
        self.manager = manager
        if config is None:
            self.config = ObjectSpawnerConfig()
        elif isinstance(config, ObjectSpawnerConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = ObjectSpawnerConfig.from_mapping(config)
        else:
            raise TypeError("config must be ObjectSpawnerConfig or a mapping")
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.next_spawn_time = 0.0
        self.last_spawn_time: float | None = None
        self.last_spawn_record: SpawnRecord | None = None
        self.spawn_history: list[SpawnRecord] = []

    @classmethod
    def from_specs(
        cls,
        specs: Sequence[ObjectSpec],
        config: ObjectSpawnerConfig | Mapping[str, Any] | None = None,
        *,
        model: Any | None = None,
        data: Any | None = None,
        seed: int | None = None,
        manager_kwargs: Mapping[str, Any] | None = None,
    ) -> "ObjectSpawner":
        """Convenience constructor for standalone generation tests."""

        manager = ObjectManager(
            specs,
            model=model,
            data=data,
            **dict(manager_kwargs or {}),
        )
        return cls(manager, config, seed=seed)

    def reset(
        self,
        seed: int | None = None,
        *,
        current_time: float = 0.0,
        clear_history: bool = True,
    ) -> None:
        if seed is not None:
            self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)
        now = _finite(current_time, "current_time")
        self.next_spawn_time = now
        self.last_spawn_time = None
        self.last_spawn_record = None
        if clear_history:
            self.spawn_history.clear()

    def sample_belt_speed(self) -> float:
        low, high = self.config.belt_speed_range
        return (
            float(self.rng.uniform(low, high))
            if self.config.randomize_belt_speed
            else 0.5 * (low + high)
        )

    def can_spawn(
        self,
        current_time: float,
        *,
        category_ids: Sequence[int] | None = None,
        enforce_spacing: bool = True,
    ) -> bool:
        now = _finite(current_time, "current_time")
        if now + 1e-12 < self.next_spawn_time:
            return False
        if len(self.manager.active_objects()) >= self.config.maximum_active_objects:
            return False
        if not self.manager.available_specs(category_ids):
            return False
        if enforce_spacing and not self._spawn_region_clear():
            return False
        return True

    should_spawn = can_spawn

    def update(
        self,
        current_time: float,
        belt_speed: float,
        *,
        category_ids: Sequence[int] | None = None,
    ) -> ObjectInstance | None:
        """Spawn at most one due object; return ``None`` while waiting."""

        if not self.can_spawn(current_time, category_ids=category_ids):
            return None
        return self.spawn(
            current_time=current_time,
            belt_speed=belt_speed,
            category_ids=category_ids,
        )

    step = update

    def spawn(
        self,
        *,
        current_time: float,
        belt_speed: float,
        category_id: int | None = None,
        category_ids: Sequence[int] | None = None,
        force: bool = False,
        logical_object_id: str | None = None,
    ) -> ObjectInstance:
        """Generate one object and activate its physical body."""

        now = _finite(current_time, "current_time")
        speed = _finite(belt_speed, "belt_speed")
        if speed < 0.0:
            raise ValueError("belt_speed must be non-negative")
        if category_id is not None and category_ids is not None:
            raise ValueError("pass category_id or category_ids, not both")
        requested_categories = (
            [int(category_id)]
            if category_id is not None
            else None
            if category_ids is None
            else [int(item) for item in category_ids]
        )
        if not force and not self.can_spawn(
            now, category_ids=requested_categories, enforce_spacing=True
        ):
            raise NoAvailableObjectSlot(
                "spawn is not due, active limit reached, body unavailable, "
                "or upstream spacing is insufficient"
            )
        available = list(self.manager.available_specs(requested_categories))
        if not available:
            raise NoAvailableObjectSlot(
                f"no body slot available for categories {requested_categories}"
            )
        spec = self._select_spec(available)
        config = self.config
        belt_axis, lateral_axis = self._horizontal_axes()
        lateral = (
            float(self.rng.uniform(*config.lateral_range))
            if config.randomize_lateral_position
            else float(np.mean(config.lateral_range))
        )
        yaw = (
            float(self.rng.uniform(*config.yaw_range))
            if config.randomize_yaw
            else 0.0
        )
        position = belt_axis * config.spawn_x + lateral_axis * lateral
        position[2] = (
            config.surface_height
            + 0.5 * spec.dimensions[2]
            + config.z_clearance
        )
        quaternion = np.asarray(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
            dtype=np.float64,
        )
        longitudinal_speed = max(
            0.0,
            speed + float(self.rng.normal(0.0, config.velocity_jitter_std)),
        )
        lateral_speed = float(
            self.rng.normal(0.0, config.lateral_velocity_jitter_std)
        )
        linear_velocity = (
            self.manager.belt_direction * longitudinal_speed
            + lateral_axis * lateral_speed
        )
        mass_scale = (
            float(self.rng.uniform(*config.mass_scale_range))
            if config.randomize_mass
            else 1.0
        )
        friction_scale = (
            float(self.rng.uniform(*config.friction_scale_range))
            if config.randomize_friction
            else 1.0
        )
        mass = spec.mass * mass_scale
        friction = tuple(value * friction_scale for value in spec.friction)
        rgba_array = np.asarray(spec.rgba, dtype=np.float64)
        if config.randomize_color and config.color_jitter > 0.0:
            rgba_array[:3] += self.rng.normal(0.0, config.color_jitter, size=3)
            rgba_array = np.clip(rgba_array, 0.0, 1.0)
            rgba_array[3] = spec.rgba[3]
        rgba = tuple(float(item) for item in rgba_array)
        spacing = float(self.rng.uniform(*config.spacing_range))
        metadata = {
            "spawn_yaw": yaw,
            "spawn_spacing": spacing,
            "belt_speed_at_spawn": speed,
            "randomized_mass": mass,
            "randomized_friction": friction,
            "randomized_rgba": rgba,
        }
        instance = self.manager.spawn(
            spec.category_id,
            timestamp=now,
            position=position,
            quaternion=quaternion,
            linear_velocity=linear_velocity,
            logical_object_id=logical_object_id,
            metadata=metadata,
            physical_overrides={
                "mass": mass,
                "friction": friction,
                "rgba": rgba,
            },
        )
        record = SpawnRecord(
            object_id=instance.object_id,
            category_id=spec.category_id,
            timestamp=now,
            position=position.copy(),
            quaternion_wxyz=quaternion.copy(),
            linear_velocity=linear_velocity.copy(),
            spacing=spacing,
            belt_speed=speed,
            mass=mass,
            friction=friction,
            rgba=rgba,
        )
        self.last_spawn_time = now
        self.last_spawn_record = record
        self.spawn_history.append(record)
        # Spacing is distance-based; at a stopped belt retain a safe finite wait
        # and let the explicit spatial check prevent overlap.
        interval = (
            config.minimum_spawn_interval
            if abs(speed) < 1e-3
            else max(config.minimum_spawn_interval, spacing / abs(speed))
        )
        self.next_spawn_time = now + interval
        return instance

    spawn_now = spawn
    spawn_object = spawn

    def spawn_with_record(self, **kwargs: Any) -> tuple[ObjectInstance, SpawnRecord]:
        instance = self.spawn(**kwargs)
        assert self.last_spawn_record is not None
        return instance, self.last_spawn_record

    def _select_spec(self, available: list[ObjectSpec]) -> ObjectSpec:
        if len(available) == 1 or not self.config.randomize_category:
            return min(available, key=lambda spec: spec.category_id)
        return available[int(self.rng.integers(0, len(available)))]

    def _spawn_region_clear(self) -> bool:
        belt_axis, lateral_axis = self._horizontal_axes()
        spawn_point = (
            belt_axis * self.config.spawn_x
            + lateral_axis * float(np.mean(self.config.lateral_range))
        )
        spawn_point[2] = self.config.surface_height
        spawn_coordinate = float(
            np.dot(spawn_point, self.manager.belt_direction)
        )
        minimum_spacing = self.config.spacing_range[0]
        for instance in self.manager.active_objects():
            coordinate = float(
                np.dot(instance.position, self.manager.belt_direction)
            )
            if abs(coordinate - spawn_coordinate) < minimum_spacing:
                return False
        return True

    def _horizontal_axes(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        belt_axis = np.asarray(self.manager.belt_direction, dtype=np.float64).copy()
        belt_axis[2] = 0.0
        norm = float(np.linalg.norm(belt_axis))
        if norm <= 1e-9:
            raise ValueError("ObjectSpawner requires a horizontal belt direction")
        belt_axis /= norm
        lateral_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), belt_axis)
        return belt_axis, lateral_axis


def _range_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    if array[1] < array[0]:
        raise ValueError(f"{name} upper bound must be >= lower bound")
    return float(array[0]), float(array[1])


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
