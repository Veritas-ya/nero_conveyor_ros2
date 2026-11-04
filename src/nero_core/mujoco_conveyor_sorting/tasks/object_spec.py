"""Configurable workpiece specifications.

The loader accepts both the repository's rich ``objects.yaml`` schema and a
minimal list of dictionaries, so replacing placeholders with real meshes does
not require environment-code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a project dependency
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """Object-local candidate pose, quaternion ordered ``w, x, y, z``."""

    position: npt.NDArray[np.float64]
    quaternion_wxyz: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.asarray([0.0, 1.0, 0.0, 0.0])
    )
    score: float = 1.0
    approach_direction: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _array(self.position, 3, "grasp position")
        )
        quaternion = _array(self.quaternion_wxyz, 4, "grasp quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("grasp quaternion cannot be zero")
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)
        score = float(self.score)
        if not np.isfinite(score) or score < 0.0:
            raise ValueError("grasp candidate score must be finite and non-negative")
        object.__setattr__(self, "score", score)
        if self.approach_direction is not None:
            approach = _array(
                self.approach_direction, 3, "grasp approach_direction"
            )
            approach_norm = float(np.linalg.norm(approach))
            if approach_norm <= 1e-12:
                raise ValueError("grasp approach_direction cannot be zero")
            object.__setattr__(
                self, "approach_direction", approach / approach_norm
            )


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """Physical, semantic and grasp metadata for one workpiece category."""

    category_id: int
    object_id: str
    display_name: str
    body_name: str
    freejoint_name: str
    dimensions: npt.NDArray[np.float64]
    mass: float
    friction: tuple[float, float, float]
    grasp_height: float
    recommended_grip_width: float
    bin_id: int
    geometry: Mapping[str, Any] = field(default_factory=lambda: {"kind": "box"})
    mesh_path: str | None = None
    symmetric: bool = False
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    grasp_candidates: tuple[GraspCandidate, ...] = ()
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    xml_geom_names: tuple[str, ...] = ()
    parking_weld_name: str | None = None
    grasp_assist_weld_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        category = int(self.category_id)
        if category < 0:
            raise ValueError("category_id must be non-negative")
        object.__setattr__(self, "category_id", category)
        for name in ("object_id", "display_name", "body_name", "freejoint_name"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        dimensions = _array(self.dimensions, 3, "dimensions")
        if np.any(dimensions <= 0.0):
            raise ValueError("dimensions must be positive")
        object.__setattr__(self, "dimensions", dimensions)
        for name in ("mass", "grasp_height", "recommended_grip_width"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        friction = _friction(self.friction)
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "bin_id", int(self.bin_id))
        rgba = tuple(float(value) for value in self.rgba)
        if len(rgba) != 4 or not all(np.isfinite(rgba)):
            raise ValueError("rgba must contain four finite values")
        object.__setattr__(
            self, "rgba", tuple(float(np.clip(value, 0.0, 1.0)) for value in rgba)
        )
        center = tuple(_array(self.center_of_mass, 3, "center_of_mass").tolist())
        object.__setattr__(self, "center_of_mass", center)
        object.__setattr__(self, "xml_geom_names", tuple(self.xml_geom_names))
        object.__setattr__(self, "grasp_candidates", tuple(self.grasp_candidates))

    @property
    def size(self) -> npt.NDArray[np.float64]:
        """Alias for integrations that call the bounding dimensions ``size``."""

        return self.dimensions

    @property
    def geometry_type(self) -> str:
        return str(
            self.geometry.get(
                "kind", self.geometry.get("type", "mesh" if self.mesh_path else "box")
            )
        )

    @property
    def symmetry_flag(self) -> bool:
        return self.symmetric

    @property
    def joint_name(self) -> str:
        return self.freejoint_name

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "object_id": self.object_id,
            "display_name": self.display_name,
            "body_name": self.body_name,
            "freejoint_name": self.freejoint_name,
            "dimensions": self.dimensions.tolist(),
            "mass": self.mass,
            "friction": list(self.friction),
            "grasp_height": self.grasp_height,
            "recommended_grip_width": self.recommended_grip_width,
            "bin_id": self.bin_id,
            "geometry": dict(self.geometry),
            "mesh_path": self.mesh_path,
            "symmetric": self.symmetric,
            "symmetry": dict(self.symmetry),
            "grasp_candidates": [
                {
                    "position": candidate.position.tolist(),
                    "quaternion_wxyz": candidate.quaternion_wxyz.tolist(),
                    "score": candidate.score,
                    **(
                        {
                            "approach_direction": candidate.approach_direction.tolist()
                        }
                        if candidate.approach_direction is not None
                        else {}
                    ),
                }
                for candidate in self.grasp_candidates
            ],
            "rgba": list(self.rgba),
            "center_of_mass": list(self.center_of_mass),
            "xml_geom_names": list(self.xml_geom_names),
            "parking_weld_name": self.parking_weld_name,
            "grasp_assist_weld_name": self.grasp_assist_weld_name,
            **dict(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        base_dir: str | Path | None = None,
        index: int = 0,
        strict: bool = False,
    ) -> "ObjectSpec":
        """Parse a rich or minimal object mapping with useful field aliases."""

        merged = dict(defaults or {})
        merged.update(dict(value))
        category = _first(merged, "category_id", "category", "class_id", "id")
        if category is None:
            if strict:
                raise ValueError("object spec is missing category_id")
            category = index
        category = int(category)
        object_id = str(
            _first(merged, "object_id", "name", "id")
            or f"placeholder_{category:02d}"
        )
        display_name = str(
            _first(merged, "display_name", "label", "class_name", "name")
            or object_id
        )
        body_name = str(
            _first(merged, "body_name", "mujoco_body", "body")
            or f"object_{category:02d}"
        )
        freejoint_name = str(
            _first(
                merged,
                "freejoint_name",
                "joint_name",
                "free_joint_name",
                "freejoint",
            )
            or f"{body_name}_freejoint"
        )
        dimensions = _first(merged, "dimensions", "size", "bbox_size")
        geometry_value = merged.get(
            "geometry", merged.get("geometry_type", merged.get("geom_type", {}))
        )
        if isinstance(geometry_value, str):
            geometry: Mapping[str, Any] = {"kind": geometry_value}
        elif isinstance(geometry_value, Mapping):
            geometry = dict(geometry_value)
        else:
            raise ValueError(f"geometry for category {category} must be a mapping/string")
        if dimensions is None:
            dimensions = _infer_dimensions(geometry)
        if dimensions is None:
            if strict:
                raise ValueError(f"object category {category} is missing dimensions")
            dimensions = (0.04, 0.04, 0.03)
        mass = _first(merged, "mass", "weight")
        friction = _first(merged, "friction", "friction_coeff", "mu")
        grasp_height = _first(merged, "grasp_height", "pickup_height")
        grip_width = _first(
            merged,
            "recommended_grip_width",
            "recommended_grasp_width",
            "grip_width",
            "grasp_width",
        )
        bin_id = _first(merged, "bin_id", "target_bin", "sorting_bin", "bin")
        if strict:
            missing = [
                name
                for name, item in (
                    ("mass", mass),
                    ("friction", friction),
                    ("grasp_height", grasp_height),
                    ("recommended_grip_width", grip_width),
                    ("bin_id", bin_id),
                )
                if item is None
            ]
            if missing:
                raise ValueError(
                    f"object category {category} is missing: {', '.join(missing)}"
                )
        height = float(np.asarray(dimensions, dtype=float)[2])
        width = float(np.asarray(dimensions, dtype=float)[1])
        mass = 0.10 if mass is None else mass
        friction = (0.8, 0.01, 0.001) if friction is None else friction
        grasp_height = max(0.005, height * 0.65) if grasp_height is None else grasp_height
        grip_width = max(0.01, width * 1.05) if grip_width is None else grip_width
        if bin_id is None:
            raise ValueError("bin_id must be explicit; category IDs do not encode bin rules")

        symmetry_value = merged.get(
            "symmetry", merged.get("symmetric", merged.get("is_symmetric", False))
        )
        if isinstance(symmetry_value, Mapping):
            symmetry = dict(symmetry_value)
            symmetric = str(symmetry.get("kind", "none")).lower() not in (
                "none",
                "asymmetric",
                "false",
            )
        else:
            if isinstance(symmetry_value, str):
                symmetric = symmetry_value.strip().lower() not in (
                    "",
                    "false",
                    "none",
                    "no",
                    "0",
                    "asymmetric",
                )
            else:
                symmetric = bool(symmetry_value)
            symmetry = {"kind": "symmetric" if symmetric else "none"}

        default_quaternion = merged.get(
            "grasp_candidate_quaternion_wxyz", (0.0, 1.0, 0.0, 0.0)
        )
        raw_candidates = merged.get(
            "grasp_candidates", merged.get("grasp_points", ())
        )
        candidates: list[GraspCandidate] = []
        candidates_source = () if raw_candidates is None else raw_candidates
        for raw in candidates_source:
            if isinstance(raw, Mapping):
                candidate_position = _first(
                    raw, "position", "point", "translation", "xyz"
                )
                if candidate_position is None:
                    raise ValueError(
                        f"grasp candidate for category {category} lacks position"
                    )
                raw_quaternion = _first(
                    raw, "quaternion_wxyz", "quaternion", "quat"
                )
                candidates.append(
                    GraspCandidate(
                        position=candidate_position,
                        quaternion_wxyz=default_quaternion
                        if raw_quaternion is None
                        else raw_quaternion,
                        score=float(raw.get("score", raw.get("confidence", 1.0))),
                        approach_direction=raw.get("approach_direction"),
                    )
                )
            else:
                candidates.append(
                    GraspCandidate(
                        position=raw,
                        quaternion_wxyz=default_quaternion,
                    )
                )
        if not candidates:
            candidates.append(
                GraspCandidate(
                    position=(0.0, 0.0, height + 0.01),
                    quaternion_wxyz=default_quaternion,
                )
            )

        mesh = _first(merged, "mesh_path", "mesh", "model_path")
        mesh_path: str | None = None
        if mesh not in (None, "", "null"):
            path = Path(str(mesh))
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir) / path
            mesh_path = str(path.resolve())

        known_keys = {
            "category_id",
            "category",
            "class_id",
            "id",
            "object_id",
            "name",
            "display_name",
            "label",
            "class_name",
            "body_name",
            "mujoco_body",
            "body",
            "freejoint_name",
            "joint_name",
            "free_joint_name",
            "freejoint",
            "dimensions",
            "size",
            "bbox_size",
            "geometry",
            "geometry_type",
            "geom_type",
            "mass",
            "weight",
            "friction",
            "friction_coeff",
            "mu",
            "grasp_height",
            "pickup_height",
            "recommended_grip_width",
            "recommended_grasp_width",
            "grip_width",
            "grasp_width",
            "bin_id",
            "target_bin",
            "sorting_bin",
            "bin",
            "symmetry",
            "symmetric",
            "is_symmetric",
            "grasp_candidates",
            "grasp_points",
            "rgba",
            "color",
            "mesh_path",
            "mesh",
            "model_path",
            "center_of_mass",
            "com",
            "xml_geom_names",
            "geom_names",
            "parking_weld_name",
            "grasp_assist_weld_name",
            "grasp_candidate_quaternion_wxyz",
        }
        metadata = {
            key: item for key, item in merged.items() if key not in known_keys
        }
        raw_rgba = merged.get("rgba", merged.get("color", (0.6, 0.6, 0.6, 1.0)))
        if len(raw_rgba) == 3:
            raw_rgba = (*raw_rgba, 1.0)
        return cls(
            category_id=category,
            object_id=object_id,
            display_name=display_name,
            body_name=body_name,
            freejoint_name=freejoint_name,
            dimensions=dimensions,
            mass=float(mass),
            friction=_friction(friction),
            grasp_height=float(grasp_height),
            recommended_grip_width=float(grip_width),
            bin_id=int(bin_id),
            geometry=geometry,
            mesh_path=mesh_path,
            symmetric=symmetric,
            symmetry=symmetry,
            grasp_candidates=tuple(candidates),
            rgba=tuple(raw_rgba),
            center_of_mass=tuple(
                merged.get("center_of_mass", merged.get("com", (0.0, 0.0, 0.0)))
            ),
            xml_geom_names=tuple(
                merged.get("xml_geom_names", merged.get("geom_names", ()))
            ),
            parking_weld_name=merged.get("parking_weld_name"),
            grasp_assist_weld_name=merged.get("grasp_assist_weld_name"),
            metadata=metadata,
        )


def load_object_specs(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    strict: bool = False,
    require_categories: int | Iterable[int] | None = None,
    base_dir: str | Path | None = None,
) -> list[ObjectSpec]:
    """Load and validate object specs from YAML/JSON or already parsed data.

    Accepted top-level layouts include a list, ``{"objects": [...]}``,
    ``{"categories": [...]}``, a category-keyed mapping, or the full project
    config ``{"objects": {"defaults": ..., "objects": [...]}}``.
    """

    base_dir = None if base_dir is None else Path(base_dir)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"object configuration does not exist: {path}")
        base_dir = path.resolve().parent
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(f"failed to read object configuration {path}") from exc
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                document: Any = json.loads(text)
            else:
                if yaml is None:
                    raise RuntimeError(
                        "PyYAML is required to load non-JSON object configuration"
                    )
                document = yaml.safe_load(text)
        except Exception as exc:
            raise ValueError(f"failed to parse object configuration {path}: {exc}") from exc
    else:
        document = source

    defaults: Mapping[str, Any] = {}
    raw_objects: Any
    if isinstance(document, Mapping):
        # A full project config and the compatibility wrapper in objects.yaml
        # can nest the actual sequence by two levels.
        for _ in range(3):
            nested = document.get("objects")
            if not isinstance(nested, Mapping) or not any(
                key in nested
                for key in ("objects", "object_specs", "categories", "specs")
            ):
                break
            document = nested
        if isinstance(document, Mapping):
            defaults_value = document.get("defaults", {})
            if defaults_value is not None and not isinstance(defaults_value, Mapping):
                raise ValueError("objects.defaults must be a mapping")
            defaults = dict(defaults_value or {})
            raw_objects = _first(
                document, "objects", "object_specs", "categories", "specs"
            )
            if isinstance(raw_objects, Mapping):
                raw_objects = _first(
                    raw_objects,
                    "objects",
                    "object_specs",
                    "categories",
                    "specs",
                )
            if raw_objects is None:
                ignored = {"defaults", "schema_version", "object_spec_contract"}
                if "category_id" in document or "class_id" in document:
                    raw_objects = [document]
                else:
                    keyed_objects: list[Mapping[str, Any]] = []
                    for key, value in document.items():
                        if key in ignored or not isinstance(value, Mapping):
                            continue
                        normalized = dict(value)
                        if (
                            "category_id" not in normalized
                            and "class_id" not in normalized
                        ):
                            try:
                                normalized["category_id"] = int(key)
                            except (TypeError, ValueError):
                                pass
                        keyed_objects.append(normalized)
                    raw_objects = keyed_objects
        else:  # pragma: no cover - guarded above
            raw_objects = document
    else:
        raw_objects = document
    if not isinstance(raw_objects, Sequence) or isinstance(
        raw_objects, (str, bytes)
    ):
        raise ValueError("object configuration must contain a sequence of specs")
    specs: list[ObjectSpec] = []
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"object spec at index {index} must be a mapping")
        specs.append(
            ObjectSpec.from_mapping(
                raw,
                defaults=defaults,
                base_dir=base_dir,
                index=index,
                strict=strict,
            )
        )
    if not specs:
        raise ValueError("object configuration contains no object specs")
    _validate_unique(specs)
    if require_categories is not None:
        expected = (
            set(range(int(require_categories)))
            if isinstance(require_categories, int)
            else {int(category) for category in require_categories}
        )
        actual = {spec.category_id for spec in specs}
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"object configuration lacks categories: {missing}")
    return sorted(specs, key=lambda spec: spec.category_id)


def default_object_specs() -> list[ObjectSpec]:
    """Load the configured task vocabulary, preserving original category IDs."""
    from .task_config import current_task_config
    from ..config import DEFAULT_CONFIG_DIR
    return load_object_specs(current_task_config()["objects"], base_dir=DEFAULT_CONFIG_DIR)


load_default_object_specs = default_object_specs


def _validate_unique(specs: Sequence[ObjectSpec]) -> None:
    for label, values in (
        ("category_id", [spec.category_id for spec in specs]),
        ("object_id", [spec.object_id for spec in specs]),
        ("body_name", [spec.body_name for spec in specs]),
        ("freejoint_name", [spec.freejoint_name for spec in specs]),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate {label} values: {duplicates}")


def _infer_dimensions(geometry: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if "half_size" in geometry:
        half = np.asarray(geometry["half_size"], dtype=np.float64)
        if half.shape == (3,):
            return tuple((2.0 * half).tolist())
    kind = str(geometry.get("kind", geometry.get("type", ""))).lower()
    radius = geometry.get("radius")
    half_height = geometry.get("half_height")
    height = geometry.get("height")
    if radius is not None and (half_height is not None or height is not None):
        full_height = 2.0 * float(half_height) if half_height is not None else float(height)
        return 2.0 * float(radius), 2.0 * float(radius), full_height
    if kind in ("sphere",) and radius is not None:
        diameter = 2.0 * float(radius)
        return diameter, diameter, diameter
    return None


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _friction(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.asarray([float(array), 0.005, 0.0001])
    if array.shape == (1,):
        array = np.asarray([array[0], 0.005, 0.0001])
    if array.shape != (3,):
        raise ValueError("friction must be a scalar or three MuJoCo coefficients")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("friction coefficients must be finite and non-negative")
    return tuple(float(item) for item in array)


def _array(value: Any, size: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()
