"""Category-to-bin mapping, target selection and VLA-friendly instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .object_manager import ObjectInstance, ObjectState
from .object_spec import ObjectSpec, default_object_specs


DEFAULT_CATEGORY_TO_BIN: Mapping[int, int] = {
    spec.category_id: spec.bin_id for spec in default_object_specs()
}

CATEGORY_NAMES_ZH: Mapping[str, str] = {
    "cylinder": "圆柱形工件",
    "cube": "方块工件",
    "flange": "法兰件",
    "ring": "环形件",
    "stepped_shaft": "阶梯轴",
    "holed_plate": "带孔零件",
    "long_bar": "长条零件",
    "symmetric_cross": "对称十字件",
    "asymmetric_l_part": "非对称L形件",
    "hex_part": "六角零件",
    "low_disc": "扁圆盘",
    "capsule": "胶囊形零件",
    "twin_boss": "双凸台零件",
    "triangular_part": "三角零件",
    "offset_step": "偏置阶梯件",
    "eccentric_disk": "偏心圆盘",
    "heavy_ellipsoid": "重型椭球件",
    "tall_block": "高方块",
    "weighted_combo": "组合重心零件",
    "eccentric_combo": "偏心组合件",
}


@dataclass(frozen=True, slots=True)
class BinSpec:
    bin_id: int
    label: str
    category_ids: tuple[int, ...] = ()
    body_name: str | None = None
    center: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bin_id", int(self.bin_id))
        if not str(self.label).strip():
            raise ValueError("bin label cannot be empty")
        object.__setattr__(self, "category_ids", tuple(map(int, self.category_ids)))
        if self.center is not None:
            center = np.asarray(self.center, dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError("bin center must contain three finite values")
            object.__setattr__(self, "center", tuple(float(item) for item in center))


@dataclass(frozen=True, slots=True)
class SortingAssignment:
    object_id: str | None
    category_id: int
    bin_id: int
    task_text: str
    language: str = "zh"
    skip_if_infeasible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category_id": self.category_id,
            "bin_id": self.bin_id,
            "task_text": self.task_text,
            "language": self.language,
            "skip_if_infeasible": self.skip_if_infeasible,
            **dict(self.metadata),
        }


class SortingTask:
    """Semantic sorting task kept separate from control and simulation."""

    def __init__(
        self,
        specs: Iterable[ObjectSpec] | None = None,
        *,
        bins: Iterable[BinSpec | Mapping[str, Any]] | None = None,
        category_to_bin: Mapping[int, int] | None = None,
        language: str = "zh",
        seed: int | None = None,
        belt_direction: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        parsed_specs = list(default_object_specs() if specs is None else specs)
        if not parsed_specs:
            raise ValueError("SortingTask requires at least one ObjectSpec")
        self.spec_by_category = {spec.category_id: spec for spec in parsed_specs}
        if len(self.spec_by_category) != len(parsed_specs):
            raise ValueError("ObjectSpec category_id values must be unique")
        if bins is None:
            # Derive default bin membership from the supplied specs, including legacy catalogs.
            bins = [dict(id=bid, label={0: "left", 1: "middle", 2: "right"}.get(bid, f"bin_{bid}"),
                         category_ids=[s.category_id for s in parsed_specs if s.bin_id == bid])
                    for bid in sorted({s.bin_id for s in parsed_specs})]
        self.bins = self._parse_bins(bins)
        mapping = (
            {spec.category_id: spec.bin_id for spec in parsed_specs}
            if category_to_bin is None
            else {int(key): int(value) for key, value in category_to_bin.items()}
        )
        missing = set(self.spec_by_category) - set(mapping)
        if missing:
            raise ValueError(f"category-to-bin mapping lacks categories: {sorted(missing)}")
        unknown_bins = set(mapping.values()) - set(self.bins)
        if unknown_bins:
            raise ValueError(f"category mapping references unknown bins: {sorted(unknown_bins)}")
        self.category_to_bin = mapping
        direction = np.asarray(belt_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("belt_direction must contain three finite values")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            raise ValueError("belt_direction cannot be zero")
        self.belt_direction = direction / direction_norm
        self.language = self._normalize_language(language)
        self.rng = np.random.default_rng(seed)
        self.current_assignment: SortingAssignment | None = None
        self.selected_target_id: str | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        specs: Iterable[ObjectSpec] | None = None,
        language: str = "zh",
        seed: int | None = None,
    ) -> "SortingTask":
        root: Mapping[str, Any] = config
        if isinstance(root.get("scene"), Mapping):
            root = root["scene"]  # type: ignore[assignment]
        bins = root.get("bins")
        mapping = root.get("category_to_bin")
        conveyor = root.get("conveyor", {})
        if not isinstance(conveyor, Mapping):
            conveyor = {}
        return cls(
            specs,
            bins=bins if isinstance(bins, Sequence) else None,
            category_to_bin=mapping if isinstance(mapping, Mapping) else None,
            language=language,
            seed=seed,
            belt_direction=conveyor.get("direction", (1.0, 0.0, 0.0)),
        )

    @property
    def task_text(self) -> str:
        if self.current_assignment is not None:
            return self.current_assignment.task_text
        if self.language == "zh":
            return "连续分拣传送带上的工件；跳过无法在时间窗内安全抓取的目标。"
        return (
            "Continuously sort conveyor parts; skip targets that cannot be "
            "grasped safely within their time window."
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_assignment = None
        self.selected_target_id = None

    def target_bin(self, category_or_object: int | ObjectSpec | ObjectInstance) -> int:
        category_id = _category_id(category_or_object)
        try:
            return self.category_to_bin[category_id]
        except KeyError as exc:
            raise KeyError(f"unmapped category_id: {category_id}") from exc

    target_bin_for_category = target_bin

    def is_correct_bin(
        self,
        category_or_object: int | ObjectSpec | ObjectInstance,
        bin_id: int,
    ) -> bool:
        return self.target_bin(category_or_object) == int(bin_id)

    validate_sort = is_correct_bin
    is_correct_sort = is_correct_bin

    def bin_label(self, bin_id: int, *, language: str | None = None) -> str:
        try:
            bin_spec = self.bins[int(bin_id)]
        except KeyError as exc:
            raise KeyError(f"unknown bin_id: {bin_id}") from exc
        selected_language = self._normalize_language(language or self.language)
        if selected_language == "en":
            return bin_spec.label
        label = bin_spec.label.lower()
        translations = {
            "left": "左侧",
            "middle": "中间",
            "center": "中间",
            "right": "右侧",
            "bin_a": "左侧",
            "bin_b": "中间",
            "bin_c": "右侧",
        }
        return translations.get(label, f"{bin_spec.bin_id}号")

    def generate_instruction(
        self,
        category_id: int,
        *,
        object_id: str | None = None,
        language: str | None = None,
        style: str = "auto",
        include_skip_rule: bool = False,
    ) -> str:
        """Create a grounded natural-language instruction for VLA policies."""

        category = int(category_id)
        try:
            spec = self.spec_by_category[category]
        except KeyError as exc:
            raise KeyError(f"unknown category_id: {category}") from exc
        selected_language = self._normalize_language(language or self.language)
        target_bin = self.target_bin(category)
        if style == "auto":
            style = "name" if spec.display_name in CATEGORY_NAMES_ZH else "category"
        if style not in ("name", "category", "object"):
            raise ValueError("style must be auto, name, category, or object")

        if selected_language == "zh":
            bin_text = self.bin_label(target_bin, language="zh")
            part_name = CATEGORY_NAMES_ZH.get(spec.display_name, spec.display_name)
            if style == "name":
                text = f"抓取{part_name}并放入{bin_text}料框"
            elif style == "object" and object_id:
                text = f"抓取目标{object_id}（类别{category}）并放入{bin_text}料框"
            else:
                text = f"将类别{category}的零件放入{bin_text}料框"
            if include_skip_rule:
                text += "；如果无法在时间窗内安全抓取，请跳过该工件"
            return text

        bin_text = self.bin_label(target_bin, language="en")
        if style == "name":
            text = f"Pick the {spec.display_name} and place it in the {bin_text} bin"
        elif style == "object" and object_id:
            text = (
                f"Pick target {object_id} (category {category}) and place it "
                f"in the {bin_text} bin"
            )
        else:
            text = f"Place the category {category} part in the {bin_text} bin"
        if include_skip_rule:
            text += "; skip it if no safe grasp window remains"
        return text

    instruction_for = generate_instruction
    build_task_text = generate_instruction

    def create_assignment(
        self,
        category_or_object: int | ObjectSpec | ObjectInstance,
        *,
        object_id: str | None = None,
        language: str | None = None,
        style: str = "auto",
        include_skip_rule: bool = True,
    ) -> SortingAssignment:
        category = _category_id(category_or_object)
        if isinstance(category_or_object, ObjectInstance):
            object_id = category_or_object.object_id
        selected_language = self._normalize_language(language or self.language)
        assignment = SortingAssignment(
            object_id=object_id,
            category_id=category,
            bin_id=self.target_bin(category),
            task_text=self.generate_instruction(
                category,
                object_id=object_id,
                language=selected_language,
                style=style,
                include_skip_rule=include_skip_rule,
            ),
            language=selected_language,
            skip_if_infeasible=include_skip_rule,
        )
        self.current_assignment = assignment
        self.selected_target_id = object_id
        return assignment

    set_target = create_assignment

    def clear_target(self) -> None:
        self.current_assignment = None
        self.selected_target_id = None

    def select_target(
        self,
        objects: Iterable[ObjectInstance | Mapping[str, Any]],
        *,
        grasp_windows: Mapping[str, Any] | None = None,
        current_time: float | None = None,
    ) -> ObjectInstance | Mapping[str, Any] | None:
        """Choose the feasible target with the earliest window deadline.

        If no window mapping is supplied, objects farthest along the configured
        belt direction are selected first.  Terminal/executing/grasped objects
        are ignored.
        """

        candidates: list[
            tuple[float, float, str, ObjectInstance | Mapping[str, Any]]
        ] = []
        allowed_states = {
            ObjectState.NEW.value,
            ObjectState.TRACKING.value,
            ObjectState.READY_TO_PLAN.value,
        }
        for obj in objects:
            object_id = str(_field(obj, "object_id", ""))
            if not object_id:
                continue
            state_value = _field(
                obj,
                "state",
                _field(obj, "status", ObjectState.TRACKING.value),
            )
            state = (
                state_value.value
                if isinstance(state_value, ObjectState)
                else str(state_value).upper()
            )
            if state not in allowed_states:
                continue
            position = np.asarray(
                _field(obj, "position", (float("-inf"), 0.0, 0.0)), dtype=float
            )
            progress = (
                float(np.dot(position, self.belt_direction))
                if position.shape == (3,)
                else float("-inf")
            )
            deadline = float("inf")
            if grasp_windows is not None:
                window = grasp_windows.get(object_id)
                if window is None:
                    continue
                feasible = bool(_field(window, "feasible", _field(window, "is_graspable", False)))
                if not feasible:
                    continue
                deadline = float(_field(window, "end_time", float("inf")))
                if current_time is not None and deadline < float(current_time):
                    continue
            # Earliest deadline, then greatest belt progress, then stable ID.
            candidates.append((deadline, -progress, object_id, obj))
        if not candidates:
            self.clear_target()
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        selected = candidates[0][3]
        selected_id = str(_field(selected, "object_id"))
        category = int(_field(selected, "category_id"))
        self.create_assignment(
            category,
            object_id=selected_id,
            style="object",
            include_skip_rule=True,
        )
        return selected

    choose_target = select_target

    @staticmethod
    def _normalize_language(value: str) -> str:
        normalized = str(value).strip().lower().replace("_", "-")
        if normalized in ("zh", "zh-cn", "cn", "chinese", "中文"):
            return "zh"
        if normalized in ("en", "en-us", "english"):
            return "en"
        raise ValueError("language must be Chinese ('zh') or English ('en')")

    @staticmethod
    def _parse_bins(
        bins: Iterable[BinSpec | Mapping[str, Any]] | None,
    ) -> dict[int, BinSpec]:
        result: dict[int, BinSpec] = {}
        for index, item in enumerate(bins):
            if isinstance(item, BinSpec):
                spec = item
            elif isinstance(item, Mapping):
                bin_id = int(item.get("id", item.get("bin_id", index)))
                center_value = item.get("center")
                spec = BinSpec(
                    bin_id=bin_id,
                    label=str(item.get("label", item.get("name", f"bin_{bin_id}"))),
                    category_ids=tuple(
                        item.get("category_ids", item.get("categories", ()))
                    ),
                    body_name=item.get("body_name"),
                    center=tuple(center_value) if center_value is not None else None,
                )
            else:
                raise TypeError("bins must contain BinSpec or mapping values")
            if spec.bin_id in result:
                raise ValueError(f"duplicate bin_id: {spec.bin_id}")
            result[spec.bin_id] = spec
        if not result:
            raise ValueError("at least one bin is required")
        return result


def _category_id(value: int | ObjectSpec | ObjectInstance) -> int:
    if isinstance(value, (ObjectSpec, ObjectInstance)):
        return value.category_id
    return int(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
