"""Offline object labels, separate from observations consumed by the policy.

Labels describe the terminal cycle containing a frame, including its initial
return/wait actions. ``target_active`` only marks a matching pre-action selected
visual track; neither flag is an image visibility or segmentation annotation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


SCHEMA = "mujoco_conveyor_sorting.pi05.annotations.v1"
LEGACY_PROMPT = "Sort the moving conveyor parts by shape into their designated left, middle, and right bins."
LEGACY_RULES_PROMPT = (
    "Sort all moving conveyor parts by shape: classes 0-4 into bin 0, "
    "classes 5-9 into bin 1, and classes 10-14 into bin 2."
)
TERMINAL_STATES = frozenset({"SORTED", "FAILED", "MISSED", "SKIPPED"})
INTEGER_FIELDS = ("category_id", "target_bin_id", "logical_object_id", "tless_object_id")
BOOLEAN_FIELDS = ("valid", "target_active")
FEATURES = {
    **{f"annotation.{key}": {"dtype": "int64", "shape": (1,), "names": [key]} for key in INTEGER_FIELDS},
    **{f"annotation.{key}": {"dtype": "bool", "shape": (1,), "names": [key]} for key in BOOLEAN_FIELDS},
}


def _id(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def category_catalog(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use the recorded effective model, never today's mutable config or id//5."""
    config = metadata.get("config", {})
    specs = config.get("objects", {}).get("object_specs", [])
    bins = config.get("scene", {}).get("bins", [])
    configured_bins = {}
    for item in bins:
        bid = _id(item["id"], "bin id")
        for cid in item.get("category_ids", []):
            cid = _id(cid, "category id")
            if cid in configured_bins:
                raise ValueError("A category is assigned to multiple scene bins")
            configured_bins[cid] = bid
    catalog = {}
    for spec in specs:
        cid, bid = _id(spec["category_id"], "category id"), _id(spec["bin_id"], "target bin id")
        if cid in catalog:
            raise ValueError("Duplicate category in recorded object catalog")
        if bins and configured_bins.get(cid) != bid:
            raise ValueError(f"Recorded scene and object catalog disagree for class {cid}")
        tless = spec.get("tless_object_id")
        catalog[cid] = dict(category_id=cid, target_bin_id=bid,
                            tless_object_id=-1 if tless is None else _id(tless, "T-LESS object id"),
                            name=str(spec.get("display_name", spec.get("object_id", f"class {cid}"))))
    return [catalog[cid] for cid in sorted(catalog)]


def rules_prompt(metadata: Mapping[str, Any]) -> str:
    catalog = category_catalog(metadata)
    if not catalog:
        return DEFAULT_PROMPT
    groups: dict[int, list[int]] = {}
    for item in catalog:
        groups.setdefault(item["target_bin_id"], []).append(item["category_id"])
    clauses = []
    for bid, ids in sorted(groups.items()):
        consecutive = ids == list(range(ids[0], ids[-1] + 1))
        classes = f"{ids[0]}-{ids[-1]}" if consecutive and len(ids) > 1 else ", ".join(map(str, ids))
        clauses.append(f"classes {classes} into bin {bid}")
    listing = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + ", and " + clauses[-1]
    return "Sort all moving conveyor parts by shape: " + listing + "."


# New recordings follow the configured task; old recordings use their own catalog.
from ..tasks.task_config import current_task_config
DEFAULT_PROMPT = rules_prompt({"config": current_task_config()})


def _unknown() -> dict[str, Any]:
    return {**{key: -1 for key in INTEGER_FIELDS}, "valid": False, "target_active": False}


def build_annotations(document: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = document.get("frames", [])
    catalog = category_catalog(document.get("metadata", {}))
    by_category = {item["category_id"]: item for item in catalog}
    labels = [_unknown() for _ in frames]
    cycles = []
    start = 0
    for end, frame in enumerate(frames):
        events = frame.get("events", [])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError("Malformed lifecycle events in annotation source")
        terminals = [event for event in events if event.get("state") in TERMINAL_STATES]
        if not terminals:
            continue
        cycle = dict(start_frame=start, end_frame_inclusive=end, terminal_events=deepcopy(terminals),
                     category_prompt=None, **_unknown())
        if len(terminals) == 1:
            event = terminals[0]
            cid, oid = event.get("category_id"), event.get("object_id")
            if cid is not None:
                cid = _id(cid, "terminal category id")
                if catalog and cid not in by_category:
                    raise ValueError(f"Terminal class {cid} is absent from the recorded catalog")
                cycle["category_id"] = cid
                if cid in by_category:
                    cycle.update({key: by_category[cid][key] for key in ("target_bin_id", "tless_object_id")})
            if oid is not None:
                cycle["logical_object_id"] = _id(oid, "logical object id")
            cycle["valid"] = all(cycle[key] >= 0 for key in ("category_id", "target_bin_id", "logical_object_id"))
            if cycle["valid"]:
                cycle["category_prompt"] = (
                    f"Pick up the class {cid} part from the conveyor and place it into bin {cycle['target_bin_id']}."
                )
            track = event.get("track_id")
            for i in range(start, end + 1):
                labels[i] = {key: cycle[key] for key in (*INTEGER_FIELDS, "valid")}
                labels[i]["target_active"] = bool(cycle["valid"] and track is not None
                    and frames[i].get("selected_target_id") == track)
        cycles.append(cycle)
        start = end + 1
    header = dict(
        schema=SCHEMA, scope="terminal_cycle", source="recorded_terminal_events_and_effective_object_catalog",
        policy_input=False, unknown_id=-1,
        target_active_semantics="pre_action_selected_track_matches_cycle_terminal_track; not_visibility",
        category_prompt_role="offline_cycle_caption; not_default_model_prompt",
        category_catalog=catalog, cycles=cycles,
        unlabeled_tail_start_frame=start if start < len(frames) else None,
    )
    return header, labels


def annotate_episode(document: dict[str, Any], *, update_prompt: bool = False) -> dict[str, Any]:
    """Annotate an owned document in place; numeric recordings are untouched."""
    metadata = document["metadata"]
    if update_prompt and metadata.get("prompt") in (LEGACY_PROMPT, LEGACY_RULES_PROMPT, DEFAULT_PROMPT):
        metadata["prompt"] = rules_prompt(metadata)
        for frame in document["frames"]:
            frame["task_text"] = metadata["prompt"]
    header, labels = build_annotations(document)
    document["annotations"] = header
    for frame, label in zip(document["frames"], labels, strict=True):
        frame["annotation"] = label
    return document


def validate_annotations(document: Mapping[str, Any]) -> None:
    """Old recordings remain readable; declared labels must match their evidence."""
    if "annotations" not in document:
        if any("annotation" in frame for frame in document.get("frames", [])):
            raise ValueError("Frame annotations lack an annotation schema")
        return
    header, labels = build_annotations(document)
    if document["annotations"] != header:
        raise ValueError("Cycle annotation or category catalog contradicts recorded evidence")
    if [frame.get("annotation") for frame in document["frames"]] != labels:
        raise ValueError("Frame annotations contradict cycle identity or selected-target alignment")


def annotation_arrays(document: Mapping[str, Any]) -> dict[str, np.ndarray]:
    validate_annotations(document)
    _, labels = build_annotations(document)
    return {
        f"annotation.{key}": np.asarray([label[key] for label in labels], dtype=np.int64 if key in INTEGER_FIELDS else bool).reshape(-1, 1)
        for key in (*INTEGER_FIELDS, *BOOLEAN_FIELDS)
    }
