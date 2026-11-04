"""Task vocabulary and data namespace, independent of robot/model contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

from ..config import DEFAULT_CONFIG_DIR, load_yaml


def effective_objects(scene: Mapping, document: Mapping) -> dict:
    """Filter the asset catalog without renumbering original category IDs."""
    result = deepcopy(dict(document))
    specs = result["object_specs"]
    by_id = {s["category_id"]: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError("Duplicate category in asset catalog")
    ids = scene.get("task", {}).get("active_category_ids", list(by_id))
    if (not ids or any(type(cid) is not int or cid < 0 for cid in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys()):
        raise ValueError("active_category_ids must be unique existing nonnegative category IDs")
    mapping = {}
    for item in scene["bins"]:
        for cid in item.get("category_ids", []):
            if cid in mapping:
                raise ValueError("Category assigned to multiple bins")
            mapping[cid] = item["id"]
    if set(mapping) != set(ids):
        raise ValueError("Scene bins must cover exactly the active categories")
    result["object_specs"] = [by_id[cid] for cid in ids]
    for spec in result["object_specs"]:
        if spec["bin_id"] != mapping[spec["category_id"]]:
            raise ValueError("Scene bin and object catalog disagree")
    if "source" in result:
        result["source"]["selected_object_ids"] = [s["tless_object_id"] for s in result["object_specs"]]
    return result


def current_task_config() -> dict:
    scene = load_yaml(DEFAULT_CONFIG_DIR / "scene.yaml")
    objects = load_yaml(DEFAULT_CONFIG_DIR / scene["object_catalog_path"])
    return {"scene": scene, "objects": effective_objects(scene, objects)}


def task_defaults() -> dict:
    scene = current_task_config()["scene"]
    return {**scene["task"], "objects": int(scene["episode"]["object_count"])}


def task_signature(config: Mapping | None = None) -> dict:
    # Local import keeps recorded-data annotation independent of current config.
    from ..recording.pi05_annotations import category_catalog, rules_prompt
    config = current_task_config() if config is None else config
    metadata = {"config": config}
    return {"task_id": config["scene"].get("task", {}).get("id", "legacy"),
            "category_catalog": category_catalog(metadata), "prompt": rules_prompt(metadata)}


def check_task_match(actual: Mapping, expected: Mapping | None = None) -> None:
    expected = task_signature() if expected is None else expected
    if dict(actual) != dict(expected):
        raise ValueError("Task vocabulary/bin rules/prompt differ; use a matching dataset and checkpoint")
