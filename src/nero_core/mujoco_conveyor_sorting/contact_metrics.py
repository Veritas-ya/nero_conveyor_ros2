"""Read-only contact evaluation shared by production collection and audits.

These helpers classify simulator contacts for scoring only. Their output must
not be inserted into policy observations or used to mutate object lifecycle.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def held_payload_rail_key(
    first_geom: int, second_geom: int,
    object_geometry: Mapping[int, Any], rail_geoms: set[int] | frozenset[int],
) -> tuple[int, int] | None:
    """Identify one unreleased GRASPED payload/rail pair for evaluation."""

    if first_geom in rail_geoms:
        runtime, rail = object_geometry.get(second_geom), first_geom
    elif second_geom in rail_geoms:
        runtime, rail = object_geometry.get(first_geom), second_geom
    else:
        return None
    if (
        runtime is None or runtime.logical_id is None
        or runtime.state != "GRASPED" or runtime.released_at is not None
    ):
        return None
    return int(runtime.logical_id), int(rail)


class ContactEpisodes:
    """Count positive-force impacts, merging gaps of at most 20 ms.

    The first loaded sample counts immediately: a short impact is still a
    collision. Debouncing avoids counting compliant-solver gaps as new impacts.
    """

    def __init__(self, gap_seconds: float = 0.02):
        if not np.isfinite(gap_seconds) or gap_seconds < 0.0:
            raise ValueError("gap_seconds must be finite and nonnegative")
        self.gap_seconds = float(gap_seconds)
        self.reset()

    def reset(self) -> None:
        self.active: dict[tuple[int, int], dict[str, Any]] = {}
        self.episodes: list[dict[str, Any]] = []

    def update(self, now: float, contacts: Mapping[tuple[int, int], float]) -> list[dict[str, Any]]:
        events = []
        for key, episode in list(self.active.items()):
            if now - episode["last_contact_at"] > self.gap_seconds + 1e-9:
                events.append({"phase": "ended", **episode})
                del self.active[key]
        for (object_id, rail_geom), force in contacts.items():
            if not np.isfinite(force) or force <= 0.0:
                continue
            key = (object_id, rail_geom)
            if key not in self.active:
                episode = {"episode_id": len(self.episodes), "logical_id": object_id,
                           "rail_geom": rail_geom, "started_at": now,
                           "last_contact_at": now, "peak_normal_force": float(force),
                           "contact_samples": 0}
                self.active[key] = episode
                self.episodes.append(episode)
                events.append({"phase": "started", **episode})
            episode = self.active[key]
            episode["last_contact_at"] = now
            episode["peak_normal_force"] = max(episode["peak_normal_force"], float(force))
            episode["contact_samples"] += 1
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = [{"phase": "ended", **episode} for episode in self.active.values()]
        self.active.clear()
        return events
