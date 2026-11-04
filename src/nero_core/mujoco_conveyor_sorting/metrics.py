"""Metrics used by online runs and compensation comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class SortingMetrics:
    total_objects: int = 0
    valid_sorting_objects: int = 0
    grasp_attempts: int = 0
    successful_grasps: int = 0
    correctly_sorted: int = 0
    empty_grasps: int = 0
    dropped: int = 0
    missed: int = 0
    skipped: int = 0
    wrong_bin: int = 0
    collisions: int = 0
    failed: int = 0
    processing_times: list[float] = field(default_factory=list)
    prediction_errors: list[float] = field(default_factory=list)
    compensation_enabled_results: list[bool] = field(default_factory=list)
    compensation_disabled_results: list[bool] = field(default_factory=list)
    _outcomes_by_object: dict[int, tuple[bool, bool]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def reset(self) -> None:
        fresh = type(self)()
        self.__dict__.update(fresh.__dict__)

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def record_outcome(
        self,
        object_id: int,
        *,
        compensation_enabled: bool,
        success: bool,
    ) -> bool:
        """Record one mutually exclusive terminal result for one logical object.

        Returns ``False`` for a duplicate object ID, allowing delayed policy
        events to be ignored without corrupting A/B denominators.
        """

        identifier = int(object_id)
        if identifier in self._outcomes_by_object:
            return False
        enabled = bool(compensation_enabled)
        succeeded = bool(success)
        self._outcomes_by_object[identifier] = (enabled, succeeded)
        target = (
            self.compensation_enabled_results
            if enabled
            else self.compensation_disabled_results
        )
        target.append(succeeded)
        return True

    @property
    def recorded_outcome_count(self) -> int:
        return len(self._outcomes_by_object)

    def as_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("_outcomes_by_object", None)
        attempt_grasp_success_rate = self._rate(
            self.successful_grasps, self.grasp_attempts
        )
        end_to_end_success_rate = self._rate(
            self.correctly_sorted, self.valid_sorting_objects
        )
        raw.update(
            {
                "average_processing_time": (
                    float(np.mean(self.processing_times))
                    if self.processing_times
                    else 0.0
                ),
                "grasp_success_rate": attempt_grasp_success_rate,
                "attempt_grasp_success_rate": attempt_grasp_success_rate,
                "sorting_success_rate": end_to_end_success_rate,
                "end_to_end_sorting_success_rate": end_to_end_success_rate,
                "average_prediction_error": (
                    float(np.mean(self.prediction_errors))
                    if self.prediction_errors
                    else 0.0
                ),
                "max_prediction_error": (
                    float(np.max(self.prediction_errors))
                    if self.prediction_errors
                    else 0.0
                ),
                "motion_compensation_success_rate": self._rate(
                    sum(self.compensation_enabled_results),
                    len(self.compensation_enabled_results),
                ),
                "no_compensation_success_rate": self._rate(
                    sum(self.compensation_disabled_results),
                    len(self.compensation_disabled_results),
                ),
                "recorded_outcomes": self.recorded_outcome_count,
            }
        )
        return raw
