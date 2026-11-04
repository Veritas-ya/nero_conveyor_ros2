"""Compatibility import: simulation and measured runtime share one reference."""
from nero_core.two_second.reference import (Intercept, predict_entry_time, EntryNotReady,
    DEFAULT_FREEZE_UPSTREAM_DISTANCE_M, APPROACH_DURATION_S)

__all__ = ["Intercept", "predict_entry_time", "EntryNotReady",
           "DEFAULT_FREEZE_UPSTREAM_DISTANCE_M", "APPROACH_DURATION_S"]
