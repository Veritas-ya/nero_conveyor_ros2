from __future__ import annotations

import pytest

from mujoco_conveyor_sorting import ConveyorSortingEnv


@pytest.fixture
def env() -> ConveyorSortingEnv:
    instance = ConveyorSortingEnv(
        render_mode=None,
        render_observations=False,
        seed=7,
        config_overrides={
            "scene": {
                "episode": {
                    "initial_objects": 1,
                    "object_count": 4,
                    "max_time": 30.0,
                }
            }
        },
    )
    try:
        yield instance
    finally:
        instance.close()
