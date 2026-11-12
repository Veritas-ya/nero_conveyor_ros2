"""Offline regression for the recorded 11.76 cm/s scheduling failure."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from nero_core.two_second.config import CellConfig
from nero_core.two_second.synchronized_reference import schedule_forward_join


CONFIG = Path(__file__).parents[1] / 'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


def test_recorded_target_fits_historical_schedule_without_claiming_a_path():
    # Pin the original regression's timing instead of mixing its saved poses
    # with later production join/follow tuning.
    # Recorded frozen grasp, velocity and TCP; all times share host monotonic clock.
    velocity = np.array([-.0004205833258817123, -.11750465412904605, .004715863407611723])
    direction = velocity / np.linalg.norm(velocity)
    schedule = schedule_forward_join(
        capture_s=12946.360887050629,
        grasp_position=[-.3635653094485638, .5142805640465032, -.11430224328248004],
        velocity=velocity, standby_position=[-.4029539131830803, -.03316729605119425, .033369018903275094],
        direction=direction, now_s=12946.813802781,
        planning_reserve_s=0., advance_s=.05, join_s=1.2,
        synchronized_s=0., follow_s=.1,
        follow_bounds=[-.2, .38], rotate_first_s=0.)
    assert schedule['contact_s'] == pytest.approx(schedule['sync_start_s'])
    assert schedule['follow_end_coordinate_m'] == pytest.approx(.1182, abs=.001)
    assert schedule['follow_end_coordinate_m'] <= .38


@pytest.mark.parametrize('key,value', [('synchronized_duration_s', -.01),
                                      ('synchronized_duration_s', float('nan')),
                                      ('synchronized_duration_s', float('inf'))])
def test_invalid_synchronized_duration_rejected(key, value, tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    for k in ('camera_calibration_file', 'tcp_calibration_file'):
        raw[k] = str((CONFIG.parent / raw[k]).resolve())
    raw[key] = value
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=key):
        CellConfig.load(path)


@pytest.mark.parametrize('change', [
    {'closure_policy': 'feedback'},
    {'intercept_strategy': 'classic'},
    {'brake_duration_s': .05},
])
def test_short_follow_requires_scheduled_confirmation_during_brake(change, tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    for key in ('camera_calibration_file', 'tcp_calibration_file'):
        raw[key] = str((CONFIG.parent / raw[key]).resolve())
    raw.update(change)
    path = tmp_path / 'short.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        CellConfig.load(path)


@pytest.mark.parametrize('brake_s,valid', [(.3, True), (.1, True), (.0999, False)])
def test_short_follow_accepts_confirmation_boundary_but_not_shorter(brake_s, valid, tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    for key in ('camera_calibration_file', 'tcp_calibration_file'):
        raw[key] = str((CONFIG.parent / raw[key]).resolve())
    raw['pick_settings']['full_plan_close_window_s'] = .1
    raw['brake_duration_s'] = brake_s
    path = tmp_path / 'boundary.yaml'
    path.write_text(yaml.safe_dump(raw))
    if valid:
        CellConfig.load(path)
    else:
        with pytest.raises(ValueError, match='不得短于0.2秒'):
            CellConfig.load(path)
