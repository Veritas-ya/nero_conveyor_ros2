"""No device: reproduce a prefetched standby offset and stale CAN snapshots."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.runtime_sweep import SweepSettings
from nero_core.two_second.streaming import fresh_feedback, intercept_prime_feedback


def refresh(initial, measured, *, width=.1, force=0.):
    now = [10.]
    events = []
    old = dict(joints=list(initial), width_m=.1, force_n=0.,
               pose_stamp=9.86, joint_stamp=9.86, gripper_stamp=9.86)
    def read():
        return dict(joints=list(measured), width_m=width, force_n=force,
                    pose_stamp=now[0]-.01, joint_stamp=now[0]-.01,
                    gripper_stamp=now[0]-.01)
    e = NS(clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt),
           device=NS(snapshot_nowait=read), feedback_stamps={},
           config=NS(settings=NS(open_width_m=.1, min_force_n=1.)),
           planner=NS(model=NS(check_feedback=lambda sample: None)),
           _active=lambda *args: None, _emit=lambda name, data: events.append((name, data)))
    path = NS(identity=(0, 1, 28), q0=np.zeros(7))
    result = intercept_prime_feedback(e, old, path, 13., 1, 0, SweepSettings())
    return result, events


def test_recorded_cycle_9_offset_survives_feedback_refresh():
    # 20261119_220208 cycle_009: measured minus configured standby (degrees),
    # at initial stream snapshot and the sample that triggered cancellation.
    initial = np.deg2rad([.242877957, -.015113040, -.022425094, -.177326880,
                         .119987417, .381963823, .338542713])
    measured = initial.copy()
    measured[0] -= np.deg2rad(.005)
    measured[5] = np.deg2rad(.371963823)
    assert abs(measured[5]) > .005  # old code rejected the already admitted path
    (sample, refreshed), events = refresh(initial, measured)
    assert refreshed
    np.testing.assert_allclose(sample['joints'], measured)
    detail = events[-1][1]
    assert detail['path_start_tolerance_rad'] == pytest.approx(np.deg2rad(1.))
    assert max(abs(x) for x in detail['refresh_joint_delta_rad']) < .005


@pytest.mark.parametrize('initial_deg,measured_deg', [(.9, .9), (1., 1.)])
def test_accepted_stationary_offset_is_not_confused_with_new_motion(initial_deg, measured_deg):
    (_, refreshed), _ = refresh(np.full(7, np.deg2rad(initial_deg)),
                               np.full(7, np.deg2rad(measured_deg)))
    assert refreshed


@pytest.mark.parametrize('initial_deg,measured_deg', [(1.01, 1.01), (0., .3), (.9, 1.01)])
def test_rejects_excess_path_offset_or_real_motion_during_wait(initial_deg, measured_deg):
    with pytest.raises(PickRejected) as exc:
        refresh(np.full(7, np.deg2rad(initial_deg)), np.full(7, np.deg2rad(measured_deg)))
    assert exc.value.details['reason'] == 'intercept_prime_start_changed'


@pytest.mark.parametrize('width,force', [(.09, 0.), (.1, 1.)])
def test_gripper_state_must_still_be_open_and_empty(width, force):
    with pytest.raises(PickRejected):
        refresh(np.zeros(7), np.zeros(7), width=width, force=force)


def test_recorded_expired_feedback_still_rejected_and_explained():
    state = dict(pose_stamp_monotonic=2011.0241680145264,
                 joint_stamp_monotonic=2011.0405614376068,
                 gripper_stamp_monotonic=2011.0392003059387, feedback_read_monotonic=2011.219638,
                 feedback_time={'end_pose_xy': dict(mapping_quality='mapped',
                                                    mapped_monotonic=2011.0241680145264,
                                                    host_monotonic_received=2011.054365972)})
    with pytest.raises(PickRejected) as exc:
        fresh_feedback(state, 2011.244472167)
    detail = exc.value.details
    assert detail['feedback_age_s'] == pytest.approx(.2203041524735454)
    assert detail['maximum_age_s'] == .2
    assert detail['packet_ages_s']['joint'] > .2
    assert detail['receive_callback_lags_s']['end_pose_xy'] == pytest.approx(.0301979574736)
    assert detail['snapshot_age_s'] == pytest.approx(.024834167)
