import numpy as np
import pytest
from experiments.two_second_grasp.controller import Intercept


def reference():
    return Intercept(1., 2., 2., np.array([.1,.2,.3]), np.array([.1,0,0]), np.array([0.,0.,.7]), np.array([1.,0,0,0]), np.array([1.,0,0,0]))


def test_capture_age_and_motion_time_are_both_in_prediction():
    ref=reference()
    np.testing.assert_allclose(ref.contact_xyz,[.4,.2,.3])
    np.testing.assert_allclose(ref.sample(0)[0],ref.start_tcp_world)
    np.testing.assert_allclose(ref.sample(1)[0],(ref.start_tcp_world+ref.contact_xyz)/2)
    np.testing.assert_allclose(ref.sample(2)[0],ref.contact_xyz)
    np.testing.assert_allclose(ref.sample(2.5)[0],[.45,.2,.3])


def test_rejects_bad_clock_and_copies_observation():
    source=np.array([.1,.2,.3]);ref=Intercept(1,2,2,source,np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0]);source[0]=8
    assert ref.capture_tcp_world[0]==.1
    with pytest.raises(ValueError):Intercept(2,1,2,[0,0,0],[0,0,0],[0,0,0],[1,0,0,0],[1,0,0,0])


def test_predict_entry_uses_exposure_timestamp_and_requires_upstream_lead():
    from experiments.two_second_grasp.controller import predict_entry_time
    assert predict_entry_time(1., -.6, .2, -.2, 1.5, .5) == pytest.approx(3.)
    with pytest.raises(ValueError, match='not upstream'):
        predict_entry_time(1., -.1, .2, -.2, 1.5, .5)
    with pytest.raises(ValueError, match='insufficient'):
        predict_entry_time(1., -.3, .2, -.2, 1.4, .5)


def test_simulation_and_physical_entry_use_same_freeze_window_and_exposure_clock():
    from nero_core.two_second.reference import predict_entry_time, EntryNotReady
    from experiments.two_second_grasp.controller import predict_entry_time as simulation_entry
    assert simulation_entry is predict_entry_time
    with pytest.raises(EntryNotReady) as pending:
        predict_entry_time(10.,-.66,.05,-.2,10.1,1.5,maximum_upstream_distance_m=.35)
    assert pending.value.reason == 'too_far_upstream'
    # The spatial boundary is inclusive, as in the original simulation check.
    assert predict_entry_time(10.,-.55,.1,-.2,10.2,1.5,maximum_upstream_distance_m=.35) == pytest.approx(13.5)
    with pytest.raises(EntryNotReady) as pending:
        predict_entry_time(10.,-.3,.1,-.2,10.5,1.5,maximum_upstream_distance_m=.35)
    assert pending.value.reason == 'insufficient_planning_time'


def test_entry_schedule_is_invariant_to_robot_base_axis_rotation():
    from scipy.spatial.transform import Rotation
    from nero_core.two_second.reference import predict_entry_time
    rotation=Rotation.from_euler('xyz',[.1,.05,1.2]).as_matrix()
    translation=np.array([.12,-.25,.31])
    position=np.array([-.45,0,.5]);velocity=np.array([.1,0,0]);direction=np.array([1.,0,0])
    moved_position=rotation@position+translation;moved_velocity=rotation@velocity;moved_direction=rotation@direction
    entry=-.2+translation@moved_direction
    sim=predict_entry_time(2.,position@direction,velocity@direction,-.2,2.3,1.5,maximum_upstream_distance_m=.35)
    real=predict_entry_time(2.,moved_position@moved_direction,moved_velocity@moved_direction,entry,2.3,1.5,maximum_upstream_distance_m=.35)
    assert real == pytest.approx(sim)
    assert sim == pytest.approx(4.5)
