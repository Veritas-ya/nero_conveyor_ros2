import numpy as np
import pytest
from experiments.graspgenx_sim.linear_tcp_demo import LinearTCPReference


def test_trapezoid_is_continuous_and_has_real_constant_speed():
    ref=LinearTCPReference(np.array([.04,0,.70]),np.array([1.,0.,0.]),.12,.1,.3)
    assert ref.duration_s==pytest.approx(1.5)
    np.testing.assert_allclose(ref.sample(0)[0],[.04,0,.70])
    np.testing.assert_allclose(ref.sample(1.5)[0],[.16,0,.70])
    np.testing.assert_allclose(ref.sample(.7)[1],[.1,0,0])
    for boundary in (.3,1.2):
        left=ref.sample(boundary-1e-8);right=ref.sample(boundary+1e-8)
        np.testing.assert_allclose(left[0],right[0],atol=3e-9)
        np.testing.assert_allclose(left[1],right[1],atol=1e-8)
    np.testing.assert_allclose(ref.sample(1.5)[1],np.zeros(3))


def test_invalid_short_move_and_invalid_vector_rejected():
    with pytest.raises(ValueError):LinearTCPReference(np.zeros(3),np.ones(3),.01,.1,.3)
    with pytest.raises(ValueError):LinearTCPReference(np.zeros(3),np.zeros(3),.12,.1,.3)
