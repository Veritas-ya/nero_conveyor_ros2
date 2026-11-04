"""Exact fast-path eligibility for Orbbec's eight-term camera calibration."""
import cv2
import numpy as np
import pytest
from mujoco_conveyor_sorting.runtime.visible_target import _project_points, _conservative_rgb_roi, _RoiUnavailable


def test_zero_rational_terms_match_opencv_pixel_labels_and_preserve_roi():
    rng = np.random.default_rng(20261030)
    color = np.column_stack((rng.uniform(-.5,.5,50000), rng.uniform(-.3,.3,50000), rng.uniform(.6,1.5,50000)))
    matrix = np.array([[603.545776,0,640.5368],[0,603.79455,360.7278],[0,0,1.]])
    coeff = np.array([-.02966746,.03541105,-.000011176,.000005424,-.01283123,0,0,0])
    reference = cv2.projectPoints(color,np.zeros(3),np.zeros(3),matrix,coeff)[0].reshape(-1,2)
    report={}; actual = _project_points(color,matrix,coeff,'auto',diagnostics=report)
    assert report['projection_backend']=='no_jacobian'
    np.testing.assert_array_equal(np.rint(actual),np.rint(reference))
    labels=np.zeros((720,1280),np.uint8);labels[200:500,400:700]=1
    selected=_conservative_rgb_roi(color,matrix,coeff,labels,{})
    pixels=np.rint(reference).astype(int)
    relevant=(pixels[:,0]>=400)&(pixels[:,0]<700)&(pixels[:,1]>=200)&(pixels[:,1]<500)
    assert selected[relevant].all()


def test_nonzero_rational_term_uses_full_opencv_fallback():
    matrix=np.array([[600.,0,640.],[0,600.,360.],[0,0,1.]])
    coeff=np.array([-.03,.035,0,0,-.013,1e-12,0,0])
    points=np.array([[.1,.2,1.],[-.2,.3,1.]])
    report={};actual=_project_points(points,matrix,coeff,'auto',diagnostics=report)
    assert report['projection_backend']=='opencv'
    np.testing.assert_array_equal(actual,cv2.projectPoints(points,np.zeros(3),np.zeros(3),matrix,coeff)[0].reshape(-1,2))
    with pytest.raises(_RoiUnavailable):
        _conservative_rgb_roi(points,matrix,coeff,np.ones((720,1280),np.uint8),{})


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_measured_belt_projection_with_zero_rational_terms_matches_opencv(dtype):
    from types import SimpleNamespace
    from nero_core.two_second.measured_rgbd import project_to_rgb
    rng=np.random.default_rng(4711)
    points=np.column_stack((rng.uniform(-.5,.5,50000),rng.uniform(-.3,.3,50000),rng.uniform(.6,1.5,50000))).astype(dtype)
    K=np.array([[603.545776,0,640.5368],[0,603.79455,360.7278],[0,0,1.]])
    d=np.array([-.02966746,.03541105,-.000011176,.000005424,-.01283123,0,0,0])
    frame=SimpleNamespace(color_intrinsics=SimpleNamespace(matrix=K,distortion=d))
    actual,positive=project_to_rgb(points,frame,np.eye(4))
    reference=cv2.projectPoints(points@np.eye(3),np.zeros(3),np.zeros(3),K,d)[0].reshape(-1,2)
    np.testing.assert_array_equal(np.rint(actual),np.rint(reference))
    assert positive.all()
