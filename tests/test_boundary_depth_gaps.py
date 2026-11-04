"""Keep uncertain clipped blobs from hiding a separate well-observed target."""
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.detection import DETECTION_DEFAULTS, color_masks
from mujoco_conveyor_sorting.runtime.nero7_boundary_depth import isolate_boundary_depth_gaps


def example():
    image=np.zeros((100,100,3),np.uint8);image[:]=[0,150,0]
    image[20:40,5:25]=[140,140,140]  # ROI-clipped background
    image[50:70,50:70]=[180,180,180]  # independent part
    depth=np.ones((100,100),np.float32);depth[20:40,5:25]=np.nan
    p=dict(DETECTION_DEFAULTS,roi_px=[5,5,95,95],min_area_px=20)
    return image,depth,p


def test_boundary_invalid_component_removed_while_independent_part_unchanged():
    image,depth,p=example();out,skipped=isolate_boundary_depth_gaps(image,depth,p)
    assert len(skipped)==1 and skipped[0]['valid_pixels']==0
    np.testing.assert_array_equal(out[50:70,50:70],image[50:70,50:70])
    assert not color_masks(out,p)[1][25:35,10:20].any()
    assert color_masks(out,p)[1][55:65,55:65].all()


def test_interior_insufficient_depth_is_not_silenced():
    image,depth,p=example();depth[50:70,50:70]=np.nan
    out,skipped=isolate_boundary_depth_gaps(image,depth,p)
    assert len(skipped)==1
    np.testing.assert_array_equal(out[50:70,50:70],image[50:70,50:70])


def test_boundary_component_with_valid_depth_is_not_removed():
    image,depth,p=example();depth[20:40,5:25]=1.
    out,skipped=isolate_boundary_depth_gaps(image,depth,p)
    assert not skipped
    np.testing.assert_array_equal(out,image)
