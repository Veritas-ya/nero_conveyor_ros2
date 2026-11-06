"""Keep uncertain clipped blobs from hiding a separate well-observed target."""
import numpy as np
from types import SimpleNamespace
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


def measured_example():
    image, depth, p = example()
    image[10:28,35:55] = [150,150,150]  # interior image blob, outside belt in 3D
    depth[10:28,35:55] = np.nan
    depth[10:16,35:55] = 1.
    camera = SimpleNamespace(rays=lambda u,v: np.c_[(u-60)*.001,
                                                   (v-60)*.001, np.ones(len(u))])
    hull = np.array([[-.025,-.025],[.025,-.025],[.025,.025],[-.025,.025]],np.float32)
    return image, depth, p, dict(camera=camera, color_to_base=np.eye(4),
        surface=SimpleNamespace(plane_to_base=np.eye(4),hull_xy=hull))


def test_measured_background_does_not_hide_independent_part():
    image, depth, p, context = measured_example()
    out, skipped = isolate_boundary_depth_gaps(image,depth,p,**context)
    assert {item['reason'] for item in skipped} == {'roi_boundary','measured_outside_belt'}
    assert not color_masks(out,p)[1][10:28,35:55].any()
    np.testing.assert_array_equal(out[50:70,50:70],image[50:70,50:70])


def test_missing_interior_depth_cannot_be_classified_as_background():
    image, depth, p, context = measured_example()
    depth[10:28,35:55] = np.nan
    out, skipped = isolate_boundary_depth_gaps(image,depth,p,**context)
    assert all(item['reason']!='measured_outside_belt' for item in skipped)
    np.testing.assert_array_equal(out[10:28,35:55],image[10:28,35:55])


def test_possible_workpiece_with_insufficient_depth_keeps_original_rejection():
    image, depth, p, context = measured_example()
    depth[50:70,50:70] = np.nan
    depth[50:54,50:70] = 1.
    out, _ = isolate_boundary_depth_gaps(image,depth,p,**context)
    np.testing.assert_array_equal(out[50:70,50:70],image[50:70,50:70])


def test_measured_boundary_background_is_distinguished_from_uncertain_clipping():
    image, depth, p, context = measured_example()
    depth[20:26,5:25] = 1.
    _, skipped = isolate_boundary_depth_gaps(image,depth,p,**context)
    assert len(skipped)==2
    assert all(item['reason']=='measured_outside_belt' for item in skipped)
