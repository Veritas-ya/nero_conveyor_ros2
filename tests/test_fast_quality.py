import numpy as np

from nero_core.two_second.fast_quality import filter_reconstruction


def sample():
    mask=np.zeros((50,60),bool);mask[10:40,15:45]=True
    y,x=np.where(mask)
    points=np.column_stack((x*.002,y*.002,np.full(len(x),.05)))
    depth=np.ones(mask.shape);depth[mask]=.95
    return points,mask,depth


def check(native,seed,depth,mask,points,z):
    return filter_reconstruction(native,seed,depth,mask,points,z,np.eye(4))


def test_small_consistent_refinement_preserves_outline_and_inputs():
    native,mask,depth=sample();points=native.copy();points[:,2]+=.003
    before=points.copy();clean,r=check(native,mask,depth,mask,points,np.full(len(points),.947))
    assert r['accepted'] and clean.sum()==mask.sum()
    np.testing.assert_array_equal(points,before)
    assert r['native_coverage_fraction']==1.


def test_edge_depth_tail_removed_without_blind_mask_erosion():
    native,mask,depth=sample();points=native.copy();z=np.full(len(points),.95)
    tail=np.where(mask)[1]>=42;points[tail,1]+=.060;points[tail,2]-=.040;z[tail]+=.060
    clean,r=check(native,mask,depth,mask,points,z)
    assert r['accepted'] and clean.sum()==mask.sum()-tail.sum()
    assert not clean[10:40,42:45].any()
    assert clean[10:40,15].all()  # Good opposite boundary remains.
    assert r['same_pixel_depth_rejected']==tail.sum()


def test_large_uniform_depth_error_rejects_whole_target():
    native,mask,depth=sample();points=native.copy();points[:,2]-=.04
    _,r=check(native,mask,depth,mask,points,np.full(len(points),.99))
    assert not r['accepted'] and r['retained_points']==0


def test_small_native_hole_can_be_completed_near_measured_support():
    full,mask,depth=sample();seed=mask.copy();seed[24:27,28:31]=False
    native=full[seed[mask]];depth[~seed]=0.
    clean,r=check(native,seed,depth,mask,full,np.full(len(full),.95))
    assert r['accepted'] and clean[24:27,28:31].all()


def test_visible_regions_split_by_holes_retain_all_native_supported_parts():
    native,mask,depth=sample();candidate=mask.copy();candidate[:,29:31]=False
    points=native[candidate[mask]]
    clean,r=check(native,mask,depth,candidate,points,np.full(len(points),.95))
    assert r['accepted'] and r['disconnected_pixels_removed']==0
    np.testing.assert_array_equal(clean,candidate)


def test_missing_half_is_rejected_instead_of_shrinking_the_object():
    native,mask,depth=sample();candidate=mask.copy();candidate[:,30:]=False
    points=native[candidate[mask]]
    _,r=check(native,mask,depth,candidate,points,np.full(len(points),.95))
    assert not r['accepted'] and 'insufficient_native_shape_coverage' in r['reasons']


def test_depth_consistent_but_spatially_detached_prediction_is_rejected():
    native,mask,depth=sample();points=native+np.array([0,.08,0])
    _,r=check(native,mask,depth,mask,points,np.full(len(points),.95))
    assert not r['accepted'] and r['native_distance_rejected']==len(native)


def test_invalid_reference_does_not_accept_prediction():
    native,mask,depth=sample();native[0,0]=np.nan
    clean,r=check(native,mask,depth,mask,np.zeros((mask.sum(),3)),np.full(mask.sum(),.95))
    assert not clean.any() and not r['accepted']


def test_site_thresholds_relax_only_global_coverage_and_center():
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.fast_perception import FastTargetPerception
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml')
    quality=FastTargetPerception(cfg, '.',backend=object()).quality_filter
    # Dense, genuine native samples, with one end missing from the reconstructed patch.
    mask=np.ones((10,100),bool)
    y,x=np.nonzero(mask);native=np.c_[x*.0015,y*.002,np.ones(len(x))*.05]
    seed=mask.copy();subset=mask.copy();subset[:,:13]=False;points=native[subset[mask]]
    args=(native,seed,np.ones_like(mask,float),subset,points,np.ones(len(points)),np.eye(4))
    strict=filter_reconstruction(*args)[1];relaxed=quality(*args)[1]
    assert 'center_shifted' in strict['reasons'] and relaxed['accepted']
    assert relaxed['limits']['maximum_center_shift_m']==.014
    assert relaxed['limits']['minimum_native_coverage_fraction']==.70
    _,bad=quality(native,seed,np.ones_like(mask,float),subset,points+.08,np.ones(len(points)),np.eye(4))
    assert not bad['accepted'] and bad['retained_points']==0


def test_relaxed_coverage_still_enforces_extent_retention_and_pointwise_gates():
    seed=np.ones((50,50),bool);y,x=np.where(seed)
    native=np.c_[x*.002,y*.002,np.full(len(x),.05)]
    mask=seed.copy();mask[10:40,10:40]=False
    args=(native,seed,np.ones(seed.shape),mask,native[mask.ravel()],np.ones(mask.sum()),np.eye(4))
    strict=filter_reconstruction(*args)[1]
    relaxed=filter_reconstruction(*args,min_native_coverage=.70,max_center_shift_m=.014)[1]
    assert .70 <= relaxed['native_coverage_fraction'] < .80
    assert 'insufficient_native_shape_coverage' in strict['reasons'] and relaxed['accepted']
    narrow=seed.copy();narrow[:,30:]=False
    _,report=filter_reconstruction(native,seed,np.ones(seed.shape),narrow,native[narrow.ravel()],
        np.ones(narrow.sum()),np.eye(4), min_native_coverage=.70,max_center_shift_m=.014)
    assert not report['accepted'] and 'shape_collapsed' in report['reasons']
