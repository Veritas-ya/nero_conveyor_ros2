"""Size admission must preserve geometry and reject genuinely oversized parts."""
from dataclasses import replace
import numpy as np
import pytest
import yaml

from nero_core.two_second.config import CellConfig
from nero_core.two_second.fixed_freeze import NativeDepthLocator
from nero_core.two_second.phase_timing_test import prepare_test
from test_fixed_freeze_interface import data
from test_two_second_hardware import site
from test_phase_timing_test_entry import base_site


def test_long_component_uses_configured_range_without_clipping_cloud():
    cfg,frame,surface=data()
    depth=np.ones_like(frame.depth_m)
    depth[115:125,44:72]=.95  # about17cm; same centre as original target
    frame=replace(frame,depth_m=depth)
    old,report=NativeDepthLocator(cfg,surface,frame).locate(frame)
    assert not old
    assert report['component_extraction']['size_rejection_count']==1
    assert report['component_extraction']['size_measurements'][0]['reason']=='length_above_limit'
    cfg.raw['maximum_target_length_m']=.20
    candidates,report=NativeDepthLocator(cfg,surface,frame).locate(frame)
    assert len(candidates)==1 and candidates[0].point_count==280
    np.testing.assert_array_equal(frame.depth_m,depth)
    assert report['size_decisions'][0]['accepted']
    assert .16 < report['size_decisions'][0]['estimated_footprint_m'][0] < .20


def test_twenty_cm_policy_still_rejects_oversized_component():
    cfg,frame,surface=data();cfg.raw['maximum_target_length_m']=.20
    depth=np.ones_like(frame.depth_m);depth[115:125,38:78]=.95
    frame=replace(frame,depth_m=depth)
    candidates,report=NativeDepthLocator(cfg,surface,frame).locate(frame)
    assert not candidates
    assert report['component_extraction']['size_measurements'][0]['reason']=='length_above_limit'


@pytest.mark.parametrize('length',[True,0,-.2,.31,float('inf'),float('nan'),'0.20'])
def test_invalid_length_fails_config_before_hardware(site,length):
    path,raw=site;raw['maximum_target_length_m']=length;path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='maximum_target_length_m'):CellConfig.load(path)


def test_default_site_unchanged_and_only_private_test_copy_expands(site,tmp_path):
    base=base_site(site);before=base.read_bytes()
    assert CellConfig.load(base).raw['maximum_target_length_m']==.16
    snapshot=prepare_test(base,tmp_path/'size_test')
    assert CellConfig.load(snapshot).raw['maximum_target_length_m']==.20
    assert base.read_bytes()==before


def test_full_geometry_receives_same_length_policy(monkeypatch):
    import nero_core.two_second.fixed_freeze as module
    cfg,frame,surface=data();cfg.raw['maximum_target_length_m']=.20
    candidate=NativeDepthLocator(cfg,surface,frame).locate(frame)[0][0]
    class ReachedDetector(Exception):pass
    def detector(*args,**kwargs):
        assert kwargs['maximum_target_length_m']==.20
        np.testing.assert_array_equal(kwargs['target_seed_pixel_indices'],candidate.seed_pixel_indices)
        raise ReachedDetector
    with pytest.raises(ReachedDetector):
        module.frozen_geometry(cfg,frame,np.eye(4),surface,candidate,10.,detect_fn=detector)
