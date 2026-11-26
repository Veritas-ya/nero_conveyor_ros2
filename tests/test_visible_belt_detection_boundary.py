from dataclasses import replace

import cv2
import numpy as np
import pytest

from test_fixed_freeze_interface import data
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, visible_detection_boundary
from nero_core.two_second.fixed_freeze import NativeDepthLocator, frozen_geometry


def scene(monkeypatch, top_row=115):
    config,frame,surface=data()
    config.raw['perception_belt_edge_margin_m']=.005
    hull=cv2.convexHull(np.array([[-.9,-.5],[.8,-.5],[.8,-.015],[-.9,-.015]],np.float32))
    surface=replace(surface,hull_xy=hull,bounds_xy_m=np.array([[-.9,.8],[-.5,-.015]]))
    green=np.zeros(frame.rgb.shape[:2],bool)
    green[top_row:196,25:281]=True
    green[frame.depth_m < 1.] = False
    monkeypatch.setattr('mujoco_conveyor_sorting.runtime.frame_products.green_rgb_mask',lambda *a:green)
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *a:True)
    return config,frame,surface


def test_visible_far_edge_reaches_both_detection_stages_without_changing_scene(monkeypatch):
    config,frame,surface=scene(monkeypatch)
    original_depth=frame.depth_m.copy();native_hull=surface.hull_xy.copy()
    assert not NativeDepthLocator(config,surface,frame).locate(frame)[0]
    updated=visible_detection_boundary(surface,frame,np.eye(4))
    assert updated.detection_boundary_diagnostics['applied']
    candidates,report=NativeDepthLocator(config,updated,frame).locate(frame)
    assert len(candidates)==1 and candidates[0].point_count==100
    result,*_=frozen_geometry(config,frame,np.eye(4),updated,candidates[0],10.,clock=lambda:10.2)
    assert len(result['object_points'])==100
    assert report['detection_bounds_source']=='visible_green_outline_on_measured_plane'
    np.testing.assert_array_equal(updated.hull_xy,native_hull)
    np.testing.assert_array_equal(updated.bounds_xy_m,surface.bounds_xy_m)
    np.testing.assert_array_equal(result['belt_region']['bounds_xy_m'],surface.bounds_xy_m)
    np.testing.assert_array_equal(frame.depth_m,original_depth)


def test_outside_visible_green_remains_rejected(monkeypatch):
    config,frame,surface=scene(monkeypatch,top_row=121)
    updated=visible_detection_boundary(surface,frame,np.eye(4))
    assert updated.detection_boundary_diagnostics['applied']
    candidates,report=NativeDepthLocator(config,updated,frame).locate(frame)
    assert not candidates
    assert report['size_decisions'][0]['reason']=='belt_edge'


def test_inconsistent_visible_outline_falls_back_to_native_hull(monkeypatch):
    config,frame,surface=scene(monkeypatch,top_row=0)
    updated=visible_detection_boundary(surface,frame,np.eye(4))
    assert updated.detection_hull_xy is None
    assert updated.detection_boundary_diagnostics['reason']=='visible_outline_inconsistent_with_native_belt'
    assert not NativeDepthLocator(config,updated,frame).locate(frame)[0]


def test_missing_visible_green_falls_back_to_native_hull(monkeypatch):
    config,frame,surface=scene(monkeypatch,top_row=200)
    updated=visible_detection_boundary(surface,frame,np.eye(4))
    assert updated.detection_hull_xy is None
    assert not updated.detection_boundary_diagnostics['applied']
    np.testing.assert_array_equal(updated.hull_xy,surface.hull_xy)
