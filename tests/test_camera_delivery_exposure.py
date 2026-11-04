"""Offline delivery stalls, SDK counter faults and Gemini335L exposure units."""
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend, OrbbecCameraConfig
from mujoco_conveyor_sorting.runtime.frame_delivery import FrameDelivery
from mujoco_conveyor_sorting.runtime.types import BackendUnavailable, SampleHeader


def frame(sequence, stamp, *, index=None, epoch=0):
    return NS(header=SampleHeader(stamp, stamp, sequence, source_time=stamp, clock_epoch=epoch),
              timestamp_metadata={'depth':{'device_time_s':stamp,
                  'frame_index':sequence if index is None else index}})


def backend(**config):
    camera=OrbbecCameraBackend(OrbbecCameraConfig(**config),clock=lambda:1.)
    camera._connected=True
    return camera


def test_processing_stall_keeps_intermediate_frames_and_original_stamps():
    camera=backend()
    frames=[frame(i, .90+i/30) for i in range(1,4)]
    # Three captures have arrived while the consumer was busy for 80ms.
    for value in frames:
        camera._delivery.append(value)
        camera._latest=value
    assert camera.read_latest().header.sequence==3
    actual=[]
    for previous in (None,1,2):
        actual.append(camera.read_next(after_sequence=previous))
    assert all(a is b for a,b in zip(actual,frames))
    assert [f.header.source_time for f in actual]==pytest.approx([.90+i/30 for i in range(1,4)])
    diag=camera.get_capture_diagnostics()
    assert diag['ordered_delivered_frames']==3
    assert diag['streams']['depth']['missing_device_indices']==0
    with pytest.raises(BackendUnavailable,match='timed out'):
        camera.read_next(after_sequence=3)


def test_buffer_overflow_and_device_drops_are_independent():
    delivery=FrameDelivery(2)
    for seq,index in ((1,100),(2,101),(3,104)):
        delivery.append(frame(seq,seq/30,index=index))
    report=delivery.report()
    assert report['buffer_overflow_frames']==1
    assert report['streams']['depth']['missing_device_indices']==2
    assert delivery.take(None,.1,.25).header.sequence==2
    assert delivery.take(2,.1,.25).header.sequence==3


def test_stale_frames_never_reenter_the_tracker_and_reset_is_not_a_drop():
    delivery=FrameDelivery()
    delivery.append(frame(1,.2,index=100))
    delivery.append(frame(2,.9,index=1,epoch=1))
    assert delivery.take(None,1.,.25).header.sequence==2
    report=delivery.report()
    assert report['stale_discarded_frames']==1
    assert report['streams']['depth']['resets']==1
    assert report['streams']['depth']['missing_device_indices']==0
    assert report['streams']['depth']['received_fps'] is None


def exposure_device(camera, fail_readback=False):
    props=NS(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL='da',OB_PROP_DEPTH_EXPOSURE_INT='de',
             OB_PROP_COLOR_AUTO_EXPOSURE_BOOL='ca',OB_PROP_COLOR_EXPOSURE_INT='ce')
    camera._sdk=NS(OBPropertyID=props,OBPermissionType=NS(PERMISSION_READ_WRITE='rw'))
    camera._product_id=0x0804
    class Device:
        def __init__(self):self.values={'da':True,'de':15203,'ca':True,'ce':100};self.writes=[]
        def is_property_supported(self,p,permission):return True
        def get_int_property_range(self,p):return NS(min=1,max=20000 if p=='de' else 1999,step=1)
        def get_bool_property(self,p):return self.values[p]
        def get_int_property(self,p):return self.values[p]
        def set_bool_property(self,p,v):self.values[p]=v;self.writes.append((p,v))
        def set_int_property(self,p,v):
            assert not self.values['da' if p=='de' else 'ca']
            self.values[p]=v+(1 if fail_readback and v==3000 else 0)
            self.writes.append((p,v))
    camera._device=Device()
    return camera._device


def test_three_ms_manual_exposure_converts_color_units_reads_back_and_restores():
    camera=backend(depth_exposure_us=3000,color_exposure_us=3000)
    device=exposure_device(camera)
    camera._apply_exposure_settings()
    assert device.writes==[('da',False),('de',3000),('ca',False),('ce',30)]
    report=camera.get_capture_diagnostics()['exposure_settings']
    assert report['depth']['readback_us']==report['color']['readback_us']==3000
    camera.close()
    assert device.values=={'da':True,'de':15203,'ca':True,'ce':100}


@pytest.mark.parametrize('configuration,reason',[
    ({'color_exposure_us':3050},'multiple'),
    ({'depth_exposure_us':25000},'range'),
    ({'depth_exposure_us':40000,'depth_width':1280,'depth_height':800,'depth_fps':30},'frame period'),
    ({'color_exposure_us':3000,'enable_color':False},'without color')])
def test_invalid_exposures_fail_without_writes(configuration,reason):
    camera=backend(**configuration);device=exposure_device(camera)
    with pytest.raises(BackendUnavailable,match=reason):camera._apply_exposure_settings()
    assert device.writes==[]


def test_exposure_readback_failure_is_explicit_and_restorable():
    camera=backend(depth_exposure_us=3000)
    device=exposure_device(camera,fail_readback=True)
    with pytest.raises(BackendUnavailable,match='readback'):camera._apply_exposure_settings()
    camera.close()
    assert device.values['da'] and device.values['de']==15203


def test_metadata_conversion_does_not_silently_call_color_raw_values_microseconds():
    sdk=NS(OBFrameMetadataType=NS(EXPOSURE='e'))
    raw=NS(has_metadata=lambda p:True,get_metadata_value=lambda p:30)
    timing=OrbbecCameraBackend._frame_timestamps(raw,sdk,exposure_unit_us=100)
    assert timing['exposure_us']==3000
    assert timing['exposure_raw']==30
    assert OrbbecCameraBackend._frame_timestamps(raw,sdk)['exposure_us'] is None


@pytest.mark.parametrize('capacity',[True,0,33,8.5])
def test_invalid_delivery_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):OrbbecCameraConfig(delivery_buffer_frames=capacity)


def test_rate_limited_archive_is_not_misdiagnosed_as_camera_drops(tmp_path):
    import json
    spec=importlib.util.spec_from_file_location('camera_audit',Path(__file__).parents[1]/'tools/check_camera_stream.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=tmp_path/'rgbd_frames';directory.mkdir()
    for seq,index,stamp in ((1,11,1.),(16,26,1.5),(31,41,2.)):
        metadata={'header':{'sequence':seq,'clock_epoch':0},'timestamp_metadata':{
            'depth':{'frame_index':index,'device_time_s':stamp,'frame_metadata':{'EXPOSURE':15203}},
            'color':{'frame_metadata':{'EXPOSURE':100}}}}
        np.savez(directory/f'frame_{seq:06}.npz',metadata=np.asarray(json.dumps(metadata)))
    report=module.summarize_archive(tmp_path)
    assert report['median_device_counter_fps']==30
    assert report['unreceived_device_frames_in_saved_intervals']==0
    assert report['samples'][0]['color_exposure_us']==10000
