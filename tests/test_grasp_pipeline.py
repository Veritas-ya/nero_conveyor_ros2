from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from nero_core import grasp_pipeline
from mujoco_conveyor_sorting.runtime.types import TargetIdentity

class Backend:
    def __init__(self):
        self.started = Event(); self.scene = Event(); self.calls = 0
    def infer(self, points, *args, **kwargs):
        self.calls += 1; self.started.set()
        assert self.scene.wait(2), 'scene must run while inference is active'
        return {'poses': np.zeros((1,4,4)), 'scores': np.ones(1), 'metadata': {}}

class Model:
    def __init__(self, backend): self.backend=backend
    def install_scene(self, *args, **kwargs):
        assert self.backend.started.wait(2)
        self.backend.scene.set(); return {'installed': True}

def detection():
    return {'target_identity':TargetIdentity.confirmed(1., 0),'object_points':np.ones((3,3)),
            'target_cloud_filter':{},'frame':SimpleNamespace(camera_to_world_cv=np.eye(4)),
            'scene_state':{},'scene_points':np.ones((2,3)), 'mask':None}

def test_inference_runs_concurrently_and_selection_uses_prepared(monkeypatch):
    backend=Backend(); ob=detection()
    def selector(ob, model, state, supplied_backend, **kw):
        assert supplied_backend is backend and backend.scene.is_set()
        prep=kw['prepared_inference']
        result=prep.resolve(prep.points,backend,*prep.parameters,target_identity=ob['target_identity'])
        assert len(result['scores']) == 1
        return {'selected':'network-candidate'}
    monkeypatch.setattr(grasp_pipeline, 'select_measured_network_grasp', selector)
    out=grasp_pipeline.prepare_and_select(ob,Model(backend),backend,motion_region={},settings=object(),snapshot_provider=lambda _:None)
    assert out.observation['selected']=='network-candidate' and backend.calls==1
    assert out.timings['inference']['selection_wait_s']>=0
    assert out.timings['overlapped_delivery_wall_s'] >= out.timings['scene_wall_s']

def test_reject_missing_identity_before_inference():
    backend=Backend(); ob=detection(); ob['target_identity']=None
    with pytest.raises(ValueError):
        grasp_pipeline.prepare_and_select(ob,Model(backend),backend,motion_region={},settings=object(),snapshot_provider=lambda _:None)
    assert backend.calls==0
