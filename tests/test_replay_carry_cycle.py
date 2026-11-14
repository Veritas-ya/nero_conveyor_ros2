"""Recorded scenes, plans and feedback may never be mixed between cycles."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_replay_selects_artifacts_by_delivered_identity_not_glob_order(tmp_path,monkeypatch):
    tool=Path(__file__).parents[1]/'tools/replay_carry_lookahead.py'
    spec=importlib.util.spec_from_file_location('replay_cycle_test',tool)
    replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
    # The test only exercises source pairing; reference and model are stubs.
    monkeypatch.setattr(replay,'SynchronizedIntercept',lambda **kw:NS(**kw))
    monkeypatch.setattr(replay,'MeasuredPlanner',lambda cfg:NS(model=None,
        install=lambda target,state:installed.append(target)))
    monkeypatch.setattr(replay,'nominal_state',lambda *args:None)
    installed=[];events=[];(tmp_path/'experiment').mkdir()
    for cycle,identity in [(2,[0,9,200]),(1,[0,4,100])]:
        # Deliberately write cycle 2 first and give it the first filename.
        stem='a' if cycle==2 else 'b'
        events.append({'event':'execution_event','data':{'cycle_index':cycle,
            'record':{'event':'planning_delivery','data':{'identity':identity}}}})
        target={'identity':identity,'detection':{},'velocity':[0.,.1,0.],
                'capture_s':float(cycle),'capture_state':{}}
        np.savez(tmp_path/'experiment'/f'{stem}_target.npz',metadata=json.dumps(target),
            scene_points=np.zeros((1,3)),scene_pixel_indices=[cycle],points=np.full((1,3),cycle))
        plan={'identity':identity,'reference':{},'version':cycle,'stage':'pick',
              'duration_s':1.,'planning_s':.1,'diagnostics':{},'scene_revision':cycle}
        np.savez(tmp_path/'experiment'/f'{stem}_plan.npz',metadata=json.dumps(plan),
            curve_c=np.full((1,1,7),cycle),curve_x=[0.,1.])
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
    for cycle in (1,2):
        p,path,_,event=replay.load_case(tmp_path,NS(settings=NS(open_width_m=.1)),cycle)
        assert path.identity==installed[-1].identity==tuple(event('planning_delivery')['data']['identity'])
        assert path.version==cycle and p.scene_revision==cycle
        assert installed[-1].capture_s==cycle
        np.testing.assert_array_equal(path.q0,np.full(7,cycle))
    with pytest.raises(ValueError,match='cycle 3: missing planning_delivery'):
        replay.load_case(tmp_path,NS(),3)
    # Never silently pick one of two plans for the same delivered identity.
    source=tmp_path/'experiment'/'b_plan.npz'
    (source.parent/'duplicate_plan.npz').write_bytes(source.read_bytes())
    with pytest.raises(ValueError,match='expected one plan'):
        replay.load_case(tmp_path,NS(settings=NS(open_width_m=.1)),1)
