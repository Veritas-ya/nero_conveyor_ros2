"""Task-space requirements of the final drop leg, independent of devices."""
from types import SimpleNamespace as NS
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected,RobotModel
from mujoco_conveyor_sorting.runtime.joint_stream_sender import FeedbackRecheckRequired
from nero_core.two_second.drop_orientation import check_final_orientation
from nero_core.two_second.return_feedback import ReturnFeedbackCheck


class TaskModel:
    ik=NS(config=NS(orientation_tolerance=.01))
    def set_state(self,q,width):self.q=q
    def tcp(self):
        t=np.eye(4);t[:3,3]=self.q[:3]
        t[:3,:3]=Rotation.from_rotvec(self.q[3:6]).as_matrix()
        return t


def test_rotation_must_finish_before_final_region_not_just_at_endpoint():
    p=NS(model=TaskModel(),config=NS(raw={'geometry_check_hz':25.}))
    def q(t):return np.array([.2*(1-t),0.,0.,0.,0.,.2*(1-t),0.])
    path=NS(duration=1.,position=q)
    with pytest.raises(PickRejected,match='姿态仍未完成'):
        check_final_orientation(p,path,.04,np.eye(4),.1,cancelled=lambda:False)
    def early(t):
        value=q(t);value[5]=max(0.,1.-t/.3)*.2;return value
    path.position=early
    assert check_final_orientation(p,path,.04,np.eye(4),.1,cancelled=lambda:False)==0.
    with pytest.raises(PickRejected,match='取消'):
        check_final_orientation(p,path,.04,np.eye(4),.1,cancelled=lambda:True)


def test_real_incident_next_packet_passes_original_threshold_within_existing_budget():
    rows=json.loads((Path(__file__).parent/'fixtures/transport_feedback_20261114.json').read_text())['rows']
    clock=[rows[0]['read_monotonic_s']]
    model=RobotModel()
    gate=ReturnFeedbackCheck(model.check_feedback,lambda:clock[0],enabled=True)
    with pytest.raises(FeedbackRecheckRequired) as caught:gate.observe(rows[0]['state'])
    details=caught.value.original_error.details
    assert details['position_error_m']==pytest.approx(.011055356756582018,abs=1e-9)
    assert details['position_tolerance_m']==.01 and details['orientation_tolerance_deg']==2.
    clock[0]=rows[1]['read_monotonic_s']
    assert clock[0]-rows[0]['read_monotonic_s']<.03
    gate.observe(rows[1]['state'])
    assert gate.pending is None and gate.resumed[0]['targets_sent_from_rejected_feedback']==0


def test_unreachable_early_orientation_does_not_silently_use_late_rotation(monkeypatch):
    from test_prepared_taught_cycle import fixture
    from nero_core.two_second.prepared_cycle import prepare_cycle,nominal_state
    cfg,p,q=fixture();original=p.payload_local
    cfg.raw.update(experimental_direct_drop_guard={'radius_m':.2},direct_drop_orientation_first=True)
    def reject(*args,**kwargs):raise PickRejected('fixed orientation unavailable')
    monkeypatch.setattr('nero_core.two_second.drop_orientation.orientation_first_drop',reject)
    with pytest.raises(PickRejected,match='fixed orientation unavailable'):
        prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert 'direct_drop' not in p.calls and p.payload_local is original
