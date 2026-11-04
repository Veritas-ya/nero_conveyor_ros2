import json
import numpy as np
import pytest
from experiments.graspgenx_sim.trajectory_overlay import historical_samples, tcp_path_from_samples


class Robot:
    def set_state(self, q, width): self.q = q
    def tcp(self):
        pose = np.eye(4)
        pose[:3, 3] = self.q[:3]
        return pose


def test_tcp_overlay_converts_base_samples_and_preserves_phase_boundary():
    samples = [[('approach' if i < 5 else 'closing'), float(i), [i, 0, 0, 0, 0, 0, 0]] for i in range(10)]
    transform = np.eye(4); transform[:3, 3] = [1, 2, 3]
    points = tcp_path_from_samples(samples, Robot(), transform, max_points=3)
    assert any(phase == 'closing' and stamp == 5 for phase, stamp, _ in points)
    np.testing.assert_allclose(points[0][2], [1, 2, 3])
    np.testing.assert_allclose(points[-1][2], [10, 2, 3])


def test_history_requires_certificate_and_matching_plan(tmp_path):
    path = tmp_path/'run.json'
    samples = [['approach', 0, [0]*7], ['lift', 1, [1]*7]]
    record = {'complete_path_certified': True, 'plan_request_id': 5,
              'planner_results': [{'request_id': 5, 'full_pick_schedule': {'certified_samples': samples}}]}
    path.write_text(json.dumps(record))
    assert historical_samples(path) == samples
    record['complete_path_certified'] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): historical_samples(path)
