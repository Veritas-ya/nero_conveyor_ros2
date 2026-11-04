import numpy as np
import mujoco
import pytest

from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud


def test_overlay_draws_capture_and_pose_without_modifying_model():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    scene = mujoco.MjvScene(model, maxgeom=12)
    points = np.arange(60, dtype=float).reshape(-1, 3) / 1000
    assert draw_live_cloud(scene, points, np.eye(4)) == 5
    assert scene.ngeom == 9
    assert model.ngeom == 0
    assert draw_live_cloud(scene, points[:1]) == 1
    assert scene.ngeom == 1
    with pytest.raises(ValueError):
        draw_live_cloud(scene, np.array([[float('nan'), 0, 0]]))
