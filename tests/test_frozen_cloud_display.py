import numpy as np
import mujoco

from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud
from experiments.two_second_grasp.frozen_cloud_display import FrozenCloudDisplay


def test_snapshots_are_distinct_and_only_show_selected_pose():
    display = FrozenCloudDisplay()
    points = np.array([[1., 2., 3.], [2., 3., 4.]])
    box = {'center_world_m': [1., 2., 3.]}
    identity = (0, 7, 1)
    display.freeze('early', identity, 1., 42, points, box)
    points[0, 0] = 9.
    box['center_world_m'][0] = 9.
    assert display.early['points'][0, 0] == 1.
    assert display.early['pose'] is None
    assert not display.select_pose('early', (0, 8, 1), [1, 0, 0, 0])
    assert display.select_pose('early', identity, [1, 0, 0, 0])
    np.testing.assert_allclose(display.early['pose'][:3, 3], [1, 2, 3])
    display.freeze('final', identity, 2., 62, points, box)
    assert display.final['pose'] is None
    assert 'frame 42' in display.labels()[0] and 'frame 62' in display.labels()[1]
    display.clear()
    assert display.labels() == ['waiting', 'waiting']


def test_two_point_clouds_coexist_in_mujoco_scene():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    scene = mujoco.MjvScene(model, maxgeom=40)
    box = {'center_world_m': [0., 0., 0.], 'axes_world_columns': np.eye(3),
           'extent_m': [.1, .1, .1]}
    points = np.zeros((4, 3))
    draw_live_cloud(scene, points, np.eye(4), obb=box, clear=False,
                    point_rgba=(.85, .2, 1., .9), reserve_geoms=20)
    assert scene.ngeom == 20
    draw_live_cloud(scene, points, np.eye(4), obb=box, clear=False,
                    point_rgba=(.05, .9, 1., .9))
    assert scene.ngeom == 40
    np.testing.assert_allclose(scene.geoms[0].rgba, [.85, .2, 1., .9])
    np.testing.assert_allclose(scene.geoms[20].rgba, [.05, .9, 1., .9])
    assert draw_live_cloud(scene, points, np.eye(4), obb=box, clear=False) == 0
