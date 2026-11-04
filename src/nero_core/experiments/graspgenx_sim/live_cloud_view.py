"""Read-only MuJoCo overlay for measured cloud and selected network TCP."""
import numpy as np
import mujoco


def draw_live_cloud(scene, points_world, grasp_world=None, *, max_points=3500, obb=None,
                    clear=True, point_rgba=(.05, .9, 1., .9),
                    grasp_rgba=(1., .35, .05, 1.), reserve_geoms=0):
    """Replace overlay with one exposure's points and corresponding TCP.

    The geometry is display-only; it never mutates MuJoCo model/data. The jaw
    sketch is a fixed 80 mm orientation guide, not a predicted gripper width.
    """
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('expected finite world points of shape (N, 3)')
    pose = None if grasp_world is None else np.asarray(grasp_world, dtype=float)
    if pose is not None and (pose.shape != (4,4) or not np.isfinite(pose).all()):
        raise ValueError('expected finite world TCP transform')
    if clear:
        scene.ngeom = 0
    available = max(0, scene.maxgeom - scene.ngeom - int(reserve_geoms))
    extra = (4 if pose is not None else 0) + (12 if obb is not None else 0)
    if clear:
        # Preserve the original caller contract, including its headroom.
        extra = max(extra, 19 if obb is not None else 7)
    if available < extra:
        return 0
    count = min(len(points), int(max_points), available - extra)
    indices = np.linspace(0, len(points)-1, count, dtype=int) if count else []
    for point in points[indices]:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([.002, 0., 0.]), point, np.eye(3).ravel(),
            np.asarray(point_rgba, dtype=np.float32))
        scene.ngeom += 1
    if pose is not None:
        local = np.array([[0,-.04,0],[0,-.04,-.04],[0,.04,-.04],[0,.04,0],
                          [0,0,-.08],[0,0,0]])
        world = local @ pose[:3,:3].T + pose[:3,3]
        for first, last in ((0,1),(1,2),(2,3),(4,5)):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                np.asarray(grasp_rgba, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .003, world[first], world[last])
            scene.ngeom += 1
    if obb is not None:
        center = np.asarray(obb['center_world_m'], dtype=float)
        axes = np.asarray(obb['axes_world_columns'], dtype=float)
        half = np.asarray(obb['extent_m'], dtype=float) / 2
        if center.shape != (3,) or axes.shape != (3,3) or half.shape != (3,) or not np.isfinite(np.r_[center,axes.ravel(),half]).all():
            raise ValueError('invalid visible OBB')
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for a in (-1, 1):
                for b in (-1, 1):
                    middle = center + a*half[other[0]]*axes[:,other[0]] + b*half[other[1]]*axes[:,other[1]]
                    first = middle-half[axis]*axes[:,axis]; last=middle+half[axis]*axes[:,axis]
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.1,1.,.15,1.],dtype=np.float32))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,first,last)
                    scene.ngeom += 1
    return count
