"""Display-only snapshots of the two independently frozen observations."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation


class FrozenCloudDisplay:
    def __init__(self):
        self.clear()

    def clear(self):
        self.early = None
        self.final = None

    def freeze(self, stage, identity, capture_s, frame_sequence, points, box):
        snapshot = {'identity': tuple(identity), 'capture_s': float(capture_s),
                    'frame_sequence': int(frame_sequence),
                    'points': np.asarray(points, dtype=float).copy(),
                    'box': deepcopy(box), 'pose': None}
        setattr(self, stage, snapshot)

    def select_pose(self, stage, identity, quaternion_wxyz):
        snapshot = getattr(self, stage)
        if snapshot is None or snapshot['identity'] != tuple(identity):
            return False
        q = np.asarray(quaternion_wxyz, dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = snapshot['box']['center_world_m']
        snapshot['pose'] = pose
        return True

    def labels(self):
        def label(snapshot):
            if snapshot is None:
                return 'waiting'
            return (f"frame {snapshot['frame_sequence']}, t={snapshot['capture_s']:.2f}s, "
                    f"n={len(snapshot['points'])}, "
                    + ('pose selected' if snapshot['pose'] is not None else 'pose pending'))
        return [label(self.early), label(self.final)]
