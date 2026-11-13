"""Read atomic preview artifacts; never access a camera, CAN, or a planner."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def points(value, maximum):
    data = np.asarray(value, dtype=float).reshape(-1, 3)
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) > maximum:
        data = data[np.linspace(0, len(data)-1, maximum, dtype=int)]
    return np.round(data, 5).tolist()


def image_bytes(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('相机 RGB 数据格式无效')
    if rgb.shape[1] > 960:
        rgb = cv2.resize(rgb, (960, round(rgb.shape[0]*960/rgb.shape[1])))
    good, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not good:
        raise ValueError('相机预览图编码失败')
    return encoded.tobytes()


def display_cloud(depth, metadata):
    k = metadata['depth_intrinsics']
    z = np.asarray(depth)[::8, ::8]
    v, u = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
    local = np.stack(((u-k['cx'])*z/k['fx'], (v-k['cy'])*z/k['fy'], z), axis=-1)
    valid = np.isfinite(z) & (z > 0)
    transform = np.asarray(metadata['depth_to_base_cv'])
    cloud = local[valid]@transform[:3, :3].T+transform[:3, 3]
    return points(cloud[np.linalg.norm(cloud, axis=1) < 2.], 5500)


class ArtifactReader:
    def __init__(self):
        self.run = None
        self.keys = {}
        self.live_image = self.frozen_image = None
        self.latest = self.target = self.plan = None
        self.error = None

    def reset(self, run):
        self.__init__()
        self.run = run

    def _key(self, path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def update(self, run):
        run = None if run is None else str(run)
        if run != self.run:
            self.reset(run)
        if run is None:
            return
        directory = Path(run)/'preview'
        for kind in ('latest', 'target', 'plan'):
            path = directory/(kind+'.npz')
            key = self._key(path)
            if key == self.keys.get(kind):
                continue
            if key is None:
                setattr(self, kind, None)
                self.keys.pop(kind, None)
                if kind == 'target':
                    self.frozen_image = None
                if kind == 'latest':
                    self.live_image = None
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata']))
                    if kind == 'latest':
                        image = image_bytes(data['frame_rgb'])
                        scene = display_cloud(data['frame_depth_m'], meta)
                        self.latest = {'metadata': meta, 'scene': scene}
                        self.live_image = image
                    elif kind == 'target':
                        image = image_bytes(data['frame_rgb'])
                        self.target = {'metadata': meta, 'points': points(data['points'], 3000),
                            'point_count': len(data['points']), 'grasps': data['grasps'].tolist()}
                        self.frozen_image = image
                    else:
                        self.plan = {'metadata': meta, 'tcp': points(data['tcp_xyz_base_m'], 1000),
                            'contact_grasp': data['contact_grasp_base'].tolist(),
                            'predicted_points': points(data['predicted_contact_points_base_m'], 2000)}
                self.keys[kind] = key
                self.error = None
            except (OSError, ValueError, KeyError, IndexError) as error:
                self.error = '预览文件暂不可读：'+str(error)

    def snapshot(self):
        latest = self.latest or {}
        frame_meta = latest.get('metadata', {})
        frame_header = frame_meta.get('header', {})
        received = frame_header.get('received_time')
        target = self.target or {}
        meta = target.get('metadata', {})
        sequence = meta.get('header', {}).get('sequence')
        plan = self.plan
        paired = bool(plan and target and plan['metadata'].get('identity') == meta.get('identity')
                      and plan['metadata'].get('frame_sequence') == sequence)
        return {'revision': '-'.join(str(self.keys.get(k, 0)) for k in ('latest', 'target', 'plan')),
            'frame': {'available': bool(self.live_image), 'sequence': frame_header.get('sequence'),
                'version': str(self.keys['latest']) if 'latest' in self.keys else None,
                'age_s': None if not isinstance(received, (int, float)) else max(0., time.monotonic()-received)},
            'scene': latest.get('scene', []),
            'target': {'available': bool(target), 'sequence': sequence,
                'identity': meta.get('identity'), 'point_count': target.get('point_count', 0),
                'points': target.get('points', []), 'grasps': target.get('grasps', []),
                'obb': meta.get('obb'), 'version': str(self.keys['target']) if 'target' in self.keys else None},
            'plan': {'available': paired, 'hidden_mismatch': bool(plan and not paired),
                'tcp': plan['tcp'] if paired else [],
                'predicted_points': plan['predicted_points'] if paired else [],
                'contact_grasp': plan['contact_grasp'] if paired else None,
                'metadata': plan['metadata'] if paired else {}},
            'error': self.error}


class DemoReader:
    """Explicit synthetic browser-test view, with no hardware or SDK imports."""
    def __init__(self):
        self.run = None
        self.active = False
        self.live_image = self.frozen_image = self._image()
        self.counter = 0

    @staticmethod
    def _image():
        # Draw a UI demonstration diagram; this is not a recorded exposure.
        image = np.full((540, 960, 3), (40, 49, 53), np.uint8)
        cv2.rectangle(image, (70, 30), (900, 530), (73, 91, 98), -1)
        cv2.rectangle(image, (110, 30), (850, 530), (50, 112, 100), -1)
        for y in range(35, 540, 38):
            cv2.line(image, (110, y), (850, y), (53, 120, 108), 2)
        cv2.line(image, (110, 300), (850, 300), (147, 227, 197), 2)
        polygon = np.array([[394, 200], [527, 178], [556, 280], [419, 309]])
        cv2.fillPoly(image, [polygon], (225, 189, 104))
        cv2.polylines(image, [polygon], True, (248, 219, 147), 3)
        cv2.rectangle(image, (367, 152), (578, 329), (126, 236, 213), 2)
        cv2.drawMarker(image, (472, 246), (245, 248, 249), cv2.MARKER_CROSS, 28, 2)
        cv2.putText(image, 'DEMO / NO CAMERA', (135, 75), cv2.FONT_HERSHEY_SIMPLEX, .65, (221, 240, 231), 1)
        return image_bytes(image)

    def update(self, run):
        self.run = run

    def snapshot(self):
        rng = np.random.default_rng(7)
        x, y = np.meshgrid(np.linspace(-.25, .25, 44), np.linspace(-.6, .65, 65))
        scene = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        target = rng.uniform([-.045, .15, .015], [.045, .26, .055], (1800, 3))
        t = np.linspace(0, 1, 80)
        path = np.c_[.15*(1-t)**2, -.4+.6*t, .24*(1-t)**2+.05]
        pose = np.eye(4); pose[:3, 3] = [0, .205, .05]
        return {'revision': 'demo', 'frame': {'available': True, 'sequence': 1284,
            'version': 'demo', 'age_s': 0., 'demo': True}, 'scene': points(scene, 5500),
            'target': {'available': True, 'sequence': 1250, 'identity': [0, 0, 1250],
                'point_count': len(target), 'points': points(target, 3000),
                'grasps': [pose.tolist()], 'version': 'demo',
                'obb': {'center_world_m': [0., .205, .035], 'extent_m': [.09, .11, .04],
                        'axes_world_columns': np.eye(3).tolist()}},
            'plan': {'available': True, 'tcp': points(path, 1000), 'predicted_points': [],
                'contact_grasp': pose.tolist(), 'metadata': {'camera_only': True,
                    'frame_sequence': 1250, 'path_duration_s': 2., 'planning_wall_s': .236,
                    'actual_remaining_send_lead_s': .184, 'demo': True}}, 'error': None}
