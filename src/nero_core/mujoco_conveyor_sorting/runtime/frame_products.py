"""Lazy products owned by one immutable exposure, never a global frame cache."""
from threading import RLock
import weakref
import cv2
import numpy as np
from .belt_surface import _geometry_fingerprint
from .depth_diagnostics import measured_stage


class FrameProducts:
    def __init__(self, frame, d2c):
        try:
            self._frame_ref = weakref.ref(frame)
        except TypeError:
            self._frame_ref = lambda: frame  # uncached legacy/test objects
        self.key = (_geometry_fingerprint(frame, d2c)
                    if all(hasattr(frame, name) for name in
                        ('depth_m', 'depth_intrinsics', 'camera_to_world_cv')) else None)
        self.arrays = tuple(getattr(frame, n, None) for n in ('depth_m', 'rgb', 'left_ir', 'right_ir'))
        self.header = getattr(frame, 'header', None)
        self.lock = RLock()
        self.values, self.timings_s, self.calls = {}, {}, {}

    @property
    def frame(self):
        return self._frame_ref()

    def matches(self, frame, d2c):
        return (self.frame is frame and self.header is frame.header
                and all(a is getattr(frame, n, None) for a, n in zip(self.arrays,
                    ('depth_m', 'rgb', 'left_ir', 'right_ir')))
                and self.key == _geometry_fingerprint(frame, d2c))

    def get(self, name, compute):
        with self.lock:
            if name not in self.values:
                label = name[0] if isinstance(name, tuple) else name
                with measured_stage(label, self.timings_s):
                    value = compute()
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
                self.values[name] = value
                self.calls[label] = self.calls.get(label, 0) + 1
            return self.values[name]


def frame_products(frame, d2c):
    """Bind arrays AND geometry; mutable legacy frames use an unshared cache."""
    arrays = [getattr(frame, n, None) for n in ('depth_m', 'rgb', 'left_ir', 'right_ir')]
    cacheable = (getattr(frame, 'header', None) is not None
        and hasattr(frame, 'depth_intrinsics') and arrays[0] is not None
        and all(a is None or (isinstance(a, np.ndarray) and not a.flags.writeable) for a in arrays))
    cached = getattr(frame, '_perception_products', None) if cacheable else None
    if cached is not None and cached.matches(frame, d2c):
        return cached
    result = FrameProducts(frame, d2c)
    if cacheable:
        object.__setattr__(frame, '_perception_products', result)
    return result


def green_rgb_mask(frame, d2c):
    products = frame_products(frame, d2c)
    hsv = products.get('rgb_to_hsv', lambda: cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV))
    return products.get('green_rgb_mask', lambda: cv2.inRange(
        hsv, np.array([35, 40, 35]), np.array([95, 255, 255])) > 0)


def world_points(frame, d2c):
    from ..perception import deproject_depth_to_world
    return frame_products(frame, d2c).get('native_deprojection', lambda:
        deproject_depth_to_world(frame.depth_m, frame.depth_intrinsics.matrix,
                                frame.camera_to_world_cv, camera_convention='opencv'))
