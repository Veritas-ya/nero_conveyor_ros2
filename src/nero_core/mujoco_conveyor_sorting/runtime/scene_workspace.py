"""One scene-install request's immutable spatial/provenance indexes.

Content keys permit exact copies of the SAME points, never reuse after a
changed point set. ContextVar scopes the cache to one installation/thread.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from .depth_diagnostics import measured_stage

_active = ContextVar('scene_workspace', default=None)


class SceneWorkspace:
    def __init__(self):
        self.trees, self.pixels = {}, {}
        self.timings_s, self.calls = {}, {}

    def key(self, points):
        with measured_stage('index_key', self.timings_s):
            array = np.ascontiguousarray(points, dtype=np.float64)
            digest = hashlib.sha256(memoryview(array).cast('B')).digest() if array.size else b''
        return (array.shape, digest), array


class _TimedTree:
    def __init__(self, tree, workspace):
        self.tree, self.workspace = tree, workspace

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def query(self, *args, **kwargs):
        return self._query('nearest_query', self.tree.query, *args, **kwargs)

    def query_ball_point(self, *args, **kwargs):
        return self._query('radius_query', self.tree.query_ball_point, *args, **kwargs)

    def _query(self, name, function, *args, **kwargs):
        self.workspace.calls[name] = self.workspace.calls.get(name, 0)+1
        with measured_stage(name, self.workspace.timings_s):
            return function(*args, **kwargs)


def spatial_tree(points):
    current = _active.get()
    if current is None:
        return cKDTree(points)
    key, values = current.key(points)
    previous = current.trees.get(key)
    if previous is not None and np.array_equal(previous.data, values):
        current.calls['tree_hits'] = current.calls.get('tree_hits', 0)+1
        return previous
    with measured_stage('tree_build', current.timings_s):
        tree = cKDTree(values, copy_data=True)
        tree.data.setflags(write=False)
        tree = _TimedTree(tree, current)
    current.trees[key] = tree
    current.calls['tree_build'] = current.calls.get('tree_build', 0)+1
    return tree


def scene_call(name, function, *args, **kwargs):
    current = _active.get()
    if current is None:
        return function(*args, **kwargs)
    current.calls[name] = current.calls.get(name, 0)+1
    with measured_stage(name, current.timings_s):
        return function(*args, **kwargs)


def native_pixels(points, frame_key, compute):
    current = _active.get()
    if current is None:
        return compute()
    key, values = current.key(points)
    key = (frame_key, key)
    previous = current.pixels.get(key)
    if previous is not None and np.array_equal(previous[0], values):
        current.calls['native_pixels_hits'] = current.calls.get('native_pixels_hits', 0)+1
        return previous[1]
    with measured_stage('native_pixels', current.timings_s):
        output = compute()
    current.pixels[key] = (values.copy(), output)
    current.calls['native_pixels'] = current.calls.get('native_pixels', 0)+1
    return output


@contextmanager
def scene_workspace():
    workspace = SceneWorkspace()
    token = _active.set(workspace)
    try:
        yield workspace
    finally:
        _active.reset(token)
