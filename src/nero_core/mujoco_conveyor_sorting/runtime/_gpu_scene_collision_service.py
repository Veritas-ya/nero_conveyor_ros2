"""Persistent optional CUDA process for a lightweight hardware Python env.

Uses the existing GraspGen-X Python environment; never imports camera/CAN code.
Scene uploads are cached by content and query replies contain only B x G flags.
"""
from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading


def _read(stream):
    header = stream.read(8)
    if len(header) != 8:
        raise EOFError("CUDA scene process closed its reply stream")
    remaining = struct.unpack("!Q", header)[0]
    parts = []
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("CUDA scene process returned an incomplete reply")
        parts.append(part)
        remaining -= len(part)
    return pickle.loads(b"".join(parts))


def _write(stream, value):
    # Hardware env uses NumPy 2 and GraspGen-X uses NumPy 1.26. Protocol 5's
    # NumPy 2 _core.numeric reducer is unavailable in NumPy 1.26; protocol 4
    # uses the compatible ndarray reconstruction entry point in both envs.
    data = pickle.dumps(value, protocol=4)
    stream.write(struct.pack("!Q", len(data)))
    stream.write(data)
    stream.flush()


class _Service:
    def __init__(self, python):
        project = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(project), env.get("PYTHONPATH"))))
        # Optional user-local development headers installed with the GPU
        # backend. No system-package or main-environment changes are made.
        include = Path.home()/".cache/nero-planning/python-dev/usr/include"
        if (include/"python3.10/Python.h").exists():
            env["CPATH"] = os.pathsep.join(filter(None, (str(include), str(include/"python3.10"), env.get("CPATH"))))
        self.process = subprocess.Popen(
            [str(python), "-u", "-m", __name__], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=project)
        self.lock = threading.Lock()
        self.loaded = set()
        self.references = {}
        atexit.register(self.close)

    def call(self, message):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("optional CUDA scene process exited")
            _write(self.process.stdin, message)
            result = _read(self.process.stdout)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.kill()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


_SERVICE = None
_SERVICE_LOCK = threading.Lock()


class RemoteCudaObservedPointQueries:
    def __init__(self, points, equations, python):
        global _SERVICE
        with _SERVICE_LOCK:
            if _SERVICE is None or _SERVICE.process.poll() is not None:
                _SERVICE = _Service(python)
            self.service = _SERVICE
        digest = hashlib.sha256(points.tobytes())
        digest.update(str(points.shape).encode())
        for equation in equations:
            digest.update(equation.tobytes())
            digest.update(str(equation.shape).encode())
        self.key = digest.hexdigest()
        # The process owns one immutable upload per point/hull combination.
        with _SERVICE_LOCK:
            try:
                if self.key not in self.service.loaded:
                    self.service.call({"operation": "load", "key": self.key,
                                       "points": points, "equations": equations})
                    self.service.loaded.add(self.key)
                self.service.references[self.key] = self.service.references.get(self.key, 0) + 1
            except Exception:
                if not self.service.references:
                    self.service.close()
                    if _SERVICE is self.service:
                        _SERVICE = None
                raise
        self.closed = False

    def candidates(self, positions, rotations, radii, margins, bounds, nearby=None):
        return self.service.call({"operation": "query", "key": self.key,
                                  "arguments": (positions, rotations, radii, margins, bounds, nearby)})

    def close(self):
        global _SERVICE
        with _SERVICE_LOCK:
            if self.closed:
                return
            self.closed = True
            remaining = self.service.references.get(self.key, 1)-1
            if remaining:
                self.service.references[self.key] = remaining
            else:
                self.service.references.pop(self.key, None)
                self.service.loaded.discard(self.key)
                if self.service.process.poll() is None:
                    self.service.call({"operation": "unload", "key": self.key})
            if not self.service.references:
                self.service.close()
                if _SERVICE is self.service:
                    _SERVICE = None


def _main():
    from ._gpu_scene_collision_kernel import CudaObservedPointQueries
    scenes = {}
    while True:
        try:
            message = _read(sys.stdin.buffer)
        except EOFError:
            break
        try:
            if message["operation"] == "load":
                scenes[message["key"]] = CudaObservedPointQueries(message["points"], message["equations"])
                result = True
            elif message["operation"] == "query":
                result = scenes[message["key"]].candidates(*message["arguments"])
            elif message["operation"] == "unload":
                scenes.pop(message["key"], None)
                result = True
            else:
                raise ValueError("unknown CUDA scene operation")
            _write(sys.stdout.buffer, {"result": result})
        except Exception as error:
            _write(sys.stdout.buffer, {"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    _main()
