"""Per-user, nonblocking ownership of a camera shared by separate viewers."""

import errno
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from ..types import BackendUnavailable


class CameraLease:
    def __init__(self, name="dabai_dc1_rgb", *, label="相机窗口"):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("camera lease name must contain 1..64 ASCII letters, digits, underscores or hyphens")
        self.name, self.label = name, str(label)[:256]
        self._fd = None
        directory = Path(tempfile.gettempdir())
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if runtime:
            candidate = Path(runtime)
            try:
                info = candidate.stat(follow_symlinks=False)
                if (stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                        and not info.st_mode & 0o077 and os.access(candidate, os.W_OK | os.X_OK)):
                    directory = candidate
            except OSError:
                pass
        self.path = directory / f"camera-lease-{os.getuid()}-{name}.lock"

    def _busy_message(self, fd):
        label, pid = "已有相机窗口", "未知"
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = json.loads(os.read(fd, 4096))
            if isinstance(owner.get("pid"), int) and owner["pid"] > 0:
                pid = str(owner["pid"])
            label = str(owner.get("label") or label)[:256]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return (f"相机正被“{label}”占用（进程号 {pid}）。"
                "请关闭已有尺寸分析或高清 RGB 窗口后重试，无需重复 sudo 或拔插相机。")

    def acquire(self):
        if self._fd is not None:
            return self
        fd = None
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise BackendUnavailable("相机锁文件不是当前用户拥有的普通文件")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise BackendUnavailable(self._busy_message(fd)) from error
            payload = json.dumps({"pid": os.getpid(), "label": self.label}, ensure_ascii=False).encode("utf-8")
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
        except Exception as error:
            if fd is not None:
                os.close(fd)
            if isinstance(error, BackendUnavailable):
                raise
            raise BackendUnavailable(f"无法获取相机使用锁 {self.path}：{error}") from error
        self._fd = fd
        return self

    def release(self):
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
