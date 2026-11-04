"""Linux cooperative motion ownership for SDK and future ROS send adapters.

Protocol nero7-motion-v1: SHA256(socketcan:<channel>) in the abstract AF_UNIX
namespace, independent of project/UID/files. Unmodified CAN writers are outside
this protocol. No bus is opened and no enable/stop/motion command is generated.
"""
import errno
import hashlib
import os
import socket
import threading
import weakref


class MotionLeaseUnavailable(RuntimeError):
    pass


_leases = weakref.WeakSet()


def _after_fork():
    # A child must neither send with its parent's authority nor retain its fd.
    for lease in list(_leases):
        if lease._socket is not None:
            lease._socket.close()
            lease._socket = None
        lease._pid = None
        lease._lock = threading.RLock()


os.register_at_fork(after_in_child=_after_fork)


class MotionLease:
    def __init__(self, channel, *, owner='sdk'):
        if not isinstance(channel, str) or not channel or '/' in channel or '\x00' in channel:
            raise ValueError('CAN channel must be a nonempty canonical interface name')
        self.channel, self.owner = channel, str(owner)
        self.address = '\x00nero7-motion-v1-'+hashlib.sha256(('socketcan:'+channel).encode()).hexdigest()
        self._socket = None
        self._pid = None
        self._lock = threading.RLock()
        _leases.add(self)

    def acquire(self):
        with self._lock:
            if self._socket is not None:
                self.assert_owned()
                return self
            handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                handle.bind(self.address)
            except OSError as error:
                handle.close()
                if error.errno == errno.EADDRINUSE:
                    raise MotionLeaseUnavailable(f'{self.channel}运动控制权已由其他SDK/ROS发送者持有') from error
                raise
            self._socket, self._pid = handle, os.getpid()
            return self

    def assert_owned(self):
        if self._socket is None or self._socket.fileno()<0 or self._pid != os.getpid():
            raise MotionLeaseUnavailable('当前进程没有有效运动发送租约')

    def guard(self, send):
        def guarded(*args, **kwargs):
            with self._lock:
                self.assert_owned()
                return send(*args, **kwargs)
        return guarded

    def close(self):
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            self._pid = None

    def __enter__(self):return self.acquire()
    def __exit__(self, *_):self.close()
