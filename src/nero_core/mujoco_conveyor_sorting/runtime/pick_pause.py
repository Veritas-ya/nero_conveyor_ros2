"""Operator pause requests; hardware actions remain in the control loop.

Pausing is one-way for a grab attempt.  A paused attempt may be exited, but
must be started again to recognise and plan for the current workpiece.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
from typing import TextIO


class PauseControl:
    """Share operator requests without issuing robot commands from input threads."""

    def __init__(self) -> None:
        # SIGINT can interrupt a property read in the main thread.  Reentrancy
        # avoids waiting on a lock held by that same interrupted thread.
        self._lock = threading.RLock()
        self._pause_requested = False
        self._exit_requested = False
        self._holding = False

    @property
    def pause_requested(self) -> bool:
        with self._lock:
            return self._pause_requested

    @property
    def exit_requested(self) -> bool:
        with self._lock:
            return self._exit_requested

    @property
    def holding(self) -> bool:
        with self._lock:
            return self._holding

    def request_pause(self) -> None:
        with self._lock:
            self._pause_requested = True

    def request_exit(self) -> None:
        with self._lock:
            # The controller establishes the hold before honouring this exit.
            self._pause_requested = True
            self._exit_requested = True

    def interrupt(self) -> None:
        """SIGINT requests a pause; another SIGINT exits only after holding."""
        with self._lock:
            self._pause_requested = True
            if self._holding:
                self._exit_requested = True

    def mark_holding(self) -> None:
        """Called by the controller after it has sent the measured hold target."""
        with self._lock:
            self._pause_requested = True
            self._holding = True


class KeyboardPauseInput:
    """Read P/space to pause and Q to exit, without waiting for Enter.

    Only a terminal is read.  Pipes, redirected input and notebook stdin remain
    untouched; their caller can still wire SIGINT to ``PauseControl.interrupt``.
    ISIG is preserved so Ctrl+C continues to reach the caller's signal handler.
    """

    def __init__(self, control: PauseControl, stream: TextIO | None = None) -> None:
        self.control = control
        self.stream = sys.stdin if stream is None else stream
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._original_attributes: list | None = None
        self._terminal_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> KeyboardPauseInput:
        with self._lifecycle_lock:
            if self._thread is not None or self._stop.is_set():
                return self
            try:
                fd = self.stream.fileno()
                if not os.isatty(fd):
                    return self
                original = termios.tcgetattr(fd)
                attributes = original.copy()
                attributes[6] = original[6].copy()
                attributes[3] &= ~(termios.ICANON | termios.ECHO)
                attributes[6][termios.VMIN] = 0
                attributes[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attributes)
            except (AttributeError, OSError, ValueError, termios.error) as exc:
                # Keyboard support is optional; signals do not need a TTY.
                self.error = str(exc)
                return self
            self._fd = fd
            self._original_attributes = original
            self._thread = threading.Thread(
                target=self._read_keys, name="pick-pause-input", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._restore_terminal()
                self._thread = None
                raise
            return self

    def _read_keys(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable or self._stop.is_set():
                    continue
                keys = os.read(fd, 64)
                if not keys:
                    break
                for key in keys:
                    if key in (ord("p"), ord("P"), ord(" ")):
                        self.control.request_pause()
                    elif key in (ord("q"), ord("Q")):
                        self.control.request_exit()
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        with self._terminal_lock:
            if self._original_attributes is None or self._fd is None:
                return
            try:
                termios.tcsetattr(
                    self._fd, termios.TCSANOW, self._original_attributes)
            except (OSError, termios.error) as exc:
                self.error = str(exc)
            finally:
                self._original_attributes = None

    def close(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            try:
                if thread is not None and thread is not threading.current_thread():
                    thread.join()
            finally:
                self._restore_terminal()

    def __enter__(self) -> KeyboardPauseInput:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()
