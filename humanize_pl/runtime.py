"""Cooperative cancellation and progress shared by API, CLI and browser flows."""

from __future__ import annotations

import threading
import warnings
from contextvars import ContextVar
from functools import wraps


class RunCancelled(BaseException):
    """Control flow: never convert a cancellation into an item/model failure."""


class RunControl:
    def __init__(self, on_progress=None):
        self._cancelled = threading.Event()
        self.on_progress = on_progress
        self.last_progress = {"stage": "oczekiwanie", "completed": None, "total": None}
        self._cleanup = []
        self._lock = threading.Lock()

    def cancel(self):
        self._cancelled.set()
        self.last_progress = {"stage": "anulowanie; oczekiwanie na zakończenie aktywnego wywołania", "completed": None, "total": None}

    def check(self):
        if self._cancelled.is_set():
            raise RunCancelled("Anulowano przebieg. Ukończone wcześniej pliki pozostają dostępne.")

    def wait(self, seconds):
        self._cancelled.wait(seconds)
        self.check()

    def report(self, stage, completed=None, total=None):
        self.check()
        self.last_progress = {"stage": stage, "completed": completed, "total": total}
        if self.on_progress:
            self.on_progress(self.last_progress)
        self.check()

    def cleanup(self, callback):
        with self._lock:
            self._cleanup.append(callback)

    def close(self):
        with self._lock:
            callbacks, self._cleanup = self._cleanup, []
        for callback in reversed(callbacks):
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - keep the original processing/cancellation error
                warnings.warn(f"Nie udało się zamknąć zasobu: {type(exc).__name__}.", ResourceWarning, stacklevel=2)


_current: ContextVar[RunControl | None] = ContextVar("humanize_run_control", default=None)


def current_control():
    return _current.get()


def checkpoint(stage="przetwarzanie", completed=None, total=None):
    control = current_control()
    if control:
        control.report(stage, completed, total)


def controlled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        requested = kwargs.pop("control", None)
        inherited = current_control()
        control = requested or inherited or RunControl()
        token = _current.set(control)
        try:
            control.check()
            return function(*args, **kwargs)
        finally:
            _current.reset(token)
            if control is not inherited:
                control.close()
    return wrapped
