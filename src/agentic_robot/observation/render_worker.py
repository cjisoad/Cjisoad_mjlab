"""Keep MuJoCo OpenGL contexts on one dedicated thread."""
from __future__ import annotations

from concurrent.futures import Future
from queue import Queue
from threading import Thread


class RenderWorker:
    def __init__(self):
        self._tasks = Queue()
        self._thread = Thread(target=self._run, name='mujoco-render', daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            function, args, future = self._tasks.get()
            if function is None:
                future.set_result(None)
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)

    def call(self, function, *args):
        if not self._thread.is_alive():
            raise RuntimeError('MuJoCo render worker is closed')
        future = Future()
        self._tasks.put((function, args, future))
        return future.result()

    def close(self):
        if self._thread.is_alive():
            future = Future()
            self._tasks.put((None, (), future))
            future.result()
            self._thread.join()
