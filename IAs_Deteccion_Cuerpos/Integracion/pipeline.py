from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from analyzers import BaseAnalyzer
from profile_store import ProfileManager


class AnalysisPipeline(threading.Thread):
    def __init__(
        self,
        analyzers: list[BaseAnalyzer],
        manager: ProfileManager,
        period: float,
    ):
        super().__init__(daemon=True, name="analysis-pipeline")
        self.analyzers = analyzers
        self.manager = manager
        self.period = period

        self._pool = ThreadPoolExecutor(
            max_workers=len(analyzers), thread_name_prefix="ia")
        self._frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()
        self._stop_evt = threading.Event()   # OJO: no llamarlo "_stop" (choca con Thread)
        self._busy = False
        self.cycle_ms = 0.0

    def wants_frame(self) -> bool:
        """True si el worker está libre (evita copiar frames de más)."""
        return not self._busy and not self._frame_ready.is_set()

    def submit(self, frame: np.ndarray) -> None:
        self._frame = frame            # quien llama pasa una copia
        self._frame_ready.set()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._frame_ready.set()
        self.join(timeout=5)
        self._pool.shutdown(wait=False, cancel_futures=True)
        for a in self.analyzers:
            a.close()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._frame_ready.wait()
            if self._stop_evt.is_set():
                break
            frame, self._frame = self._frame, None
            self._frame_ready.clear()
            if frame is None:
                continue

            self._busy = True
            t0 = time.perf_counter()

            futures = {a.name: self._pool.submit(a.run, frame)
                       for a in self.analyzers}
            results = {name: f.result() for name, f in futures.items()}

            self.manager.update(results)

            elapsed = time.perf_counter() - t0
            self.cycle_ms = elapsed * 1000
            self._busy = False

            # Pausa para no saturar la CPU (y dejar fluida la cámara)
            if elapsed < self.period:
                self._stop_evt.wait(self.period - elapsed)