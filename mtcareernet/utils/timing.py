import time
from contextlib import contextmanager


class StopWatch:
    def __init__(self): self.events = {}; self._t = None
    def tick(self, label="default"): self._t = time.perf_counter(); return self
    def tock(self, label="default"):
        dt = time.perf_counter() - (self._t or time.perf_counter())
        self.events.setdefault(label, []).append(dt); return dt


@contextmanager
def chrono(label: str = "default"):
    t = time.perf_counter(); yield; dt = time.perf_counter() - t
    print(f"[chrono::{label}] {dt:.4f}s")
