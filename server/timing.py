"""Per-request stage timer: every stage records time.perf_counter_ns() durations (microseconds)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Timer:
    def __init__(self) -> None:
        self.us: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter_ns() - start) / 1000.0)

    def add(self, name: str, us: float) -> None:
        self.us[name] = self.us.get(name, 0.0) + us
