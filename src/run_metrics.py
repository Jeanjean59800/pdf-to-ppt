from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricEntry:
    count: int = 0
    total_seconds: float = 0.0


@dataclass
class RunMetrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, MetricEntry] = field(default_factory=lambda: defaultdict(MetricEntry))

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def record_duration(self, name: str, seconds: float) -> None:
        entry = self.timings[name]
        entry.count += 1
        entry.total_seconds += seconds

    @contextmanager
    def timed(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_duration(name, time.perf_counter() - start)

    def snapshot(self) -> dict[str, dict]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "timings": {
                name: {
                    "count": entry.count,
                    "total_seconds": round(entry.total_seconds, 4),
                    "avg_seconds": round(entry.total_seconds / entry.count, 4) if entry.count else 0.0,
                }
                for name, entry in sorted(self.timings.items())
            },
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return path


run_metrics = RunMetrics()

