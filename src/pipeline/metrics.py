"""
Per-stage latency tracking.

You cannot tune a voice pipeline you cannot measure: the product metric is
"how long after I stop talking does it start answering?" and it decomposes
exactly into endpoint wait + ASR + respond + TTS-first-chunk. We keep a
rolling window per stage and report p50/p95 (never averages — latency is
long-tailed and the tail is what users feel).

In real deployments this would be Prometheus histograms + traces; the
structure (name -> observations -> percentiles) is identical.
"""

from __future__ import annotations

from collections import defaultdict, deque


class Metrics:
    def __init__(self, window: int = 500):
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._counters: dict[str, int] = defaultdict(int)

    def observe(self, stage: str, ms: float) -> None:
        self._samples[stage].append(ms)

    def count(self, name: str, n: int = 1) -> None:
        self._counters[name] += n

    @staticmethod
    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, round(q * (len(s) - 1)))
        return s[idx]

    def summary(self) -> dict:
        return {
            "latency_ms": {
                stage: {
                    "count": len(vals),
                    "p50": round(self._pct(list(vals), 0.50), 1),
                    "p95": round(self._pct(list(vals), 0.95), 1),
                    "last": round(vals[-1], 1),
                }
                for stage, vals in self._samples.items()
                if vals
            },
            "counters": dict(self._counters),
        }
