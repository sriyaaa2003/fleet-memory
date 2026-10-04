"""Per-segment condition memory.

DecayMemory keeps, per road segment, a log-odds deviation from a configured prior. Each observation
adds a (clipped) log-likelihood ratio; between observations the deviation decays exponentially, so
stale evidence fades and a segment returns to the prior. TimestampMemory is the baseline: a segment
is flagged for a fixed hold window after the last raw trigger, with no confidence or decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import MemoryConfig, TimestampConfig


@dataclass(frozen=True)
class SegmentState:
    p: float  # posterior probability that the segment's condition has changed
    n_eff: float  # decayed count of observations backing the state (confidence)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class DecayMemory:
    def __init__(self, cfg: MemoryConfig):
        self._cfg = cfg
        self._prior = _logit(cfg.prior_p)
        self._dev: dict[int, float] = {}
        self._n: dict[int, float] = {}
        self._last: dict[int, float] = {}
        self._seen: set[str] = set()

    def _at(self, segment: int, t: float) -> tuple[float, float]:
        if segment not in self._last:
            return 0.0, 0.0
        dt = max(0.0, t - self._last[segment])  # a late (out-of-order) update decays by 0
        f = 0.5 ** (dt / self._cfg.half_life_s)
        return self._dev[segment] * f, self._n[segment] * f

    def predict(self, segment: int, t: float) -> SegmentState:
        dev, n = self._at(segment, t)
        return SegmentState(p=_sigmoid(self._prior + dev), n_eff=n)

    def update(self, obs_id: str, segment: int, t: float, llr: float | None) -> bool:
        """Fold one observation in. Returns False if ignored (duplicate id or no evidence)."""
        if obs_id in self._seen:
            return False
        self._seen.add(obs_id)
        if llr is None:
            return False
        dev, n = self._at(segment, t)
        c = self._cfg.llr_clip
        self._dev[segment] = dev + max(-c, min(c, llr))
        self._n[segment] = n + 1.0
        self._last[segment] = max(t, self._last.get(segment, t))
        return True


class TimestampMemory:
    def __init__(self, cfg: TimestampConfig):
        self._hold = cfg.hold_s
        self._last_trigger: dict[int, float] = {}
        self._seen: set[str] = set()

    def predict(self, segment: int, t: float) -> SegmentState:
        last = self._last_trigger.get(segment)
        active = last is not None and 0.0 <= t - last <= self._hold
        return SegmentState(p=1.0 if active else 0.0, n_eff=1.0 if active else 0.0)

    def update(self, obs_id: str, segment: int, t: float, llr: float | None) -> bool:
        if obs_id in self._seen:
            return False
        self._seen.add(obs_id)
        if llr is not None and llr > 0:
            self._last_trigger[segment] = max(t, self._last_trigger.get(segment, t))
            return True
        return False
