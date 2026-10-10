"""Per-road-cell condition memory built from other vehicles' passes.

Each cell keeps an exponentially decayed count of slow passes `s` and of all passes `w` (evidence halves every
`half_life_s`). The estimate shrinks toward a static prior p0 that counts as `prior_weight` observations:

    p = (s + prior_weight * p0) / (w + prior_weight)

so with no fresh evidence the memory returns the prior, and with plenty it follows the fleet. `n_eff` = w is the confidence
that goes with the estimate. Updates are idempotent per pass id and a late update decays by zero."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellState:
    p: float          # probability the next pass is slow
    n_eff: float      # decayed number of passes behind it
    last_slow: float | None   # was the most recent pass slow (None if there is none)
    age_s: float | None       # seconds since that pass


class DecayMemory:
    def __init__(self, half_life_s: float, prior_weight: float):
        if half_life_s <= 0 or prior_weight <= 0:
            raise ValueError("half_life_s and prior_weight must be positive")
        self.half_life = half_life_s
        self.prior_weight = prior_weight
        self._s: dict[int, float] = {}
        self._w: dict[int, float] = {}
        self._t: dict[int, float] = {}
        self._last: dict[int, tuple[float, float]] = {}      # key -> (time of last pass, slow flag)
        self._seen: set[int] = set()

    def _decay(self, key: int, t: float) -> float:
        if key not in self._t:
            return 0.0
        return 0.5 ** (max(0.0, t - self._t[key]) / self.half_life)

    def update(self, pass_id: int, key: int, t: float, slow: bool) -> bool:
        if pass_id in self._seen:
            return False
        self._seen.add(pass_id)
        f = self._decay(key, t)
        self._s[key] = self._s.get(key, 0.0) * f + float(slow)
        self._w[key] = self._w.get(key, 0.0) * f + 1.0
        self._t[key] = max(t, self._t.get(key, t))
        if key not in self._last or t >= self._last[key][0]:
            self._last[key] = (t, float(slow))
        return True

    def query(self, key: int, t: float, prior: float) -> CellState:
        f = self._decay(key, t)
        s, w = self._s.get(key, 0.0) * f, self._w.get(key, 0.0) * f
        p = (s + self.prior_weight * prior) / (w + self.prior_weight)
        last = self._last.get(key)
        return CellState(p=p, n_eff=w, last_slow=None if last is None else last[1], age_s=None if last is None else max(0.0, t - last[0]))
