"""Time-ordered replay of the fleet's passes: what would a vehicle arriving at a road cell have known?

A pass becomes visible to the memory when it *ends* (t1). A vehicle about to enter a cell at t0 queries the memory before
its own pass is folded in, and only passes that ended at or before t0 are visible, so nothing from the future leaks.
Updates are processed before queries at equal times."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .memory import DecayMemory


def condition_table(passes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep scorable cells and label every pass slow / not slow against its cell's free-flow speed (from training days only)."""
    c, p = cfg.condition, cfg.periods
    train = passes[(passes.day >= p.train_days[0]) & (passes.day < p.train_days[1])]
    g = train.groupby("key").v_mean
    stats = pd.DataFrame({"n": g.size(), "ff": g.quantile(c.free_flow_quantile)})
    cells = stats[(stats.n >= c.min_train_passes) & (stats.ff >= c.min_free_flow_kmh)]
    df = passes[passes.key.isin(cells.index)].copy()
    df["ff"] = df.key.map(cells.ff)
    df["slow"] = df.v_mean < c.slow_ratio * df.ff
    df["period"] = np.select([df.day < p.train_days[1], df.day < p.val_days[1]], ["train", "val"], "test")
    return df.reset_index(drop=True)


def bucket(day: np.ndarray, t: np.ndarray, hour_block: int) -> np.ndarray:
    """0..(2 * 24/hour_block - 1): weekday/weekend x time-of-day block. Day 0 of VED is a Wednesday."""
    weekday = (np.floor(day).astype(int) + 2) % 7
    weekend = (weekday >= 5).astype(int)
    hour = (t % 86400.0) // 3600.0
    return weekend * (24 // hour_block) + (hour // hour_block).astype(int)


class StaticPrior:
    """P(slow | cell, weekday/weekend, time-of-day block), smoothed toward the cell's overall slow rate. Training days only."""

    def __init__(self, train: pd.DataFrame, cfg: Config):
        self.block = cfg.static.hour_block
        self.m = cfg.static.smoothing
        b = bucket(train.day.to_numpy(), train.t0.to_numpy(), self.block)
        d = pd.DataFrame({"key": train.key.to_numpy(), "b": b, "slow": train.slow.to_numpy().astype(float)})
        cell = d.groupby("key").slow.mean()
        self.cell_rate = cell
        agg = d.groupby(["key", "b"]).slow.agg(["sum", "count"])
        self.table = {(k, bb): (s, n) for k, bb, s, n in zip(agg.index.get_level_values(0), agg.index.get_level_values(1), agg["sum"], agg["count"])}
        self.default = float(d.slow.mean())

    def predict(self, keys: np.ndarray, day: np.ndarray, t: np.ndarray) -> np.ndarray:
        b = bucket(day, t, self.block)
        out = np.empty(len(keys))
        for i, (k, bb) in enumerate(zip(keys, b)):
            base = self.cell_rate.get(k, self.default)
            s, n = self.table.get((k, bb), (0.0, 0))
            out[i] = (s + self.m * base) / (n + self.m)
        return out


def replay(df: pd.DataFrame, prior: np.ndarray, cfg: Config, feed: np.ndarray | None = None) -> pd.DataFrame:
    """Memory features for every pass in the validation and test periods.

    `feed` marks which passes the memory may learn from (all by default); the fleet-size experiment passes a subset."""
    feed = np.ones(len(df), dtype=bool) if feed is None else feed
    mem = DecayMemory(cfg.memory.half_life_s, cfg.memory.prior_weight)
    key, t0, t1, slow = df.key.to_numpy(), df.t0.to_numpy(), df.t1.to_numpy(), df.slow.to_numpy()
    query = (df.period != "train").to_numpy()
    n = len(df)
    # events: updates (type 0) at t1, queries (type 1) at t0; at equal times updates come first
    times = np.concatenate([t1[feed], t0[query]])
    kind = np.concatenate([np.zeros(feed.sum(), dtype=int), np.ones(query.sum(), dtype=int)])
    idx = np.concatenate([np.flatnonzero(feed), np.flatnonzero(query)])
    order = np.lexsort((kind, times))
    p_mem = np.full(n, np.nan)
    n_eff = np.zeros(n)
    last = np.full(n, np.nan)
    age = np.full(n, np.nan)
    for e in order:
        i = idx[e]
        if kind[e] == 0:
            mem.update(i, int(key[i]), float(t1[i]), bool(slow[i]))
        else:
            st = mem.query(int(key[i]), float(t0[i]), float(prior[i]))
            p_mem[i], n_eff[i] = st.p, st.n_eff
            if st.last_slow is not None and st.age_s <= cfg.memory.last_window_s:
                last[i], age[i] = st.last_slow, st.age_s
    out = df.assign(prior=prior, p_mem=p_mem, n_eff=n_eff, last_slow=last, last_age=age)
    return out[out.period != "train"].reset_index(drop=True)
