"""Does a shared, decaying memory of other vehicles' passes predict the next vehicle's slowdown better than history alone?

Task: a vehicle is about to enter a road cell. Will its pass be slow (mean speed below `slow_ratio` x the cell's free-flow
speed)? Predictors use only what was visible before it entered:

  static   P(slow | cell, weekday/weekend, time-of-day block) from the training days
  last     was the most recent visible pass of the cell slow (within last_window_s), else static
  memory   the decaying fleet memory, shrunk toward the static prior
  fused    logistic regression on static, memory, confidence and last-pass features, fitted on the validation days

Scored on the test days with day-block bootstrap intervals. The fleet-size experiment lets the memory learn from only a
random share of the vehicles while every pass is still scored."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from .config import Config
from .replay import StaticPrior, condition_table, replay

METHODS = ["static", "last", "memory", "fused"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(F: pd.DataFrame, window: float) -> np.ndarray:
    has = F.last_slow.notna().to_numpy()
    return np.column_stack([
        _logit(F.prior.to_numpy()), _logit(F.p_mem.to_numpy()), np.minimum(F.n_eff.to_numpy(), 10.0),
        np.where(has, F.last_slow.fillna(0).to_numpy(), 0.0), has.astype(float),
        np.log1p(np.where(has, F.last_age.fillna(window).to_numpy(), window)),
    ])


def scores(F_val: pd.DataFrame, F_test: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    w = cfg.memory.last_window_s
    model = LogisticRegression(C=1.0, max_iter=1000).fit(design(F_val, w), F_val.slow.astype(int))
    last = np.where(F_test.last_slow.notna(), F_test.last_slow.fillna(0), F_test.prior)
    return {"static": F_test.prior.to_numpy(), "last": np.asarray(last, dtype=float), "memory": F_test.p_mem.to_numpy(),
            "fused": model.predict_proba(design(F_test, w))[:, 1]}


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"ap": float("nan"), "auroc": float("nan"), "brier": float("nan")}
    return {"ap": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)), "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}


def day_bootstrap(F: pd.DataFrame, sc: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    """95% intervals from resampling whole days (passes within a day share weather and events)."""
    rng = np.random.default_rng(seed)
    days = np.floor(F.day.to_numpy()).astype(int)
    uniq = np.unique(days)
    where = {d: np.flatnonzero(days == d) for d in uniq}
    y = F.slow.to_numpy().astype(int)
    out = {m: {"ap": [], "auroc": []} for m in sc}
    delta = {"ap": [], "auroc": []}
    for _ in range(n_boot):
        idx = np.concatenate([where[d] for d in rng.choice(uniq, size=len(uniq))])
        res = {m: metrics(y[idx], sc[m][idx]) for m in sc}
        for m in sc:
            out[m]["ap"].append(res[m]["ap"])
            out[m]["auroc"].append(res[m]["auroc"])
        delta["ap"].append(res["fused"]["ap"] - res["static"]["ap"])
        delta["auroc"].append(res["fused"]["auroc"] - res["static"]["auroc"])
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"ci": {m: {k: q(v) for k, v in d.items()} for m, d in out.items()}, "fused_minus_static_ci": {k: q(v) for k, v in delta.items()}}


def warning_point(y_val, s_val, y_test, s_test, target: float) -> dict:
    """Loosest threshold with at least `target` precision on validation, applied unchanged to the test days."""
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    ok = np.flatnonzero(prec[:-1] >= target)
    if len(ok) == 0:
        return {"threshold": None}
    t = float(thr[ok[0]])
    warn = s_test >= t
    tp = int((warn & (y_test == 1)).sum())
    return {"threshold": t, "precision": tp / max(int(warn.sum()), 1), "recall": tp / max(int(y_test.sum()), 1),
            "warnings_per_1000_passes": 1000.0 * float(warn.mean())}


def run(cfg: Config, passes: pd.DataFrame, out_dir: Path) -> dict:
    df = condition_table(passes, cfg)
    train = df[df.period == "train"]
    prior = StaticPrior(train, cfg).predict(df.key.to_numpy(), df.day.to_numpy(), df.t0.to_numpy())
    vehicles = np.sort(df.veh.unique())
    rng = np.random.default_rng(cfg.seed)
    res = {"data": {"passes": int(len(df)), "cells": int(df.key.nunique()), "vehicles": int(len(vehicles)),
                    "slow_rate": {p: float(df[df.period == p].slow.mean()) for p in ("train", "val", "test")},
                    "passes_by_period": {p: int((df.period == p).sum()) for p in ("train", "val", "test")}},
           "fleet_fraction": {}}
    main = None
    for frac in cfg.evaluate.fleet_fractions:
        chosen = rng.choice(vehicles, size=max(1, int(round(frac * len(vehicles)))), replace=False)
        feed = df.veh.isin(chosen).to_numpy()
        F = replay(df, prior, cfg, feed)
        val, test = F[F.period == "val"], F[F.period == "test"]
        sc = scores(val, test, cfg)
        y = test.slow.to_numpy().astype(int)
        res["fleet_fraction"][str(frac)] = {m: metrics(y, s) for m, s in sc.items()}
        res["fleet_fraction"][str(frac)]["fresh_share"] = float((test.n_eff >= cfg.evaluate.fresh_n_eff).mean())
        if frac == max(cfg.evaluate.fleet_fractions):
            main = (val, test, sc)
    val, test, sc = main
    y = test.slow.to_numpy().astype(int)
    fresh = (test.n_eff >= cfg.evaluate.fresh_n_eff).to_numpy()
    surprise = (test.prior < 0.1).to_numpy()
    res["test"] = {
        "all": {m: metrics(y, s) for m, s in sc.items()},
        "fresh_evidence": {m: metrics(y[fresh], s[fresh]) for m, s in sc.items()},
        "no_fresh_evidence": {m: metrics(y[~fresh], s[~fresh]) for m, s in sc.items()},
        "unexpected": {m: metrics(y[surprise], s[surprise]) for m, s in sc.items()},
        "shares": {"fresh_evidence": float(fresh.mean()), "unexpected_cells": float(surprise.mean()),
                   "slow_rate_when_unexpected": float(y[surprise].mean()) if surprise.any() else float("nan")},
    }
    res["bootstrap"] = day_bootstrap(test, sc, cfg.evaluate.n_boot, cfg.seed)
    yv = val.slow.to_numpy().astype(int)
    scv = scores(val, val, cfg)
    res["warning"] = {m: warning_point(yv, scv[m], y, sc[m], cfg.evaluate.warn_precision) for m in ("static", "memory", "fused")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({"y": y, **{m: s for m, s in sc.items()}}).sample(min(len(y), 200000), random_state=0).to_parquet(out_dir / "test_scores_sample.parquet")
    return res
