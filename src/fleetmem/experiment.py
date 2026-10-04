"""End-to-end experiment: train on one simulated fleet run, evaluate on held-out runs.

Test runs use different seeds, so they have different vehicles' biases, event placements and noise
(road layout and event-type signatures are shared). Sensor dropout is applied to test data only.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .detectors import Detector, build_detectors
from .memory import DecayMemory, TimestampMemory
from .metrics import aggregate, summarize
from .recommend import EvidenceLedger, Recommender
from .replay import replay
from .schema import Observation
from .sim import generate

METHODS = ("timestamp", "can_only", "camera_only", "fusion")


def scenarios(levels: list[float]) -> list[tuple[str, float, float]]:
    """(name, p_drop_can, p_drop_cam). Level 0 is the clean scenario."""
    out: list[tuple[str, float, float]] = [("clean", 0.0, 0.0)]
    for lv in levels:
        if lv > 0:
            out.append((f"drop_cam_{lv:.2f}", 0.0, lv))
    for lv in levels:
        if lv > 0:
            out.append((f"drop_can_{lv:.2f}", lv, 0.0))
    return out


def apply_dropout(
    obs: list[Observation], p_can: float, p_cam: float, rng: np.random.Generator
) -> list[Observation]:
    out = []
    for o in obs:
        drop_can = bool(rng.random() < p_can)
        drop_cam = bool(rng.random() < p_cam)
        out.append(o.drop(can=drop_can, cam=drop_cam))
    return out


def _run_method(det: Detector, obs, labels, events, cfg: Config) -> dict:
    llrs, contribs = det.evaluate(obs)
    if det.memory_kind == "timestamp":
        memory = TimestampMemory(cfg.timestamp)
        passes = replay(obs, llrs, contribs, labels, memory, cfg.warn)
    else:
        memory = DecayMemory(cfg.memory)
        ledger = EvidenceLedger(cfg.memory.ledger_max_per_segment)
        rec = Recommender(cfg.recommend, cfg.warn, memory, ledger)
        passes = replay(obs, llrs, contribs, labels, memory, cfg.warn, ledger, rec)
    return summarize(passes, events)


def run(cfg: Config) -> dict:
    train = generate(cfg.sim, cfg.eval.train_seed)
    detectors = build_detectors(cfg.timestamp)
    for det in detectors.values():
        det.fit(train.observations, train.labels)

    per_scenario: dict[str, dict[str, list[dict]]] = {}
    for seed in cfg.eval.test_seeds:
        test = generate(cfg.sim, seed)
        for name, p_can, p_cam in scenarios(cfg.eval.dropout_levels):
            rng = np.random.default_rng([seed, int(p_can * 100), int(p_cam * 100)])
            obs = apply_dropout(test.observations, p_can, p_cam, rng)
            for method in METHODS:
                result = _run_method(detectors[method], obs, test.labels, test.events, cfg)
                per_scenario.setdefault(name, {}).setdefault(method, []).append(result)

    return {
        "note": "All numbers come from the synthetic fleet simulator, not from real vehicle data.",
        "train_seed": cfg.eval.train_seed,
        "test_seeds": cfg.eval.test_seeds,
        "scenarios": {
            name: {m: aggregate(runs) for m, runs in methods.items()}
            for name, methods in per_scenario.items()
        },
    }
