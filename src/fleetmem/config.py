"""Strict, typed configuration loaded from YAML. Unknown or missing keys are errors."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EventKindConfig:
    weight: float
    slip_shift: float
    decel_shift: float
    cam_shift: float


@dataclass(frozen=True)
class SimConfig:
    road_seed: int
    n_segments: int
    seg_time_s: float
    rest_s: float
    n_vehicles: int
    duration_s: float
    time_jitter_s: float
    n_events: int
    event_min_s: float
    event_max_s: float
    effect_prob: float
    cam_dim: int
    cam_noise: float
    appearance_std: float
    slip_base: float
    slip_sd: float
    decel_base: float
    decel_sd: float
    vehicle_slip_bias_sd: float
    kinds: dict[str, EventKindConfig]


@dataclass(frozen=True)
class MemoryConfig:
    half_life_s: float
    prior_p: float
    llr_clip: float
    ledger_max_per_segment: int


@dataclass(frozen=True)
class WarnConfig:
    p_threshold: float
    min_n_eff: float


@dataclass(frozen=True)
class TimestampConfig:
    hold_s: float
    slip_trigger: float
    decel_trigger: float


@dataclass(frozen=True)
class RecommendConfig:
    window_s: float
    min_vehicles: int
    max_age_s: float
    conflict_ratio: float
    dominance: float
    max_cited: int
    max_frames: int
    policy: dict[str, list[str]]


@dataclass(frozen=True)
class EvalConfig:
    train_seed: int
    test_seeds: list[int]
    dropout_levels: list[float]


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    memory: MemoryConfig
    warn: WarnConfig
    timestamp: TimestampConfig
    recommend: RecommendConfig
    eval: EvalConfig

    def validate(self) -> None:
        s, m = self.sim, self.memory
        _require(s.n_segments > 0 and s.n_vehicles > 0, "sim needs segments and vehicles")
        _require(s.seg_time_s > 0 and s.duration_s > 0, "sim times must be positive")
        _require(0 <= s.effect_prob <= 1, "sim.effect_prob must be in [0, 1]")
        _require(s.event_min_s <= s.event_max_s, "sim.event_min_s must be <= event_max_s")
        _require(bool(s.kinds), "sim.kinds must not be empty")
        _require(sum(k.weight for k in s.kinds.values()) > 0, "sim.kinds weights must sum > 0")
        _require(0 < m.prior_p < 1, "memory.prior_p must be in (0, 1)")
        _require(m.half_life_s > 0 and m.llr_clip > 0, "memory half-life and clip must be > 0")
        _require(0 < self.warn.p_threshold < 1, "warn.p_threshold must be in (0, 1)")
        _require(
            all(0 <= d <= 1 for d in self.eval.dropout_levels), "dropout levels must be in [0, 1]"
        )
        _require(bool(self.eval.test_seeds), "eval.test_seeds must not be empty")
        _require(
            self.eval.train_seed not in self.eval.test_seeds,
            "train seed must not be a test seed",
        )
        for cause in ("reduced_traction", "visual_obstruction", "unspecified_condition_change"):
            _require(cause in self.recommend.policy, f"recommend.policy missing '{cause}'")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise ConfigError(f"{path}: missing keys {sorted(missing)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        where = f"{path}.{name}"
        if is_dataclass(tp):
            kwargs[name] = _build(tp, value, where)
        elif get_origin(tp) is dict and is_dataclass(get_args(tp)[1]):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping")
            kwargs[name] = {k: _build(get_args(tp)[1], v, f"{where}.{k}") for k, v in value.items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _build(Config, raw, "config")
    cfg.validate()
    return cfg
