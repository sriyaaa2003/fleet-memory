"""Typed configuration loaded from YAML; unknown or missing keys are errors."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Data:
    raw_dir: str
    passes_file: str
    cell_lat_deg: float
    cell_lon_deg: float
    heading_bins: int
    min_records_per_pass: int
    min_speed_records: int


@dataclass(frozen=True)
class Periods:
    train_days: list[int]
    val_days: list[int]
    test_days: list[int]


@dataclass(frozen=True)
class Condition:
    min_train_passes: int
    free_flow_quantile: float
    min_free_flow_kmh: float
    slow_ratio: float


@dataclass(frozen=True)
class Memory:
    half_life_s: float
    prior_weight: float
    last_window_s: float


@dataclass(frozen=True)
class Static:
    hour_block: int
    smoothing: float


@dataclass(frozen=True)
class Evaluate:
    fleet_fractions: list[float]
    warn_precision: float
    n_boot: int
    fresh_n_eff: float


@dataclass(frozen=True)
class Config:
    seed: int
    data: Data
    periods: Periods
    condition: Condition
    memory: Memory
    static: Static
    evaluate: Evaluate


def _build(tp: typing.Any, data: typing.Any, path: str) -> typing.Any:
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        hints = typing.get_type_hints(tp)
        unknown = set(data) - set(hints)
        if unknown:
            raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name not in data:
                raise ConfigError(f"{path}.{f.name}: missing")
            kwargs[f.name] = _build(hints[f.name], data[f.name], f"{path}.{f.name}")
        return tp(**kwargs)
    if typing.get_origin(tp) is list:
        (arg,) = typing.get_args(tp)
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list")
        return [_build(arg, x, f"{path}[{i}]") for i, x in enumerate(data)]
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ConfigError(f"{path}: expected a number")
        return float(data)
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ConfigError(f"{path}: expected an integer")
        return data
    if not isinstance(data, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}")
    return data


def validate(cfg: Config) -> None:
    p = cfg.periods
    for name, span in (("train_days", p.train_days), ("val_days", p.val_days), ("test_days", p.test_days)):
        if len(span) != 2 or span[0] >= span[1]:
            raise ConfigError(f"periods.{name}: need [start, end) with start < end")
    if not (p.train_days[1] <= p.val_days[0] and p.val_days[1] <= p.test_days[0]):
        raise ConfigError("periods must be ordered train < val < test without overlap")
    c = cfg.condition
    if not 0 < c.slow_ratio < 1 or not 0.5 <= c.free_flow_quantile < 1:
        raise ConfigError("condition: slow_ratio in (0,1) and free_flow_quantile in [0.5,1)")
    if cfg.memory.half_life_s <= 0 or cfg.memory.prior_weight <= 0:
        raise ConfigError("memory: half_life_s and prior_weight must be positive")
    if not all(0 < f <= 1 for f in cfg.evaluate.fleet_fractions):
        raise ConfigError("evaluate.fleet_fractions must be in (0,1]")
    if 24 % cfg.static.hour_block != 0:
        raise ConfigError("static.hour_block must divide 24")


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    with open(path, encoding="utf-8") as fh:
        cfg = _build(Config, yaml.safe_load(fh), "config")
    validate(cfg)
    return cfg
