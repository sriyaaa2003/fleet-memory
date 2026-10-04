from dataclasses import replace
from pathlib import Path

import pytest

from fleetmem.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def small_cfg(cfg):
    """Shorter run for fast end-to-end tests."""
    sim = replace(cfg.sim, duration_s=3600.0, n_events=8)
    ev = replace(cfg.eval, test_seeds=[1, 2], dropout_levels=[0.0, 0.5, 1.0])
    return replace(cfg, sim=sim, eval=ev)
