from pathlib import Path

import pytest
import yaml

from fleetmem.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, mutate):
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_default_loads(cfg):
    assert cfg.sim.n_segments == 40
    assert set(cfg.sim.kinds) == {"low_traction", "obstruction"}


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, lambda r: r["memory"].update(bogus=1)))


def test_missing_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write(tmp_path, lambda r: r["warn"].pop("p_threshold")))


def test_invalid_value_rejected(tmp_path):
    with pytest.raises(ConfigError, match="prior_p"):
        load_config(_write(tmp_path, lambda r: r["memory"].update(prior_p=1.5)))


def test_train_seed_must_not_be_test_seed(tmp_path):
    with pytest.raises(ConfigError, match="train seed"):
        load_config(_write(tmp_path, lambda r: r["eval"].update(train_seed=1)))
