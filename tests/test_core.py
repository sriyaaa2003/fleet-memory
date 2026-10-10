import numpy as np
import pandas as pd
import pytest

from fleetmem.config import ConfigError, load_config, validate
from fleetmem.memory import DecayMemory
from fleetmem.replay import StaticPrior, bucket, condition_table, replay
from fleetmem.ved import cell_key, heading_bin

import dataclasses
import polars as pl


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


def frame(rows):
    """Hand-built passes (unit-test inputs, not study data): key, veh, t0, t1, v_mean."""
    df = pd.DataFrame(rows, columns=["key", "veh", "t0", "t1", "v_mean"])
    df["day"] = df.t0 / 86400.0
    return df


def test_heading_bins_split_the_compass_into_four_directions():
    # north, east, south, west as (dlat, dlon)
    b = heading_bin(np.array([1.0, 0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0, -1.0]), 4)
    assert b.tolist() == [0, 1, 2, 3]


def test_cell_key_is_unique_per_cell_and_heading():
    k = pl.DataFrame({"a": [10, 10, 11], "b": [5, 5, 5], "h": [0, 1, 0]}).select(cell_key(pl.col("a"), pl.col("b"), pl.col("h")))["a"].to_list()
    assert len(set(k)) == 3


def test_memory_halves_evidence_every_half_life_and_returns_to_the_prior():
    m = DecayMemory(half_life_s=600.0, prior_weight=1.0)
    m.update(1, key=5, t=0.0, slow=True)
    fresh = m.query(5, 0.0, prior=0.1)
    assert fresh.p == pytest.approx((1 + 0.1) / 2) and fresh.n_eff == pytest.approx(1.0)
    later = m.query(5, 600.0, prior=0.1)
    assert later.n_eff == pytest.approx(0.5)
    assert later.p == pytest.approx((0.5 + 0.1) / 1.5)
    assert m.query(5, 6000.0, prior=0.1).p == pytest.approx(0.1, abs=0.01)


def test_memory_ignores_duplicate_pass_ids_and_unseen_cells():
    m = DecayMemory(600.0, 1.0)
    assert m.update(7, 1, 0.0, True) and not m.update(7, 1, 1.0, True)
    s = m.query(99, 10.0, prior=0.3)
    assert s.p == pytest.approx(0.3) and s.n_eff == 0 and s.last_slow is None


def test_late_updates_do_not_move_the_last_pass_backwards():
    m = DecayMemory(600.0, 1.0)
    m.update(1, 1, 100.0, True)
    m.update(2, 1, 50.0, False)         # arrives late
    assert m.query(1, 110.0, 0.1).last_slow == 1.0


def test_condition_table_labels_slow_passes_against_free_flow_from_training_days(cfg):
    rows = [[1, v, d * 86400.0 + 100, d * 86400.0 + 130, 50.0] for d in range(0, 120, 1) for v in (1,)]    # 120 free-flow training passes
    rows += [[1, 2, 200 * 86400.0 + 100, 200 * 86400.0 + 140, 20.0], [1, 3, 300 * 86400.0 + 100, 300 * 86400.0 + 140, 48.0]]
    t = condition_table(frame(rows), cfg)
    val, test = t[t.period == "val"], t[t.period == "test"]
    assert val.slow.tolist() == [True] and test.slow.tolist() == [False]          # 20 < 0.6 x 50; 48 is normal
    assert set(t.period) == {"train", "val", "test"}


def test_buckets_separate_weekdays_from_weekends_and_hours():
    day = np.array([0.5, 0.5, 3.5, 3.5])        # day 0 is a Wednesday: day 3 is Saturday
    t = np.array([8 * 3600.0, 20 * 3600.0, 8 * 3600.0, 20 * 3600.0])
    b = bucket(day, t, 3)
    assert len(set(b.tolist())) == 4 and b[0] < 8 <= b[2]


def test_static_prior_follows_the_cells_own_rush_hour(cfg):
    rows = []
    for d in range(40):                          # a cell that is slow every weekday morning at 8 and fine in the evening
        base = d * 86400.0
        rows += [[1, 1, base + 8 * 3600, base + 8 * 3600 + 30, 15.0], [1, 2, base + 20 * 3600, base + 20 * 3600 + 30, 50.0]]
    df = frame(rows)
    df["slow"] = df.v_mean < 30
    df["period"] = "train"
    prior = StaticPrior(df, cfg)
    p = prior.predict(np.array([1, 1]), np.array([5.0, 5.0]), np.array([8 * 3600.0 + 5, 20 * 3600.0 + 5]))
    assert p[0] > 0.6 > 0.3 > p[1]


def test_replay_shows_a_pass_only_what_ended_before_it_entered(cfg):
    base = 250 * 86400.0
    df = frame([[1, 1, base, base + 100, 10.0],              # slow pass A, ends at +100
                [1, 2, base + 50, base + 150, 50.0],         # enters before A has ended: must not see A
                [1, 3, base + 200, base + 260, 50.0]])       # enters after both ended: sees both, the later one is the last
    df["ff"], df["slow"], df["period"] = 50.0, [True, False, False], "test"
    F = replay(df, np.full(3, 0.1), cfg)
    assert F.n_eff.tolist()[1] == 0.0 and F.last_slow.isna().tolist()[1]
    assert F.n_eff.tolist()[2] > 1.7 and F.last_slow.tolist()[2] == 0.0
    assert F.n_eff.tolist()[0] == 0.0                         # a pass never sees itself


def test_replay_only_learns_from_the_fed_vehicles(cfg):
    base = 250 * 86400.0
    df = frame([[1, 1, base, base + 10, 10.0], [1, 2, base + 100, base + 110, 50.0]])
    df["slow"], df["period"] = [True, False], "test"
    F = replay(df, np.full(2, 0.1), cfg, feed=np.array([False, True]))
    assert F.n_eff.tolist()[1] == 0.0


def test_config_rejects_overlapping_periods(cfg):
    bad = dataclasses.replace(cfg, periods=dataclasses.replace(cfg.periods, val_days=[100, 240]))
    with pytest.raises(ConfigError):
        validate(bad)
