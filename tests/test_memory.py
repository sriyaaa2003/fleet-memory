import math

from fleetmem.memory import DecayMemory, TimestampMemory


def test_prior_when_empty(cfg):
    st = DecayMemory(cfg.memory).predict(0, 100.0)
    assert math.isclose(st.p, cfg.memory.prior_p)
    assert st.n_eff == 0.0


def test_evidence_raises_then_decays_to_prior(cfg):
    m = DecayMemory(cfg.memory)
    m.update("a", 3, 0.0, 3.0)
    now = m.predict(3, 0.0)
    later = m.predict(3, cfg.memory.half_life_s)
    far = m.predict(3, cfg.memory.half_life_s * 40)
    assert now.p > later.p > cfg.memory.prior_p
    assert math.isclose(far.p, cfg.memory.prior_p, abs_tol=1e-6)
    assert math.isclose(later.n_eff, 0.5)


def test_negative_evidence_lowers_state(cfg):
    m = DecayMemory(cfg.memory)
    m.update("a", 1, 0.0, 3.0)
    p_before = m.predict(1, 0.0).p
    m.update("b", 1, 0.0, -3.0)
    assert m.predict(1, 0.0).p < p_before


def test_llr_is_clipped(cfg):
    m = DecayMemory(cfg.memory)
    m.update("a", 1, 0.0, 1000.0)
    n = DecayMemory(cfg.memory)
    n.update("a", 1, 0.0, cfg.memory.llr_clip)
    assert math.isclose(m.predict(1, 0.0).p, n.predict(1, 0.0).p)


def test_duplicate_observation_is_idempotent(cfg):
    m = DecayMemory(cfg.memory)
    assert m.update("a", 1, 0.0, 2.0) is True
    p = m.predict(1, 0.0).p
    assert m.update("a", 1, 0.0, 2.0) is False
    assert m.predict(1, 0.0).p == p


def test_none_llr_does_not_change_state(cfg):
    m = DecayMemory(cfg.memory)
    assert m.update("a", 1, 0.0, None) is False
    assert m.predict(1, 0.0).n_eff == 0.0


def test_segments_are_independent(cfg):
    m = DecayMemory(cfg.memory)
    m.update("a", 1, 0.0, 3.0)
    assert math.isclose(m.predict(2, 0.0).p, cfg.memory.prior_p)


def test_late_update_does_not_move_time_backwards(cfg):
    m = DecayMemory(cfg.memory)
    m.update("a", 1, 100.0, 2.0)
    m.update("b", 1, 50.0, 2.0)  # late arrival
    assert m.predict(1, 100.0).p > m.predict(1, 100.0 + cfg.memory.half_life_s).p


def test_timestamp_memory_hold_window(cfg):
    m = TimestampMemory(cfg.timestamp)
    m.update("a", 5, 10.0, 1.0)
    assert m.predict(5, 10.0 + cfg.timestamp.hold_s).p == 1.0
    assert m.predict(5, 10.0 + cfg.timestamp.hold_s + 1).p == 0.0
    m.update("b", 6, 10.0, 0.0)  # no trigger
    assert m.predict(6, 11.0).p == 0.0
