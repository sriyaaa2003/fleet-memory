import pytest

from fleetmem.memory import DecayMemory
from fleetmem.metrics import aggregate, summarize
from fleetmem.replay import PassRecord, replay
from fleetmem.schema import Event, Observation


def _obs(i, t, seg=0, vehicle="v0"):
    return Observation(obs_id=f"o{i}", vehicle_id=vehicle, t=t, segment=seg)


def test_no_self_confirmation(cfg):
    """The first pass is judged on empty memory even if its own evidence is huge."""
    obs = [_obs(0, 0.0)]
    out = replay(obs, [4.0], [{}], {"o0": 1}, DecayMemory(cfg.memory), cfg.warn)
    assert out[0].warn is False


def test_later_vehicle_is_warned_by_earlier_ones(cfg):
    obs = [_obs(0, 0.0, vehicle="a"), _obs(1, 10.0, vehicle="b"), _obs(2, 20.0, vehicle="c")]
    labels = {"o0": 1, "o1": 1, "o2": 1}
    out = replay(obs, [4.0] * 3, [{}] * 3, labels, DecayMemory(cfg.memory), cfg.warn)
    assert [p.warn for p in out] == [False, False, True]
    assert out[2].warn_since == 10.0


def test_unsorted_input_rejected(cfg):
    with pytest.raises(ValueError, match="sorted"):
        replay(
            [_obs(0, 5.0), _obs(1, 1.0)],
            [1.0, 1.0],
            [{}, {}],
            {},
            DecayMemory(cfg.memory),
            cfg.warn,
        )


def test_misaligned_inputs_rejected(cfg):
    with pytest.raises(ValueError, match="align"):
        replay([_obs(0, 0.0)], [], [], {}, DecayMemory(cfg.memory), cfg.warn)


def _p(i, t, warn, truth, event_id=None, since=None):
    return PassRecord(f"p{i}", t, 0, warn, truth, event_id, 0.0, since)


def test_summarize_hand_computed():
    ev = [Event(0, 0, 100.0, 500.0, "k"), Event(1, 1, 0.0, 50.0, "k")]
    passes = [
        _p(0, 120.0, False, True, 0),  # FN, first pass of event 0
        _p(1, 160.0, True, True, 0, since=150.0),  # TP, lead 10 s
        _p(2, 200.0, True, False),  # FP
        _p(3, 300.0, False, False),  # TN
        _p(4, 400.0, False, False),  # TN
    ]
    s = summarize(passes, ev)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert s["fpr"] == pytest.approx(1 / 3)
    assert s["event_recall"] == 1.0  # event 1 had no passes: not counted as observed
    assert s["detection_delay_s"] == 60.0
    assert s["lead_time_s"] == 10.0


def test_undefined_metrics_are_none_not_zero():
    s = summarize([_p(0, 1.0, False, False)], [])
    assert s["precision"] is None and s["recall"] is None and s["lead_time_s"] is None


def test_aggregate_skips_none():
    agg = aggregate([{"x": 1.0, "y": None}, {"x": 3.0, "y": None}])
    assert agg["x"]["mean"] == 2.0 and agg["x"]["n"] == 2
    assert agg["y"]["mean"] is None
