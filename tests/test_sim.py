import numpy as np

from fleetmem.sim import generate


def test_deterministic(cfg):
    a, b = generate(cfg.sim, 5), generate(cfg.sim, 5)
    assert [o.obs_id for o in a.observations] == [o.obs_id for o in b.observations]
    assert np.allclose(a.observations[10].cam, b.observations[10].cam)
    assert a.events == b.events


def test_different_seed_differs(cfg):
    assert generate(cfg.sim, 5).events != generate(cfg.sim, 6).events


def test_sorted_and_labels_match_events(cfg):
    r = generate(cfg.sim, 3)
    ts = [o.t for o in r.observations]
    assert ts == sorted(ts)
    by_id = {e.event_id: e for e in r.events}
    for o in r.observations:
        eid = r.labels[o.obs_id]
        if eid is not None:
            assert by_id[eid].active(o.segment, o.t)
    assert any(v is not None for v in r.labels.values())


def test_events_do_not_overlap_per_segment(cfg):
    ev = generate(cfg.sim, 4).events
    for i, a in enumerate(ev):
        for b in ev[i + 1 :]:
            if a.segment == b.segment:
                assert a.end <= b.onset or b.end <= a.onset


def test_road_shared_across_seeds(cfg):
    """Same road_seed means the same segment appearance: normal passes cluster per segment."""
    a, b = generate(cfg.sim, 1), generate(cfg.sim, 2)

    def mean_seg0(r):
        rows = [o.cam for o in r.observations if o.segment == 0 and r.labels[o.obs_id] is None]
        return np.mean(rows, axis=0)

    assert np.linalg.norm(mean_seg0(a) - mean_seg0(b)) < 0.5
