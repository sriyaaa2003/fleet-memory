from dataclasses import replace

import pytest

from fleetmem.memory import DecayMemory
from fleetmem.recommend import Claim, EvidenceLedger, EvidenceRecord, Recommender, verify

SEG = 7


def _setup(cfg, rows):
    memory = DecayMemory(cfg.memory)
    ledger = EvidenceLedger(cfg.memory.ledger_max_per_segment)
    for r in rows:
        ledger.add(r)
        memory.update(r.obs_id, r.segment, r.t, r.llr)
    return memory, ledger, Recommender(cfg.recommend, cfg.warn, memory, ledger)


def _rec(i, vehicle, t, llr=4.0, can=None, cam=None, frame=True):
    return EvidenceRecord(
        obs_id=f"o{i}", vehicle_id=vehicle, t=t, segment=SEG, llr=llr, can_llr=can, cam_llr=cam,
        slip_ratio=0.12 if can is not None else None,
        decel_mps2=1.4 if can is not None else None,
        frame_ref=f"frame{i}" if frame and cam is not None else None,
    )


def test_abstains_without_any_evidence(cfg):
    _, _, rec = _setup(cfg, [])
    out = rec(SEG, 100.0)
    assert out.kind == "abstain" and out.reason == "below_threshold" and not out.claims


def test_abstains_with_a_single_vehicle(cfg):
    rows = [_rec(i, "v1", 10.0 * i, can=3.0) for i in range(3)]
    _, _, rec = _setup(cfg, rows)
    out = rec(SEG, 40.0)
    assert out.kind == "abstain" and out.reason == "insufficient_vehicles"


def test_abstains_on_stale_evidence(cfg):
    rows = [_rec(0, "v1", 0.0, can=3.0), _rec(1, "v2", 1.0, can=3.0)]
    _, _, rec = _setup(cfg, rows)
    t = cfg.recommend.max_age_s + 100.0
    # keep the state above threshold so the staleness rule, not the threshold rule, fires
    cfg2 = replace(cfg, memory=replace(cfg.memory, half_life_s=1e9))
    _, _, rec2 = _setup(cfg2, rows)
    assert rec2(SEG, t).reason == "stale_evidence"
    assert rec(SEG, t).kind == "abstain"


def test_abstains_on_conflicting_evidence(cfg):
    # near-balanced evidence: net state stays above the (lowered) threshold but the negative
    # share of evidence mass (12 / 23.7) exceeds conflict_ratio
    rows = [_rec(i, f"v{i}", float(i), llr=3.9, can=3.0) for i in range(3)]
    rows += [_rec(3 + i, "v4", 3.0 + i, llr=-4.0, can=-3.0) for i in range(3)]
    cfg2 = replace(cfg, warn=replace(cfg.warn, p_threshold=0.01))
    _, _, rec = _setup(cfg2, rows)
    assert rec(SEG, 10.0).reason == "conflicting_evidence"


def test_warns_with_cited_verified_evidence(cfg):
    rows = [_rec(i, f"v{i}", 5.0 * i, can=3.0) for i in range(3)]
    _, ledger, rec = _setup(cfg, rows)
    out = rec(SEG, 20.0)
    assert out.kind == "warn" and out.cause == "reduced_traction"
    assert verify(out, ledger)
    cited = {i for c in out.claims for i in c.evidence_ids}
    assert cited <= {r.obs_id for r in rows}
    assert "segment 7" in out.actions[0]


def test_camera_dominated_cause_cites_frames(cfg):
    rows = [_rec(i, f"v{i}", 5.0 * i, cam=3.0) for i in range(3)]
    _, ledger, rec = _setup(cfg, rows)
    out = rec(SEG, 20.0)
    assert out.cause == "visual_obstruction"
    cam = [c for c in out.claims if c.kind == "camera_frames"]
    assert cam and all(f.startswith("frame") for f in cam[0].params["frames"])
    assert verify(out, ledger)


def test_mixed_evidence_is_unspecified_cause(cfg):
    rows = [_rec(i, f"v{i}", 5.0 * i, can=2.0, cam=2.0) for i in range(3)]
    _, _, rec = _setup(cfg, rows)
    assert rec(SEG, 20.0).cause == "unspecified_condition_change"


def test_verify_rejects_tampered_claim(cfg):
    rows = [_rec(i, f"v{i}", 5.0 * i, can=3.0) for i in range(3)]
    _, ledger, rec = _setup(cfg, rows)
    out = rec(SEG, 20.0)
    bad = tuple(
        Claim(c.kind, {**c.params, "mean_slip": 0.99} if c.kind == "can_signal" else c.params,
              c.evidence_ids)
        for c in out.claims
    )
    assert not verify(replace(out, claims=bad), ledger)


def test_verify_rejects_unknown_evidence_id(cfg):
    rows = [_rec(i, f"v{i}", 5.0 * i, can=3.0) for i in range(3)]
    _, ledger, rec = _setup(cfg, rows)
    out = rec(SEG, 20.0)
    ghost = Claim("vehicle_reports", out.claims[0].params, ("does-not-exist",))
    assert not verify(replace(out, claims=(ghost,)), ledger)


def test_warn_without_claims_is_invalid(cfg):
    _, ledger, rec = _setup(cfg, [_rec(i, f"v{i}", 5.0 * i, can=3.0) for i in range(3)])
    out = rec(SEG, 20.0)
    assert not verify(replace(out, claims=()), ledger)


def test_ledger_is_bounded():
    ledger = EvidenceLedger(3)
    for i in range(10):
        ledger.add(_rec(i, "v", float(i), can=1.0))
    assert len(ledger.window(SEG, 100.0, 1000.0)) == 3


@pytest.mark.parametrize("kind", ["warn", "abstain"])
def test_abstain_has_no_claims_and_is_valid(cfg, kind):
    _, ledger, rec = _setup(cfg, [])
    out = rec(SEG, 1.0)
    assert out.kind == "abstain" and verify(out, ledger)
