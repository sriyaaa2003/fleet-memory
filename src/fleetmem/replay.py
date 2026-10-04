"""Online replay of a fleet's observations in time order.

For each pass the memory is queried *before* the pass's own observation is folded in, so a warning
only ever uses information from earlier vehicles (no look-ahead, no self-confirmation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import WarnConfig
from .recommend import EvidenceLedger, EvidenceRecord, Recommendation, verify
from .schema import CAN_DECEL, CAN_SLIP, Observation


@dataclass(frozen=True)
class PassRecord:
    obs_id: str
    t: float
    segment: int
    warn: bool
    truth: bool
    event_id: int | None
    p: float
    warn_since: float | None
    rec_kind: str | None = None  # "warn" | "abstain" for warned passes when a recommender runs
    rec_verified: bool | None = None


def replay(
    obs: list[Observation],
    llrs: list[float | None],
    contribs: list[dict[str, float]],
    labels: dict[str, int | None],
    memory,
    warn_cfg: WarnConfig,
    ledger: EvidenceLedger | None = None,
    recommender: Callable[[int, float], Recommendation] | None = None,
) -> list[PassRecord]:
    if any(b.t < a.t for a, b in zip(obs, obs[1:], strict=False)):
        raise ValueError("observations must be sorted by time")
    if not (len(obs) == len(llrs) == len(contribs)):
        raise ValueError("observations, llrs and contribs must align")

    def warned(state) -> bool:
        return state.p >= warn_cfg.p_threshold and state.n_eff >= warn_cfg.min_n_eff

    since: dict[int, float] = {}
    out: list[PassRecord] = []
    for o, llr, contrib in zip(obs, llrs, contribs, strict=True):
        state = memory.predict(o.segment, o.t)
        warn = warned(state)
        if not warn:
            since.pop(o.segment, None)
        rec_kind = rec_ok = None
        if warn and recommender is not None:
            rec = recommender(o.segment, o.t)
            rec_kind, rec_ok = rec.kind, (verify(rec, ledger) if ledger is not None else None)
        event_id = labels.get(o.obs_id)
        out.append(
            PassRecord(
                o.obs_id, o.t, o.segment, warn, event_id is not None, event_id, state.p,
                since.get(o.segment) if warn else None, rec_kind, rec_ok,
            )
        )
        memory.update(o.obs_id, o.segment, o.t, llr)
        if llr is not None and ledger is not None:
            ledger.add(
                EvidenceRecord(
                    obs_id=o.obs_id,
                    vehicle_id=o.vehicle_id,
                    t=o.t,
                    segment=o.segment,
                    llr=llr,
                    can_llr=contrib.get("can"),
                    cam_llr=contrib.get("cam"),
                    slip_ratio=o.can[CAN_SLIP] if o.can else None,
                    decel_mps2=o.can[CAN_DECEL] if o.can else None,
                    frame_ref=o.frame_ref,
                )
            )
        if warned(memory.predict(o.segment, o.t)):
            since.setdefault(o.segment, o.t)
    return out
