"""Evidence-grounded recommendations.

A recommendation is built only from records in the evidence ledger. Every claim is structured
(`kind` + numeric `params` + cited `evidence_ids`); its text is rendered from those params, and
`verify` recomputes the params from the cited ledger records. The recommender abstains when the
evidence is thin, stale or conflicting. An LLM phraser may later replace the template renderer, but
its output would have to pass the same `verify` gate (see docs/BLUEPRINT.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import RecommendConfig, WarnConfig
from .memory import DecayMemory

_TOL = 1e-9


@dataclass(frozen=True)
class EvidenceRecord:
    obs_id: str
    vehicle_id: str
    t: float
    segment: int
    llr: float
    can_llr: float | None = None
    cam_llr: float | None = None
    slip_ratio: float | None = None
    decel_mps2: float | None = None
    frame_ref: str | None = None


class EvidenceLedger:
    def __init__(self, max_per_segment: int):
        self._max = max_per_segment
        self._by_seg: dict[int, deque[EvidenceRecord]] = {}

    def add(self, rec: EvidenceRecord) -> None:
        self._by_seg.setdefault(rec.segment, deque(maxlen=self._max)).append(rec)

    def window(self, segment: int, t: float, window_s: float) -> list[EvidenceRecord]:
        return [r for r in self._by_seg.get(segment, ()) if t - window_s <= r.t <= t]

    def get(self, segment: int, obs_id: str) -> EvidenceRecord | None:
        return next((r for r in self._by_seg.get(segment, ()) if r.obs_id == obs_id), None)


@dataclass(frozen=True)
class Claim:
    kind: str
    params: dict[str, Any]
    evidence_ids: tuple[str, ...]

    def text(self) -> str:
        p = self.params
        if self.kind == "vehicle_reports":
            return (
                f"{p['n_vehicles']} vehicle(s) ({', '.join(p['vehicles'])}) reported anomalous "
                f"observations between t={p['t_first']:.0f}s and t={p['t_last']:.0f}s."
            )
        if self.kind == "can_signal":
            return (
                f"CAN evidence: mean slip ratio {p['mean_slip']:.3f}, "
                f"peak deceleration {p['max_decel']:.2f} m/s^2 across cited passes."
            )
        if self.kind == "camera_frames":
            return "Camera evidence in frames: " + ", ".join(p["frames"]) + "."
        raise ValueError(f"unknown claim kind {self.kind}")


@dataclass(frozen=True)
class Recommendation:
    kind: str  # "warn" | "abstain"
    segment: int
    t: float
    p: float
    cause: str | None = None
    claims: tuple[Claim, ...] = ()
    actions: tuple[str, ...] = ()
    reason: str | None = None  # abstention reason


class Recommender:
    def __init__(
        self, cfg: RecommendConfig, warn: WarnConfig, memory: DecayMemory, ledger: EvidenceLedger
    ):
        self._cfg, self._warn, self._memory, self._ledger = cfg, warn, memory, ledger

    def _abstain(self, segment: int, t: float, p: float, reason: str) -> Recommendation:
        return Recommendation("abstain", segment, t, p, reason=reason)

    def __call__(self, segment: int, t: float) -> Recommendation:
        cfg = self._cfg
        st = self._memory.predict(segment, t)
        if st.p < self._warn.p_threshold or st.n_eff < self._warn.min_n_eff:
            return self._abstain(segment, t, st.p, "below_threshold")
        recs = self._ledger.window(segment, t, cfg.window_s)
        pos = [r for r in recs if r.llr > 0]
        neg_mass = sum(-r.llr for r in recs if r.llr < 0)
        pos_mass = sum(r.llr for r in pos)
        if len({r.vehicle_id for r in pos}) < cfg.min_vehicles:
            return self._abstain(segment, t, st.p, "insufficient_vehicles")
        if t - max(r.t for r in pos) > cfg.max_age_s:
            return self._abstain(segment, t, st.p, "stale_evidence")
        if neg_mass / (neg_mass + pos_mass) > cfg.conflict_ratio:
            return self._abstain(segment, t, st.p, "conflicting_evidence")

        cited = sorted(pos, key=lambda r: r.llr, reverse=True)[: cfg.max_cited]
        cited.sort(key=lambda r: r.t)
        cause = self._cause(cited)
        claims = [self._vehicle_claim(cited)]
        can_rows = [r for r in cited if r.slip_ratio is not None and r.decel_mps2 is not None]
        if can_rows:
            claims.append(self._can_claim(can_rows))
        cam_rows = [r for r in cited if r.frame_ref is not None and (r.cam_llr or 0.0) > 0]
        if cam_rows:
            claims.append(self._cam_claim(cam_rows[: cfg.max_frames]))
        actions = tuple(a.format(segment=segment) for a in cfg.policy[cause])
        return Recommendation("warn", segment, t, st.p, cause, tuple(claims), actions)

    def _cause(self, cited: list[EvidenceRecord]) -> str:
        can = sum(max(r.can_llr or 0.0, 0.0) for r in cited)
        cam = sum(max(r.cam_llr or 0.0, 0.0) for r in cited)
        dom = self._cfg.dominance
        if can > 0 and can >= dom * cam:
            return "reduced_traction"
        if cam > 0 and cam >= dom * can:
            return "visual_obstruction"
        return "unspecified_condition_change"

    @staticmethod
    def _vehicle_params(rows: list[EvidenceRecord]) -> dict[str, Any]:
        vehicles = sorted({r.vehicle_id for r in rows})
        return {
            "n_vehicles": len(vehicles),
            "vehicles": vehicles,
            "t_first": min(r.t for r in rows),
            "t_last": max(r.t for r in rows),
        }

    def _vehicle_claim(self, rows: list[EvidenceRecord]) -> Claim:
        return Claim("vehicle_reports", self._vehicle_params(rows), tuple(r.obs_id for r in rows))

    @staticmethod
    def _can_params(rows: list[EvidenceRecord]) -> dict[str, Any]:
        return {
            "mean_slip": sum(r.slip_ratio for r in rows) / len(rows),
            "max_decel": max(r.decel_mps2 for r in rows),
        }

    def _can_claim(self, rows: list[EvidenceRecord]) -> Claim:
        return Claim("can_signal", self._can_params(rows), tuple(r.obs_id for r in rows))

    @staticmethod
    def _cam_claim(rows: list[EvidenceRecord]) -> Claim:
        return Claim(
            "camera_frames", {"frames": [r.frame_ref for r in rows]}, tuple(r.obs_id for r in rows)
        )


def verify(rec: Recommendation, ledger: EvidenceLedger) -> bool:
    """True iff every claim's cited records exist in the ledger and reproduce the claim's params.

    An abstention makes no claims and is trivially valid. A 'warn' without claims is invalid.
    """
    if rec.kind == "abstain":
        return not rec.claims
    if not rec.claims:
        return False
    for claim in rec.claims:
        rows = [ledger.get(rec.segment, i) for i in claim.evidence_ids]
        if not rows or any(r is None for r in rows):
            return False
        if claim.kind == "vehicle_reports":
            want = Recommender._vehicle_params(rows)
            ok = (
                claim.params["vehicles"] == want["vehicles"]
                and claim.params["n_vehicles"] == want["n_vehicles"]
                and abs(claim.params["t_first"] - want["t_first"]) < _TOL
                and abs(claim.params["t_last"] - want["t_last"]) < _TOL
            )
        elif claim.kind == "can_signal":
            if any(r.slip_ratio is None or r.decel_mps2 is None for r in rows):
                return False
            want = Recommender._can_params(rows)
            ok = all(abs(claim.params[k] - want[k]) < _TOL for k in want)
        elif claim.kind == "camera_frames":
            ok = claim.params["frames"] == [r.frame_ref for r in rows]
        else:
            return False
        if not ok:
            return False
    return True
