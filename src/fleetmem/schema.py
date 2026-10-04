"""Core data types shared by every component."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CAN_SLIP = "slip_ratio"
CAN_DECEL = "decel_mps2"


@dataclass(frozen=True, eq=False)
class Observation:
    """One vehicle passing one road segment. A missing modality is None, never a fake value."""

    obs_id: str
    vehicle_id: str
    t: float
    segment: int
    can: dict[str, float] | None = None
    cam: np.ndarray | None = None
    frame_ref: str | None = None

    def drop(self, can: bool = False, cam: bool = False) -> Observation:
        return replace(
            self,
            can=None if can else self.can,
            cam=None if cam else self.cam,
            frame_ref=None if cam else self.frame_ref,
        )


@dataclass(frozen=True)
class Event:
    """Ground-truth condition change (simulation only)."""

    event_id: int
    segment: int
    onset: float
    end: float
    kind: str

    def active(self, segment: int, t: float) -> bool:
        return segment == self.segment and self.onset <= t < self.end


@dataclass(frozen=True)
class SimResult:
    observations: list[Observation]
    labels: dict[str, int | None]  # obs_id -> event_id active at that pass, or None
    events: list[Event]
