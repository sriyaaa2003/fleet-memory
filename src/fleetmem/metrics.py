"""Evaluation metrics over replayed passes.

Definitions (all computed from pass records, never from the detector's internals):
- precision / recall / f1 / fpr: per pass. A pass is positive if a ground-truth event is active on
  its segment. Recall is bounded below 1 because the first affected vehicle has nobody to warn it.
- event_recall: share of events (with at least one pass during them) that triggered a warning on
  at least one pass during the event.
- detection_delay_s: onset of an event to the first warned pass inside it (detected events only).
- lead_time_s: for each warned true-positive pass, the time between the state first crossing the
  warning threshold and that pass, i.e. how long the warning was available before the vehicle came.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from .replay import PassRecord
from .schema import Event


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def summarize(passes: list[PassRecord], events: list[Event]) -> dict[str, float | int | None]:
    tp = sum(p.warn and p.truth for p in passes)
    fp = sum(p.warn and not p.truth for p in passes)
    fn = sum(not p.warn and p.truth for p in passes)
    tn = sum(not p.warn and not p.truth for p in passes)

    delays: list[float] = []
    observed = detected = 0
    by_event: dict[int, list[PassRecord]] = {}
    for p in passes:
        if p.event_id is not None:
            by_event.setdefault(p.event_id, []).append(p)
    for e in events:
        rows = by_event.get(e.event_id)
        if not rows:
            continue
        observed += 1
        hits = [r.t for r in rows if r.warn]
        if hits:
            detected += 1
            delays.append(min(hits) - e.onset)

    leads = [p.t - p.warn_since for p in passes if p.warn and p.truth and p.warn_since is not None]
    emitted = [p for p in passes if p.rec_kind == "warn"]
    warned_with_rec = [p for p in passes if p.rec_kind is not None]
    return {
        "passes": len(passes),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": _ratio(fp, fp + tn),
        "event_recall": _ratio(detected, observed),
        "detection_delay_s": statistics.median(delays) if delays else None,
        "lead_time_s": statistics.median(leads) if leads else None,
        "rec_emit_rate": _ratio(len(emitted), len(warned_with_rec)),
        "rec_precision": _ratio(sum(p.truth for p in emitted), len(emitted)),
        "rec_verified": _ratio(sum(bool(p.rec_verified) for p in emitted), len(emitted)),
    }


def aggregate(
    runs: Iterable[dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Mean and std across seeds per metric, skipping undefined (None) values."""
    runs = list(runs)
    out: dict[str, dict[str, float | int | None]] = {}
    for key in runs[0]:
        vals = [r[key] for r in runs if r[key] is not None]
        out[key] = {
            "mean": statistics.fmean(vals) if vals else None,
            "std": statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None),
            "n": len(vals),
        }
    return out
