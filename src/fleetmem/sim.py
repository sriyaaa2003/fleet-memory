"""Synthetic fleet simulator.

This produces *simulated* multi-vehicle drives with injected condition changes so the pipeline can
be developed and evaluated under controlled ground truth. It is not real vehicle data, and results
obtained on it must never be presented as real-world performance. Real data enters through the
same `Observation` type (see docs/BLUEPRINT.md, "Real-data adapters").
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .schema import CAN_DECEL, CAN_SLIP, Event, Observation, SimResult


def _sample_events(cfg: SimConfig, rng: np.random.Generator, kinds: list[str]) -> list[Event]:
    weights = np.array([cfg.kinds[k].weight for k in kinds], dtype=float)
    weights /= weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < cfg.n_events and attempts < 2000:
        attempts += 1
        seg = int(rng.integers(cfg.n_segments))
        length = float(rng.uniform(cfg.event_min_s, cfg.event_max_s))
        onset = float(rng.uniform(0.0, max(cfg.duration_s - cfg.event_min_s, 1.0)))
        end = min(onset + length, cfg.duration_s)
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        if any(e.segment == seg and onset < e.end and e.onset < end for e in events):
            continue
        events.append(Event(len(events), seg, onset, end, kind))
    return events


def generate(cfg: SimConfig, run_seed: int) -> SimResult:
    road = np.random.default_rng(cfg.road_seed)
    appearance = road.normal(0.0, cfg.appearance_std, (cfg.n_segments, cfg.cam_dim))
    kinds = sorted(cfg.kinds)
    shift_dir: dict[str, np.ndarray] = {}
    for k in kinds:
        v = road.normal(size=cfg.cam_dim)
        shift_dir[k] = v / np.linalg.norm(v)

    rng = np.random.default_rng(run_seed)
    events = _sample_events(cfg, rng, kinds)
    by_segment: dict[int, list[Event]] = {}
    for e in events:
        by_segment.setdefault(e.segment, []).append(e)
    bias = rng.normal(0.0, cfg.vehicle_slip_bias_sd, cfg.n_vehicles)

    lap = cfg.n_segments * cfg.seg_time_s + cfg.rest_s
    observations: list[Observation] = []
    labels: dict[str, int | None] = {}
    for v in range(cfg.n_vehicles):
        offset = v * lap / cfg.n_vehicles
        k = 0
        while offset + k * lap < cfg.duration_s:
            base = offset + k * lap
            for s in range(cfg.n_segments):
                t = base + s * cfg.seg_time_s + float(rng.normal(0.0, cfg.time_jitter_s))
                if not 0.0 <= t < cfg.duration_s:
                    continue
                active = next((e for e in by_segment.get(s, []) if e.active(s, t)), None)
                affected = active is not None and rng.random() < cfg.effect_prob
                slip_shift = decel_shift = cam_shift = 0.0
                direction = None
                if affected:
                    kc = cfg.kinds[active.kind]
                    slip_shift, decel_shift, cam_shift = kc.slip_shift, kc.decel_shift, kc.cam_shift
                    direction = shift_dir[active.kind]
                slip = cfg.slip_base + bias[v] + slip_shift + rng.normal(0.0, cfg.slip_sd)
                decel = cfg.decel_base + decel_shift + rng.normal(0.0, cfg.decel_sd)
                emb = appearance[s] + rng.normal(0.0, cfg.cam_noise, cfg.cam_dim)
                if direction is not None:
                    emb = emb + direction * cam_shift * np.sqrt(cfg.cam_dim)
                obs_id = f"sim{run_seed}-v{v}-l{k}-s{s}"
                observations.append(
                    Observation(
                        obs_id=obs_id,
                        vehicle_id=f"v{v}",
                        t=t,
                        segment=s,
                        can={CAN_SLIP: float(slip), CAN_DECEL: float(decel)},
                        cam=emb,
                        frame_ref=f"sim://seed{run_seed}/v{v}/lap{k}/seg{s}",
                    )
                )
                labels[obs_id] = active.event_id if active is not None else None
            k += 1
    observations.sort(key=lambda o: o.t)
    return SimResult(observations, labels, events)
