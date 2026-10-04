"""Per-observation evidence models.

Each detector maps an Observation to a log-likelihood ratio (LLR) for "the segment's condition has
changed", or None when it has no usable signal (modality missing). Models are trained on labelled
simulated or reviewed drives and are never given the ground-truth label at inference time.

- TimestampDetector: fixed single-threshold trigger on raw CAN; pairs with TimestampMemory.
- CanOnlyDetector / CameraOnlyDetector: one logistic model on one modality.
- FusionDetector: a logistic model per modality-availability set; with both modalities it fuses,
  with one it degrades to that modality, with none it abstains (None).
"""

from __future__ import annotations

import math

import numpy as np

from .config import TimestampConfig
from .schema import CAN_DECEL, CAN_SLIP, Observation


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class _Reference:
    """Per-segment mean camera embedding from event-free passes (the segment's 'normal look')."""

    def __init__(self) -> None:
        self._mean: dict[int, np.ndarray] = {}

    def fit(self, obs: list[Observation], labels: dict[str, int | None]) -> None:
        acc: dict[int, list[np.ndarray]] = {}
        for o in obs:
            if o.cam is not None and labels.get(o.obs_id) is None:
                acc.setdefault(o.segment, []).append(o.cam)
        self._mean = {s: np.mean(v, axis=0) for s, v in acc.items()}

    def distance(self, o: Observation) -> float | None:
        ref = self._mean.get(o.segment)
        if o.cam is None or ref is None:
            return None
        return float(np.linalg.norm(o.cam - ref) / math.sqrt(len(ref)))


class _Model:
    """L2-regularised logistic regression fitted by IRLS (Newton), on standardised features."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self._l2, self._max_iter, self._tol = l2, max_iter, tol
        self._mu = np.zeros(1)
        self._sd = np.ones(1)
        self._w = np.zeros(1)
        self._b = 0.0
        self._prior_logit = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(set(y.tolist())) < 2:
            raise ValueError("training data needs both event and event-free passes")
        self._prior_logit = _logit(float(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
        self._mu = x.mean(axis=0)
        self._sd = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        z = np.hstack([(x - self._mu) / self._sd, np.ones((len(x), 1))])
        reg = self._l2 * np.eye(z.shape[1])
        reg[-1, -1] = 0.0  # do not penalise the intercept
        beta = np.zeros(z.shape[1])
        for _ in range(self._max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -35, 35)))
            grad = z.T @ (p - y) + reg @ beta
            hess = (z.T * (p * (1 - p))) @ z + reg
            step = np.linalg.solve(hess, grad)
            beta = beta - step
            if np.max(np.abs(step)) < self._tol:
                break
        self._w, self._b = beta[:-1], float(beta[-1])

    def llr(self, x: np.ndarray) -> np.ndarray:
        return ((x - self._mu) / self._sd) @ self._w + self._b - self._prior_logit


def _can_row(o: Observation) -> list[float]:
    assert o.can is not None
    return [o.can[CAN_SLIP], o.can[CAN_DECEL]]


class Detector:
    name = "base"
    memory_kind = "decay"

    def fit(self, obs: list[Observation], labels: dict[str, int | None]) -> None:
        raise NotImplementedError

    def evaluate(
        self, obs: list[Observation]
    ) -> tuple[list[float | None], list[dict[str, float]]]:
        """Return (LLR per observation, per-modality LLR contributions for attribution)."""
        raise NotImplementedError


def _train_arrays(obs, labels, ref):
    rows_can, rows_cam, y = [], [], []
    for o in obs:
        d = ref.distance(o)
        if o.can is None or d is None:
            continue
        rows_can.append(_can_row(o))
        rows_cam.append([d])
        y.append(1 if labels.get(o.obs_id) is not None else 0)
    return np.array(rows_can), np.array(rows_cam), np.array(y)


class TimestampDetector(Detector):
    name = "timestamp"
    memory_kind = "timestamp"

    def __init__(self, cfg: TimestampConfig):
        self._cfg = cfg

    def fit(self, obs, labels) -> None:  # nothing to learn: fixed rule
        return None

    def evaluate(self, obs):
        llrs: list[float | None] = []
        for o in obs:
            if o.can is None:
                llrs.append(None)
                continue
            hit = (
                o.can[CAN_SLIP] > self._cfg.slip_trigger
                or o.can[CAN_DECEL] > self._cfg.decel_trigger
            )
            llrs.append(1.0 if hit else 0.0)
        return llrs, [{} for _ in obs]


class CanOnlyDetector(Detector):
    name = "can_only"

    def __init__(self) -> None:
        self._model = _Model()

    def fit(self, obs, labels) -> None:
        ref = _Reference()
        ref.fit(obs, labels)
        x_can, _, y = _train_arrays(obs, labels, ref)
        self._model.fit(x_can, y)

    def evaluate(self, obs):
        llrs: list[float | None] = [None] * len(obs)
        contribs: list[dict[str, float]] = [{} for _ in obs]
        idx = [i for i, o in enumerate(obs) if o.can is not None]
        if idx:
            out = self._model.llr(np.array([_can_row(obs[i]) for i in idx]))
            for i, v in zip(idx, out, strict=True):
                llrs[i] = float(v)
                contribs[i] = {"can": float(v)}
        return llrs, contribs


class CameraOnlyDetector(Detector):
    name = "camera_only"

    def __init__(self) -> None:
        self._ref = _Reference()
        self._model = _Model()

    def fit(self, obs, labels) -> None:
        self._ref.fit(obs, labels)
        _, x_cam, y = _train_arrays(obs, labels, self._ref)
        self._model.fit(x_cam, y)

    def evaluate(self, obs):
        llrs: list[float | None] = [None] * len(obs)
        contribs: list[dict[str, float]] = [{} for _ in obs]
        dists = [self._ref.distance(o) for o in obs]
        idx = [i for i, d in enumerate(dists) if d is not None]
        if idx:
            out = self._model.llr(np.array([[dists[i]] for i in idx]))
            for i, v in zip(idx, out, strict=True):
                llrs[i] = float(v)
                contribs[i] = {"cam": float(v)}
        return llrs, contribs


class FusionDetector(Detector):
    name = "fusion"

    def __init__(self) -> None:
        self._ref = _Reference()
        self._can = _Model()
        self._cam = _Model()
        self._both = _Model()

    def fit(self, obs, labels) -> None:
        self._ref.fit(obs, labels)
        x_can, x_cam, y = _train_arrays(obs, labels, self._ref)
        self._can.fit(x_can, y)
        self._cam.fit(x_cam, y)
        self._both.fit(np.hstack([x_can, x_cam]), y)

    def evaluate(self, obs):
        n = len(obs)
        llrs: list[float | None] = [None] * n
        contribs: list[dict[str, float]] = [{} for _ in range(n)]
        dists = [self._ref.distance(o) for o in obs]
        has_can = [o.can is not None for o in obs]
        has_cam = [d is not None for d in dists]

        both = [i for i in range(n) if has_can[i] and has_cam[i]]
        can_only = [i for i in range(n) if has_can[i] and not has_cam[i]]
        cam_only = [i for i in range(n) if has_cam[i] and not has_can[i]]

        if both:
            x_can = np.array([_can_row(obs[i]) for i in both])
            x_cam = np.array([[dists[i]] for i in both])
            total = self._both.llr(np.hstack([x_can, x_cam]))
            c_can, c_cam = self._can.llr(x_can), self._cam.llr(x_cam)
            for j, i in enumerate(both):
                llrs[i] = float(total[j])
                contribs[i] = {"can": float(c_can[j]), "cam": float(c_cam[j])}
        if can_only:
            out = self._can.llr(np.array([_can_row(obs[i]) for i in can_only]))
            for i, v in zip(can_only, out, strict=True):
                llrs[i], contribs[i] = float(v), {"can": float(v)}
        if cam_only:
            out = self._cam.llr(np.array([[dists[i]] for i in cam_only]))
            for i, v in zip(cam_only, out, strict=True):
                llrs[i], contribs[i] = float(v), {"cam": float(v)}
        return llrs, contribs


def build_detectors(timestamp_cfg: TimestampConfig) -> dict[str, Detector]:
    dets = [
        TimestampDetector(timestamp_cfg),
        CanOnlyDetector(),
        CameraOnlyDetector(),
        FusionDetector(),
    ]
    return {d.name: d for d in dets}
