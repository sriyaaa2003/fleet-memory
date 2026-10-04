import numpy as np
import pytest

from fleetmem.detectors import (
    CameraOnlyDetector,
    CanOnlyDetector,
    FusionDetector,
    TimestampDetector,
)
from fleetmem.sim import generate


@pytest.fixture(scope="module")
def train(cfg):
    return generate(cfg.sim, cfg.eval.train_seed)


@pytest.fixture(scope="module")
def test(cfg):
    return generate(cfg.sim, 1)


def _mean_llr(llrs, labels, obs, want_event):
    v = [
        x
        for x, o in zip(llrs, obs, strict=True)
        if x is not None and (labels[o.obs_id] is not None) == want_event
    ]
    return float(np.mean(v))


@pytest.mark.parametrize("cls", [CanOnlyDetector, CameraOnlyDetector, FusionDetector])
def test_event_passes_score_higher(cls, train, test):
    det = cls()
    det.fit(train.observations, train.labels)
    llrs, _ = det.evaluate(test.observations)
    assert _mean_llr(llrs, test.labels, test.observations, True) > _mean_llr(
        llrs, test.labels, test.observations, False
    )


def test_missing_modality_yields_none_not_a_guess(train, test):
    obs = [o.drop(can=True, cam=True) for o in test.observations[:50]]
    for cls in (CanOnlyDetector, CameraOnlyDetector, FusionDetector):
        det = cls()
        det.fit(train.observations, train.labels)
        llrs, _ = det.evaluate(obs)
        assert all(x is None for x in llrs)


def test_modality_specific_detectors_ignore_other_modality(train, test):
    can = CanOnlyDetector()
    can.fit(train.observations, train.labels)
    cam = CameraOnlyDetector()
    cam.fit(train.observations, train.labels)
    no_cam = [o.drop(cam=True) for o in test.observations[:50]]
    no_can = [o.drop(can=True) for o in test.observations[:50]]
    assert all(x is not None for x in can.evaluate(no_cam)[0])
    assert all(x is None for x in cam.evaluate(no_cam)[0])
    assert all(x is None for x in can.evaluate(no_can)[0])
    assert all(x is not None for x in cam.evaluate(no_can)[0])


def test_fusion_degrades_to_single_modality(train, test):
    fus = FusionDetector()
    fus.fit(train.observations, train.labels)
    can = CanOnlyDetector()
    can.fit(train.observations, train.labels)
    obs = [o.drop(cam=True) for o in test.observations[:100]]
    assert np.allclose(fus.evaluate(obs)[0], can.evaluate(obs)[0])


def test_fusion_contributions_present_when_both(train, test):
    fus = FusionDetector()
    fus.fit(train.observations, train.labels)
    _, contribs = fus.evaluate(test.observations[:5])
    assert all({"can", "cam"} <= set(c) for c in contribs)


def test_timestamp_rule(cfg, test):
    det = TimestampDetector(cfg.timestamp)
    llrs, _ = det.evaluate([test.observations[0].drop(can=True)])
    assert llrs == [None]


def test_fit_needs_both_classes(cfg, train):
    obs = [o for o in train.observations if train.labels[o.obs_id] is None]
    with pytest.raises(ValueError):
        CanOnlyDetector().fit(obs, train.labels)
