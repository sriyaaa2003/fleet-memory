import json

import numpy as np

from fleetmem.cli import main
from fleetmem.experiment import METHODS, apply_dropout, run, scenarios
from fleetmem.sim import generate


def test_scenarios_include_clean_and_both_dropouts():
    names = [n for n, *_ in scenarios([0.0, 0.5, 1.0])]
    assert names == ["clean", "drop_cam_0.50", "drop_cam_1.00", "drop_can_0.50", "drop_can_1.00"]


def test_dropout_removes_only_requested_modality(cfg):
    obs = generate(cfg.sim, 1).observations[:200]
    out = apply_dropout(obs, 1.0, 0.0, np.random.default_rng(0))
    assert all(o.can is None and o.cam is not None for o in out)
    out = apply_dropout(obs, 0.0, 1.0, np.random.default_rng(0))
    assert all(o.can is not None and o.cam is None and o.frame_ref is None for o in out)


def test_end_to_end_ordering_and_sanity(small_cfg):
    res = run(small_cfg)
    clean = res["scenarios"]["clean"]
    assert set(clean) == set(METHODS)
    f1 = {m: clean[m]["f1"]["mean"] for m in METHODS}
    # fusion should not lose to either single modality on data where both modalities matter
    assert f1["fusion"] >= max(f1["can_only"], f1["camera_only"]) - 0.02
    # with every signal gone nothing can be detected
    gone = res["scenarios"]["drop_can_1.00"]["can_only"]["recall"]["mean"]
    assert gone == 0.0 or gone is None
    # false alarms stay rare in clean conditions
    assert clean["fusion"]["fpr"]["mean"] < 0.05
    # emitted recommendations are always verifiable
    assert clean["fusion"]["rec_verified"]["mean"] in (1.0, None)


def test_cli_writes_outputs(tmp_path, small_cfg, cfg):
    out = tmp_path / "out"
    code = main(["run", "--config", "configs/default.yaml", "--out", str(out)])
    # default config is large; only check the contract of the outputs
    assert code == 0
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "clean" in data["scenarios"]
    assert (out / "results.md").read_text(encoding="utf-8").startswith("# Results")


def test_cli_reports_bad_config(tmp_path, capsys):
    assert main(["run", "--config", str(tmp_path / "missing.yaml")]) == 2
