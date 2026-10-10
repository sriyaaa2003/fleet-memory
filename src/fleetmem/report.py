"""Markdown tables and figures from results.json."""
from __future__ import annotations

from pathlib import Path

import numpy as np

NAMES = {"static": "static (history only)", "last": "last pass", "memory": "fleet memory", "fused": "fused"}


def _fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def to_markdown(res: dict) -> str:
    d, t, b = res["data"], res["test"], res["bootstrap"]
    L = [f"{d['passes']:,} passes through {d['cells']:,} road cells by {d['vehicles']} vehicles. Slow-pass rate: "
         + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d["slow_rate"].items()) + ".", ""]
    L += ["| test days, all passes | AP [95% CI] | AUROC [95% CI] | Brier |", "|---|---|---|---|"]
    for m in ("static", "last", "memory", "fused"):
        r, ci = t["all"][m], b["ci"][m]
        L.append(f"| {NAMES[m]} | {_fmt(r['ap'])} [{ci['ap'][0]:.3f}, {ci['ap'][1]:.3f}] | {_fmt(r['auroc'])} [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}] | {_fmt(r['brier'])} |")
    dl = b["fused_minus_static_ci"]
    L += ["", f"Fused minus static: AP {t['all']['fused']['ap'] - t['all']['static']['ap']:+.3f} [{dl['ap'][0]:+.3f}, {dl['ap'][1]:+.3f}], "
          f"AUROC {t['all']['fused']['auroc'] - t['all']['static']['auroc']:+.3f} [{dl['auroc'][0]:+.3f}, {dl['auroc'][1]:+.3f}].", ""]
    sh = t["shares"]
    L += [f"Fresh evidence (decayed pass count at least the configured level) exists for {100 * sh['fresh_evidence']:.0f}% of queries. "
          f"'Unexpected' passes (static prior below 0.10) are {100 * sh['unexpected_cells']:.0f}% of queries and {100 * sh['slow_rate_when_unexpected']:.1f}% of them are slow.", ""]
    L += ["| subset | static AP | last AP | memory AP | fused AP |", "|---|---|---|---|---|"]
    for key, label in (("fresh_evidence", "fresh evidence"), ("no_fresh_evidence", "no fresh evidence"), ("unexpected", "unexpected (prior < 0.10)")):
        L.append(f"| {label} | " + " | ".join(_fmt(t[key][m]["ap"]) for m in METHODS4) + " |")
    L += ["", "| memory fed by this share of vehicles | static AP | memory AP | fused AP | share with fresh evidence |", "|---|---|---|---|---|"]
    for f, r in res["fleet_fraction"].items():
        L.append(f"| {float(f):.0%} | {_fmt(r['static']['ap'])} | {_fmt(r['memory']['ap'])} | {_fmt(r['fused']['ap'])} | {100 * r['fresh_share']:.0f}% |")
    L += ["", "| warning rule (threshold from validation days) | threshold | precision | recall | warnings per 1000 passes |", "|---|---|---|---|---|"]
    for m, w in res["warning"].items():
        if w["threshold"] is None:
            L.append(f"| {NAMES[m]} | none reaches the target precision | | | |")
        else:
            L.append(f"| {NAMES[m]} | {w['threshold']:.3f} | {w['precision']:.3f} | {w['recall']:.3f} | {w['warnings_per_1000_passes']:.1f} |")
    return "\n".join(L) + "\n"


METHODS4 = ["static", "last", "memory", "fused"]


def plot(res: dict, scores_path: Path, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        return None
    df = pd.read_parquet(scores_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in METHODS4:
        p, r, _ = precision_recall_curve(df.y, df[m])
        axes[0].plot(r, p, label=NAMES[m])
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Test days: will the next pass be slow?")
    axes[0].legend(fontsize=8)
    fr = sorted(res["fleet_fraction"], key=float)
    for m in ("static", "memory", "fused"):
        axes[1].plot([float(f) for f in fr], [res["fleet_fraction"][f][m]["ap"] for f in fr], marker="o", label=NAMES[m])
    axes[1].set_xlabel("share of vehicles feeding the memory")
    axes[1].set_ylabel("average precision")
    axes[1].set_title("Fleet size")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "fleet_memory.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
