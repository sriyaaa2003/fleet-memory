"""Render experiment results as Markdown (and an optional dropout plot)."""

from __future__ import annotations

from pathlib import Path

from .experiment import METHODS


def _fmt(cell: dict, digits: int = 3) -> str:
    if cell["mean"] is None:
        return "n/a"
    return f"{cell['mean']:.{digits}f} ± {cell['std']:.{digits}f}"


def to_markdown(res: dict) -> str:
    sc = res["scenarios"]
    n = len(res["test_seeds"])
    lines = [
        "# Results (synthetic fleet simulator)",
        "",
        f"> {res['note']} Mean ± std over {n} held-out seeds; `n/a` means undefined "
        "(e.g. precision with zero warnings).",
        "",
        "## Clean sensors",
        "",
        "| method | precision | recall | F1 | false-alarm rate | event recall "
        "| detection delay (s) | lead time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        r = sc["clean"][m]
        lines.append(
            f"| {m} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} "
            f"| {_fmt(r['fpr'], 4)} | {_fmt(r['event_recall'])} "
            f"| {_fmt(r['detection_delay_s'], 0)} | {_fmt(r['lead_time_s'], 0)} |"
        )
    for prefix, label in (("drop_cam", "camera"), ("drop_can", "CAN")):
        names = sorted(k for k in sc if k.startswith(prefix))
        lines += [
            "",
            f"## F1 when the {label} signal is dropped (fraction of passes)",
            "",
            "| method | 0.00 | " + " | ".join(k.split("_")[-1] for k in names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        for m in METHODS:
            cells = [_fmt(sc["clean"][m]["f1"])] + [_fmt(sc[k][m]["f1"]) for k in names]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    f = sc["clean"]["fusion"]
    lines += [
        "",
        "## Evidence-grounded recommender (fusion, clean sensors)",
        "",
        "- Warned passes that produced a recommendation (rest abstained): "
        f"{_fmt(f['rec_emit_rate'])}",
        f"- Precision of emitted recommendations: {_fmt(f['rec_precision'])} "
        f"(vs. warning precision {_fmt(f['precision'])})",
        f"- Emitted recommendations passing citation verification: {_fmt(f['rec_verified'])}",
        "",
        "Citation verification is 1.0 by construction for the template renderer; it becomes a "
        "real test once an LLM phrases the claims.",
        "",
    ]
    return "\n".join(lines)


def plot_dropout(res: dict, out_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    sc = res["scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, prefix, label in zip(axes, ("drop_cam", "drop_can"), ("camera", "CAN"), strict=True):
        names = sorted(k for k in sc if k.startswith(prefix))
        xs = [0.0] + [float(k.split("_")[-1]) for k in names]
        for m in METHODS:
            ys = [sc["clean"][m]["f1"]["mean"]] + [sc[k][m]["f1"]["mean"] for k in names]
            ax.plot(xs, [0.0 if y is None else y for y in ys], marker="o", label=m)
        ax.set_xlabel(f"fraction of passes with {label} dropped")
        ax.set_title(f"{label} dropout")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 (per pass)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    path = out_dir / "dropout_f1.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
