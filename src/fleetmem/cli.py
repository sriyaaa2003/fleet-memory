"""Command line: fleetmem download | passes | run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fleetmem")
    ap.add_argument("command", choices=["download", "passes", "run"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
    except (OSError, ConfigError) as exc:
        print(f"config error: {exc}")
        return 2
    from . import ved

    if a.command == "download":
        ved.download(cfg)
    elif a.command == "passes":
        df = ved.build_passes(cfg)
        print(f"{df.height:,} passes, {df['key'].n_unique():,} road cells, {df['veh'].n_unique()} vehicles")
    else:
        from .evaluate import run
        from .report import plot, to_markdown

        passes = ved.build_passes(cfg).to_pandas()
        out = Path(a.out)
        res = run(cfg, passes, out)
        (out / "results.md").write_text(to_markdown(res), encoding="utf-8")
        plot(res, out / "test_scores_sample.parquet", out)
        print(json.dumps(res["test"]["all"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
