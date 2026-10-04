"""Command line entry point: `fleetmem run --config configs/default.yaml --out results`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_config
from .experiment import run
from .report import plot_dropout, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fleetmem")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="train, replay held-out fleets, write results")
    p_run.add_argument("--config", default="configs/default.yaml")
    p_run.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ConfigError) as exc:
        print(f"config error: {exc}")
        return 2

    res = run(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(res), encoding="utf-8")
    plot = plot_dropout(res, out)
    print(f"wrote {out / 'results.json'} and {out / 'results.md'}" + (f", {plot}" if plot else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
