"""Vehicle Energy Dataset (VED): real OBD-II logs of 383 cars driven around Ann Arbor, Michigan, Nov 2017 to Nov 2018.

Oh, Leblanc, Peng, "Vehicle Energy Dataset (VED), a large-scale dataset for vehicle energy consumption research",
IEEE Trans. Intelligent Transportation Systems, 2022. Apache-2.0. https://github.com/gsoh/VED

Each record has a trip-relative timestamp, GPS position and vehicle speed. A *pass* is one trip driving through one road cell
(a small grid square seen from one of a few headings): the unit a fleet memory is built from."""
from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config

BASE = "https://raw.githubusercontent.com/gsoh/VED/master/Data"
PARTS = ["VED_DynamicData_Part1.7z", "VED_DynamicData_Part2.7z"]
COLS = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]"]


def download(cfg: Config) -> None:
    """Fetch the two archives (about 176 MB) and unpack them into data.raw_dir. Needs `7z`, or Windows' bsdtar (`tar`)."""
    raw = Path(cfg.data.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if glob.glob(str(raw / "VED_*_week.csv")):
        return
    for name in PARTS:
        dest = raw / name
        if not dest.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        tool = shutil.which("7z") or shutil.which("7za")
        cmd = [tool, "x", "-y", f"-o{raw}", str(dest)] if tool else [os.path.join(os.environ.get("SystemRoot", ""), "System32", "tar.exe"), "-xf", str(dest), "-C", str(raw)]
        subprocess.run(cmd, check=True)


def heading_bin(dlat: np.ndarray, dlon: np.ndarray, bins: int) -> np.ndarray:
    ang = (np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0
    return (((ang + 180.0 / bins) % 360.0) // (360.0 / bins)).astype(np.int64)


def cell_key(lat_i: pl.Expr, lon_i: pl.Expr, head: pl.Expr) -> pl.Expr:
    return (lat_i * 100_000 + (lon_i + 50_000)) * 10 + head


def passes_from_week(path: str, cfg: Config) -> pl.DataFrame:
    d = cfg.data
    df = pl.read_csv(path, columns=COLS, infer_schema_length=100000,
                     schema_overrides={"Vehicle Speed[km/h]": pl.Float64, "DayNum": pl.Float64, "Timestamp(ms)": pl.Int64})
    df = (df.rename({"VehId": "veh", "Trip": "trip", "Latitude[deg]": "lat", "Longitude[deg]": "lon", "Vehicle Speed[km/h]": "v"})
          .with_columns(((pl.col("DayNum") - 1.0) * 86400.0 + pl.col("Timestamp(ms)") / 1000.0).alias("t"))
          .drop_nulls(["lat", "lon", "v"])
          .sort(["veh", "trip", "t"]))
    dlat = pl.col("lat").diff().over(["veh", "trip"])
    dlon = (pl.col("lon").diff() * (pl.col("lat") * math.pi / 180.0).cos()).over(["veh", "trip"])
    df = df.with_columns(dlat.alias("dlat"), dlon.alias("dlon")).filter(pl.col("dlat").is_not_null())
    moving = (pl.col("dlat").abs() + pl.col("dlon").abs()) > 1e-6
    head = heading_bin(df["dlat"].to_numpy(), df["dlon"].to_numpy(), d.heading_bins)
    df = (df.with_columns(pl.Series("head", head).cast(pl.Int64))
          .with_columns(pl.when(moving).then(pl.col("head")).otherwise(None).alias("head"))
          .with_columns(pl.col("head").forward_fill().over(["veh", "trip"]))
          .drop_nulls("head")
          .with_columns(cell_key((pl.col("lat") / d.cell_lat_deg).floor().cast(pl.Int64),
                                 (pl.col("lon") / d.cell_lon_deg).floor().cast(pl.Int64), pl.col("head")).alias("key")))
    df = df.with_columns((pl.col("key") != pl.col("key").shift(1)).over(["veh", "trip"]).fill_null(True).cast(pl.Int64).cum_sum().over(["veh", "trip"]).alias("run"))
    out = (df.group_by(["veh", "trip", "run"], maintain_order=True)
           .agg(pl.col("key").first(), pl.col("t").min().alias("t0"), pl.col("t").max().alias("t1"),
                pl.col("v").mean().cast(pl.Float32).alias("v_mean"), pl.col("v").min().cast(pl.Float32).alias("v_min"), pl.len().alias("n"))
           .filter(pl.col("n") >= d.min_records_per_pass)
           .drop("run"))
    return out.with_columns(pl.col("veh").cast(pl.Int32), pl.col("trip").cast(pl.Int32))


def build_passes(cfg: Config) -> pl.DataFrame:
    out = Path(cfg.data.passes_file)
    if out.exists():
        return pl.read_parquet(out)
    files = sorted(glob.glob(os.path.join(cfg.data.raw_dir, "VED_*_week.csv")))
    if not files:
        raise FileNotFoundError(f"no VED weekly files in {cfg.data.raw_dir}: run `fleetmem download`")
    frames = [passes_from_week(f, cfg) for f in files]
    df = pl.concat(frames).with_columns((pl.col("t0") / 86400.0).alias("day")).sort("t0")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    return df
