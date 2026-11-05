"""Turn the Google 2011 cluster trace (task_usage table) into a machine-level dataset.

One row = one machine during one 5-minute window.
Features describe that window; the target is the machine's CPU usage in the NEXT window.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

BUCKET_US = 300 * 1_000_000  # 5 minutes, in microseconds

# The 20 columns of task_usage in clusterdata-2011-2, in file order.
# Please verify against the "v2.1 format + schema" document linked in ClusterData2011_2.md.
COLUMNS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id", "cpu_rate",
    "canonical_mem", "assigned_mem", "unmapped_cache", "total_cache", "max_mem",
    "disk_io_time", "disk_space", "max_cpu", "max_disk_io", "cpi", "mai",
    "sample_portion", "aggregation_type", "sampled_cpu",
]
USE = ["start_time", "machine_id", "cpu_rate", "canonical_mem", "assigned_mem", "total_cache",
       "max_mem", "disk_io_time", "disk_space", "max_cpu", "cpi", "mai"]
SUM_COLS = ["cpu_rate", "canonical_mem", "assigned_mem", "total_cache", "max_mem",
            "disk_io_time", "disk_space", "max_cpu"]
FEATURES = SUM_COLS + ["n_tasks", "cpi_mean", "mai_mean"]
TARGET = "target_cpu_next"

# How these relate to the paper's five features (the 2011 trace has no network bandwidth or latency).
FEATURE_LABELS = {
    "cpu_rate": "CPU usage", "max_cpu": "CPU usage (peak)", "canonical_mem": "Memory consumption",
    "assigned_mem": "Memory (assigned)", "total_cache": "Memory (page cache)", "max_mem": "Memory (peak)",
    "disk_io_time": "Disk I/O", "disk_space": "Disk space", "n_tasks": "Running tasks",
    "cpi_mean": "Cycles per instruction (latency proxy)", "mai_mean": "Memory accesses per instruction",
}


def _aggregate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["machine_id", "start_time"]).copy()
    df["bucket"] = (df["start_time"] // BUCKET_US).astype("int64")
    g = df.groupby(["machine_id", "bucket"])
    out = g[SUM_COLS].sum()
    out["n_tasks"] = g.size()
    out["cpi_sum"], out["cpi_n"] = g["cpi"].sum(), g["cpi"].count()
    out["mai_sum"], out["mai_n"] = g["mai"].sum(), g["mai"].count()
    return out


def build_machine_dataset(raw_dir: str, max_files: int | None = None, chunksize: int = 1_000_000) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "part-*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No part-*.csv.gz files found in {raw_dir}")
    files = files[:max_files] if max_files else files

    per_file = []
    for i, f in enumerate(files, 1):
        ncols = pd.read_csv(f, header=None, nrows=1).shape[1]
        if ncols != len(COLUMNS):
            raise ValueError(f"{f} has {ncols} columns, expected {len(COLUMNS)}. Check the schema document.")
        parts = [_aggregate_chunk(c) for c in
                 pd.read_csv(f, header=None, names=COLUMNS, usecols=USE, chunksize=chunksize)]
        per_file.append(pd.concat(parts).groupby(level=[0, 1]).sum())  # chunks can split a window: sums add up
        print(f"  [{i}/{len(files)}] {os.path.basename(f)}")

    df = pd.concat(per_file).groupby(level=[0, 1]).sum().reset_index()
    df["cpi_mean"] = (df["cpi_sum"] / df["cpi_n"]).where(df["cpi_n"] > 0)
    df["mai_mean"] = (df["mai_sum"] / df["mai_n"]).where(df["mai_n"] > 0)
    df[["cpi_mean", "mai_mean"]] = df[["cpi_mean", "mai_mean"]].fillna(df[["cpi_mean", "mai_mean"]].median())
    df = df.drop(columns=["cpi_sum", "cpi_n", "mai_sum", "mai_n"]).sort_values(["machine_id", "bucket"])

    # Target: same machine, next consecutive window.
    grp = df.groupby("machine_id")
    df[TARGET] = grp["cpu_rate"].shift(-1)
    gap = grp["bucket"].shift(-1) - df["bucket"]
    df = df[gap == 1]
    # The first and last windows are only partly covered by the downloaded files: drop them.
    bmin, bmax = df["bucket"].min(), df["bucket"].max()
    return df[(df["bucket"] > bmin) & (df["bucket"] < bmax)].reset_index(drop=True)
