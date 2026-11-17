"""Layer 2 data preparation: 8-step sliding windows per machine, split by time (no torch needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data2011 import TARGET

SEQ = 8


def prepare(df: pd.DataFrame, feats: list[str], seq: int = SEQ, delta: bool = True,
            max_train: int = 200_000, max_eval: int = 50_000, seed: int = 0) -> dict:
    """Build train/val/test windows.

    One sample = `seq` consecutive 5-minute windows of ONE machine -> that machine's CPU in the next window.
    Splits are by time. Training windows whose target falls into the validation period are purged.
    If delta=True the network predicts (next CPU - current CPU), which is added back afterwards.
    """
    df = df.sort_values(["machine_id", "bucket"]).reset_index(drop=True)
    m, b = df["machine_id"].to_numpy(), df["bucket"].to_numpy()
    ends = np.arange(seq - 1, len(df))
    start = ends - (seq - 1)
    ends = ends[(m[ends] == m[start]) & (b[ends] - b[start] == seq - 1)]  # only gap-free windows of one machine
    if len(ends) == 0:
        raise ValueError("No complete windows. Download more part files.")

    ub = np.sort(np.unique(b))
    t1, t2 = ub[int(len(ub) * 0.6)], ub[int(len(ub) * 0.8)]
    be = b[ends]
    idx = {"train": ends[be < t1 - 1], "val": ends[(be >= t1) & (be < t2 - 1)], "test": ends[be >= t2]}
    rng = np.random.default_rng(seed)
    for k, cap in (("train", max_train), ("val", max_eval), ("test", max_eval)):
        if len(idx[k]) == 0:
            raise ValueError(f"Empty {k} split. Download more part files.")
        if len(idx[k]) > cap:
            idx[k] = np.sort(rng.choice(idx[k], cap, replace=False))

    scaler = StandardScaler().fit(df.loc[b < t1, feats].to_numpy(float))
    vals = scaler.transform(df[feats].to_numpy(float)).astype(np.float32)
    cur = df["cpu_rate"].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    base = cur if delta else np.zeros_like(cur)
    tr_t = (y - base)[idx["train"]]
    mu, sd = float(tr_t.mean()), float(tr_t.std() or 1.0)
    off = np.arange(-(seq - 1), 1)

    out = {"mu": mu, "sd": sd, "feats": feats, "delta": delta, "seq": seq}
    for k, e in idx.items():
        out[k] = {
            "x": vals[e[:, None] + off],                       # (N, seq, n_features)
            "z": ((y - base)[e] - mu) / sd,                    # scaled training target
            "base": base[e], "y": y[e], "cur": cur[e], "bucket": b[e],
        }
    return out
