import numpy as np
import pandas as pd

from sladro.data2011 import TARGET
from sladro.layer2_data import prepare


def _df(machines=20, buckets=40, gap_machine=3):
    rows = []
    rng = np.random.default_rng(0)
    for m in range(machines):
        x = 0.3
        for t in range(buckets):
            if m == gap_machine and t == 20:
                continue  # a missing window: no window may span it
            x = float(np.clip(0.8 * x + 0.06 + 0.03 * rng.normal(), 0, 1))
            rows.append(dict(machine_id=m, bucket=t, cpu_rate=x, mem=rng.random()))
    d = pd.DataFrame(rows).sort_values(["machine_id", "bucket"])
    d[TARGET] = d.groupby("machine_id")["cpu_rate"].shift(-1)
    return d[d.groupby("machine_id")["bucket"].shift(-1) - d["bucket"] == 1].reset_index(drop=True)


def test_windows_shapes_and_time_order():
    d = prepare(_df(), ["cpu_rate", "mem"], seq=8)
    assert d["train"]["x"].shape[1:] == (8, 2)
    assert d["train"]["bucket"].max() < d["val"]["bucket"].min()
    assert d["val"]["bucket"].max() < d["test"]["bucket"].min()


def test_delta_reconstruction_is_exact():
    d = prepare(_df(), ["cpu_rate", "mem"], seq=8, delta=True)
    te = d["test"]
    assert np.allclose(te["z"] * d["sd"] + d["mu"] + te["base"], te["y"])
