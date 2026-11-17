"""Run Layer 1 (OOA-PSO feature selection) and save the result as JSON."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sladro.data2011 import FEATURE_LABELS  # noqa: E402
from sladro.layer1 import run_layer1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/machine_dataset.parquet")
ap.add_argument("--out", default="outputs/layer1_result.json")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--beta", type=float, default=0.05, help="penalty per fraction of features used")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--plot", action="store_true", help="also save a convergence chart")
a = ap.parse_args()

df = pd.read_parquet(a.data)
print(f"Loaded {len(df):,} rows. Running OOA-PSO...")
r = run_layer1(df, n_iter=a.iters, beta=a.beta, seed=a.seed)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(r, indent=2))

print(f"\nKept {len(r['selected'])} of {len(r['all_features'])} features:")
for f in r["selected"]:
    print(f"  + {f:14s} {FEATURE_LABELS.get(f, '')}")
print("Dropped:", ", ".join(f for f in r["all_features"] if f not in r["selected"]) or "(none)")
q = r["r2"]
print("\nHow well each choice predicts next-window CPU (R2, higher is better):")
print(f"  {'':22s}{'validation':>12s}{'test':>10s}")
print(f"  {'all features':22s}{q['val_all']:12.3f}{q['test_all']:10.3f}")
print(f"  {'selected features':22s}{q['val_selected']:12.3f}{q['test_selected']:10.3f}")
print(f"  {'naive: next = current':22s}{'':12s}{q['test_persistence']:10.3f}")
print(f"\n{r['n_evals']} subsets scored in {r['seconds']}s. Saved {a.out}")

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.5))
    plt.plot(r["history"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title("OOA-PSO convergence")
    plt.tight_layout()
    png = str(Path(a.out).with_suffix(".png"))
    plt.savefig(png, dpi=150)
    print("Saved", png)
