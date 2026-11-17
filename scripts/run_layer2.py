"""Train the CNN-LSTM twice (all features vs Layer-1-selected features) and save artifacts for the API."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sladro.layer2 import run_layer2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/machine_dataset.pkl")
ap.add_argument("--layer1", default="outputs/layer1_result.json")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--no-delta", action="store_true", help="predict the next value directly instead of the change")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

df = pd.read_pickle(a.data)
l1 = json.loads(Path(a.layer1).read_text())
art, out = Path("artifacts"), Path("outputs")
art.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
shutil.copy(a.layer1, art / "layer1.json")  # the API reads this

rows = {}
for name, feats in (("all", l1["all_features"]), ("selected", l1["selected"])):
    print(f"\n== CNN-LSTM on '{name}' features ({len(feats)}): {', '.join(feats)}")
    res, state = run_layer2(df, feats, epochs=a.epochs, delta=not a.no_delta, seed=a.seed)
    (art / f"layer2_{name}.json").write_text(json.dumps(res))
    torch.save({"state": state, "features": feats}, out / f"layer2_{name}.pt")
    rows[name] = res

print("\nTest-set results (higher R2 is better, lower MSE/MAE is better):")
print(f"  {'':22s}{'R2':>8s}{'MSE':>12s}{'MAE':>10s}")
for n, r in rows.items():
    m = r["metrics"]
    print(f"  {'CNN-LSTM ' + n:22s}{m['r2']:8.3f}{m['mse']:12.6f}{m['mae']:10.5f}")
m = rows["all"]["metrics"]
print(f"  {'naive: next = current':22s}{m['persistenceR2']:8.3f}{m['persistenceMSE']:12.6f}")
print("\nSaved artifacts/ (commit these) and outputs/*.pt (model weights, not committed)")
