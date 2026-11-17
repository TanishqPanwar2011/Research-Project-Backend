"""Aggregate downloaded task_usage part files into one machine-level dataset (Parquet)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sladro.data2011 import FEATURES, build_machine_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/task_usage")
ap.add_argument("--out", default="data/processed/machine_dataset.parquet")
ap.add_argument("--max-files", type=int, default=None)
ap.add_argument("--chunksize", type=int, default=1_000_000, help="lower this if you run out of memory")
a = ap.parse_args()

print("Reading and aggregating...")
df = build_machine_dataset(a.raw, a.max_files, a.chunksize)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out, index=False)

n_win = df["bucket"].nunique()
print(f"\nSaved {a.out}")
print(f"  rows: {len(df):,}   machines: {df['machine_id'].nunique():,}   5-minute windows: {n_win} (~{n_win * 5 / 60:.1f} hours)")
print(f"  features: {', '.join(FEATURES)}")
if n_win < 30:
    print("  Note: fewer than 30 windows. Fine for a first run; download more parts for stronger results.")
