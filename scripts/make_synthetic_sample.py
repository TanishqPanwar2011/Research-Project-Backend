"""Create fake task_usage files in the REAL 2011 format, so you can test the whole pipeline
before downloading anything. Machine load follows a pattern; only CPU-related columns carry signal."""
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sladro.data2011 import COLUMNS  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/task_usage")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
M, T, PARTS = 300, 48, 4
load = np.zeros((M, T))
load[:, 0] = rng.uniform(0.1, 0.6, M)
for t in range(1, T):
    load[:, t] = np.clip(0.8 * load[:, t - 1] + 0.08 + 0.08 * rng.normal(size=M), 0.01, 0.95)

per = T // PARTS
for p in range(PARTS):
    rows = []
    for t in range(p * per, (p + 1) * per):
        for m in range(M):
            n = 1 + int(round(3 * load[m, t]))
            for k in range(n):
                cpu = load[m, t] / n * (1 + 0.1 * rng.normal())
                start = (t + 2) * 300_000_000 + int(rng.integers(0, 5_000_000))
                row = dict.fromkeys(COLUMNS, "")
                row.update(start_time=start, end_time=start + 300_000_000, job_id=1000 + m, task_index=k,
                           machine_id=5000 + m, cpu_rate=cpu, canonical_mem=rng.uniform(0, .05),
                           assigned_mem=rng.uniform(0, .06), total_cache=rng.uniform(0, .02),
                           max_mem=rng.uniform(0, .06), disk_io_time=rng.uniform(0, .01),
                           disk_space=rng.uniform(0, .01), max_cpu=cpu * 1.4 + 0.01 * rng.normal(),
                           cpi=rng.uniform(.8, 3), mai=rng.uniform(0, .01))
                rows.append(",".join(str(row[c]) for c in COLUMNS))
    with gzip.open(out / f"part-{p:05d}-of-00500.csv.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
print(f"Wrote {PARTS} synthetic part files to {out}")
