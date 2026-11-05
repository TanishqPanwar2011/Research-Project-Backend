# Layer 1 (OOA-PSO) in Python: setup, data and code

## 1. What is in the cluster-data repo, and what you can use

| Dataset | Access | Use it? |
|---|---|---|
| **2011 trace** (`ClusterData2011_2.md`): 29 days, ~12.5k machines, May 2011 | Download files from Google Cloud Storage, bucket `clusterdata-2011-2`, ~41 GB compressed in total | **Yes, start here.** The paper says its data spans 29 days, which matches this trace. Free to download, no BigQuery bill. |
| **2019 trace** (`ClusterData2019.md`): 8 cells, May 2019 | **BigQuery only** (~2.4 TiB). The notebooks require a Cloud project with billing enabled | Later, if you want it. `clusterdata_trace_format_v3.proto` and `clusterdata_analysis_colab.ipynb` describe it. |
| **Power trace** (`PowerData2019.md`) | Small CSVs in bucket `powerdata_2019`, or BigQuery | Not needed for SLADRO. |
| Trace v1, ETA traces | Old / unrelated | Skip. |

Important limits to know (they affect how you describe your work):
- The 2011 trace has **no network bandwidth or latency** columns, which the paper lists as features. We use what exists: CPU, memory, disk I/O, cycles per instruction (a latency proxy) and more, 11 candidate features in total.
- The paper's "load" is not defined precisely. We define it as **a machine's CPU usage in the next 5-minute window**.
- The data is CC-BY: cite the trace and mention the source in your report. The repo asks you to join its mailing list (fill in the "reason" field).

## 2. Install the tools (Windows)

1. **Python 3.12**, using the Python install manager from python.org (the `.msix` file, or the Microsoft Store version):
   - Install it (double-click the `.msix`). If a setup window asks questions, answer **n** to "Install CPython now?". You do not need the latest Python.
   - Open a **new** PowerShell and run `py install 3.12`, then check with `py -3.12 --version`.
   - Alternative: the Windows installer for **3.12.10** from https://www.python.org/downloads/release/python-31210/ (pick "Windows installer (64-bit)", tick "Add python.exe to PATH"). Newer 3.12.x versions are source-only (`.tar.xz`) and cannot be installed on Windows.
2. **VS Code** with the **Python** extension.
3. **Git** (you have it).
4. **Google Cloud CLI**, only for downloading data in section 5: https://docs.cloud.google.com/sdk/docs/install-sdk

## 3. Get the project and set it up (no cloning needed)

1. Download `sladro-backend.zip`, right-click it, **Extract All**, and put the folder somewhere like `Desktop\Projects`. You now have a folder called `sladro-backend`. Keep it separate from your Next.js project.
2. Open VS Code, choose **File, Open Folder**, and pick `sladro-backend`.
3. Open the terminal with **Terminal, New Terminal** (or Ctrl+`). It opens already inside the project folder. The prompt must end with `sladro-backend>`. If it doesn't, you are in the wrong folder, and nothing below will work.
4. Create the environment and install the packages:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   The prompt now starts with `(.venv)`. The install takes a minute or two. The "new release of pip" notice is harmless.
5. If VS Code asks about the new environment, click **Yes**. Otherwise press Ctrl+Shift+P, type **Python: Select Interpreter**, and choose the one inside `.venv`.

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and try again. Run every later command in this same VS Code terminal. A new terminal tab in VS Code activates `.venv` automatically.

## 4. Smoke test with fake data (5 minutes, no download)

```
python scripts/make_synthetic_sample.py
python scripts/prepare_data.py --raw data/synthetic/task_usage
python scripts/run_layer1.py --plot
python -m pytest -q
```

You should see a list of kept features, an R² table, and `2 passed`. On this fake data only CPU carries signal, so OOA-PSO keeps `cpu_rate` and drops the noise columns. That proves the code works before you spend time on big downloads.

## 5. Download the real data

Make the folder: `mkdir data\raw\task_usage`

**Option A: gcloud (recommended)**
```
gcloud storage ls gs://clusterdata-2011-2/
gcloud storage ls -l gs://clusterdata-2011-2/task_usage/ | Select-Object -First 5
```
The first command confirms the folder names, the second shows file sizes. Then download the **first 3 parts** to test:
```
0..2 | ForEach-Object { $n = "{0:D5}" -f $_; gcloud storage cp "gs://clusterdata-2011-2/task_usage/part-$n-of-00500.csv.gz" data\raw\task_usage\ }
```
If gcloud asks you to log in, run `gcloud auth login` (a free Google account is fine).

**Option B: no gcloud (untested fallback)**
```
Invoke-WebRequest "https://storage.googleapis.com/clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz" -OutFile data\raw\task_usage\part-00000-of-00500.csv.gz
```
If you get a 404, the names differ: use Option A's `ls` to see the real ones.

**How much?** Each part covers only about 1.5 hours of the 29 days. Start with 3 parts. Once everything works, go to about 10 (roughly 15 hours). Check disk space first and use the file sizes from `ls -l`. You do not need the full 41 GB.

## 6. Build the dataset and run Layer 1

```
python scripts/prepare_data.py --raw data/raw/task_usage --max-files 10
python scripts/run_layer1.py --plot
```

`prepare_data` adds up task records into one row per machine per 5-minute window, and the target is that machine's CPU in the next window. If it says `expected 20 columns`, compare `COLUMNS` in `src/sladro/data2011.py` with the schema document. If you run out of memory, add `--chunksize 200000`.

## 7. Read the output

- **Kept / Dropped:** the features OOA-PSO chose. These feed Layer 2.
- **R² table:** compare "all features" and "selected features" on **test** data. They should be close (the selected set is simpler at no real accuracy cost). If selected is far worse, increase `--iters` or lower `--beta`.
- **naive: next = current:** the simplest possible forecast. A real model should beat it. If it doesn't, that tells you the data has little to predict. It is not a bug.
- `outputs/layer1_result.json` holds everything, including `selected`, which Layer 2 will read. `outputs/layer1_result.png` is the convergence chart.

## 8. How the code maps to the paper

- `orthogonal.py`: orthogonal arrays (the "OOA" part), built from a Hadamard matrix.
- `ooa_pso.py`: Algorithm 1: orthogonal exploration, then binary PSO, with an epsilon stopping rule.
- `layer1.py`: the fitness function (validation accuracy minus a feature-count penalty), the time-ordered split, and the final test comparison.
- `data2011.py`: loading and aggregation of the trace.

## 9. Put it on GitHub (separate repo)

```
git init
git add .
git status
git commit -m "Layer 1: OOA-PSO feature selection"
```
`git status` must not list `data/` or `outputs/` (the `.gitignore` hides them). Create an empty repo on GitHub, add it as `origin`, and `git push -u origin main`.

## 10. Next

Layer 2 (CNN-LSTM) will read `outputs/layer1_result.json`, build 8-step windows per machine from the selected features, and predict the next window.
