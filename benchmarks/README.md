# Run the MiniRocket benchmarks

Compare original MiniRocket with exact joint PPV. Both use the same data and
classifier pipeline; the benchmark checks matching features and predictions.
See the [project overview](../README.md) or [published report](../results/joint_ppv_2026-10-26/report.md).

## Setup and dataset download

Run commands from the repository root. If needed, create an environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy numba scikit-learn scipy aeon threadpoolctl
```

Download the [UCR 2018 univariate archive in `.ts` format](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip),
the dataset source used for the published benchmark:

```bash
mkdir -p data
curl --fail --location --output data/UCR-univariate.zip \
  "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
unzip data/UCR-univariate.zip -d data
```

This creates `data/Univariate_ts/`. For matching software versions, see the
[published run's environment](../results/joint_ppv_2026-10-26/metadata.json).

## Run the full benchmark

**Only the input and a new output directory are required.**
Paths such as `./results/full-benchmark-run/` are example folders relative to the
repository root; choose your own new output folder name if preferred.

```bash
.venv/bin/python -B -m benchmarks.run \
  --data ./data/Univariate_ts/ \
  --output-dir ./results/full-benchmark-run/
```

Analysis runs automatically. Start with `./results/full-benchmark-run/report.md`.
Choose a new output directory for each run; completed results are not overwritten.

Defaults are all compatible datasets, requested feature counts `2500 5000 10000`
(actual `2436 4956 9996`), fitting seeds `0 1 2`, one thread and three transforms
per split. For the published archive, this covers 113 datasets and 2,034 cases.

## Optional: change experiment settings

For a quick single-dataset check, add only the settings you want to change:

```bash
.venv/bin/python -B -m benchmarks.run \
  --data ./data/Univariate_ts/ \
  --output-dir ./results/gunpoint-benchmark-run/ \
  --datasets GunPoint \
  --feature-counts 2500 \
  --seeds 0
```

For all optional settings and defaults:

```bash
.venv/bin/python -B -m benchmarks.run --help
```

These include threads, batch size and timing repetitions. Optional
`--dataset-label` and `--dataset-source-url` describe the input collection in
metadata; they do not select files or download data.

## Run analysis without benchmarking

For a folder containing `metadata.json` and `measurements.csv`
(for example, `./results/full-benchmark-run/`):

```bash
.venv/bin/python -B -m benchmarks.results ./results/full-benchmark-run/
```

This calculates tables, confidence intervals and the report from saved
measurements, without running models. An already-complete package is checked
rather than replaced. To explicitly verify it:

```bash
.venv/bin/python -B -m benchmarks.results ./results/full-benchmark-run/ --verify
```

To regenerate a completed report with the current generator, copy the two inputs
to a **new folder**, then analyze them:

```bash
mkdir ./results/analysis-only-run/
cp ./results/full-benchmark-run/metadata.json ./results/analysis-only-run/
cp ./results/full-benchmark-run/measurements.csv ./results/analysis-only-run/
.venv/bin/python -B -m benchmarks.results ./results/analysis-only-run/
```

## Output files

| File | Contents |
|---|---|
| `report.md` | Main findings, ratios, confidence intervals and limitations |
| `summary.csv` | Overall results by feature count |
| `per_dataset.csv` | Paired dataset comparisons |
| `measurements.csv` | Individual cases and timing repetitions |
| `metadata.json` | Dataset source, environment, settings and exclusions |
| `verification.json` | Correctness and file-integrity checks |

The runner executes the full classifier pipeline but times TRAIN and TEST
transforms separately. There is no mode that skips classification. Memory is the
whole worker's peak, not transform-only allocation. Timings depend on hardware
and software; avoid competing heavy workloads.

For a recorded failed run, wait for all workers to exit, then repeat the same
command with `--resume`. Keep its inputs and settings unchanged.
