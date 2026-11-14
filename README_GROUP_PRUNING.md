Name: Arya Sahane
Student ID: 25238553
This is my part of the code. (Group Pruning and Class-Stratified Biases for MiniRocket - three pieces built on top of MiniRocket and Arnav's faster version: a way to switch off whole groups of features so the transform does less work, a check of whether choosing those groups carefully actually matters, and a test of whether picking the threshold examples class by class helps)

# MiniRocket Extensions: Group Pruning, Controls and Class-Stratified Biases

This README covers my part of the group project. It adds three pieces to MiniRocket without
changing any of the original or Arnav's files.

| # | Module | One-line summary | Main file |
|---|---|---|---|
| 1 | **Group pruning** | Switches off whole groups of features chosen from the training data; the transform skips their work, and the features that remain are exactly the same as before | `code/minirocket_pruned.py` |
| 2 | **Controls** | Checks whether choosing the groups carefully beats choosing them at random, or beats simply building a smaller MiniRocket | `experiments/controls.py` |
| 3 | **Class-stratified biases** | Picks the training example that sets each group's thresholds so that every class gets a turn, instead of picking at random | `code/minirocket_stratified.py` |

---

## Contents

1. [Read this first: which part do you need?](#1-read-this-first-which-part-do-you-need)
2. [Quick start (5 commands)](#2-quick-start-5-commands)
3. [Background: MiniRocket in plain words](#3-background-minirocket-in-plain-words)
4. [Setup](#4-setup)
5. [Folder structure](#5-folder-structure)
6. [Module 1: Group pruning](#6-module-1-group-pruning)
7. [Module 2: Controls](#7-module-2-controls)
8. [Module 3: Class-stratified biases](#8-module-3-class-stratified-biases)
9. [Using your own data](#9-using-your-own-data)
10. [All options (command line and Python)](#10-all-options-command-line-and-python)
11. [Results summary](#11-results-summary)
12. [How the work was evaluated (and why it is honest)](#12-how-the-work-was-evaluated-and-why-it-is-honest)
13. [Tests](#13-tests)
14. [Relationship to Arnav's code](#14-relationship-to-arnavs-code)
15. [Troubleshooting / FAQ](#15-troubleshooting--faq)
16. [Glossary](#16-glossary)
17. [Limitations and possible next steps](#17-limitations-and-possible-next-steps)
18. [References](#18-references)
19. [Experiments: every run in detail](#19-experiments-every-run-in-detail)

---

## 1. Read this first: which part do you need?

| You are... | Read |
|---|---|
| A teammate who wants to **run** it | Sections 2, 4 and 15 |
| Someone who wants to **understand** the ideas | Sections 3, 6.1, 7.1, 8.1 |
| An examiner / reviewer checking the **results** | Sections 11, 12 and 19 |
| A developer who wants to **use or extend** the code | Sections 9, 10 and 14 |
| Someone who wants to **benchmark** it (timings) | Sections 4.5 and 19.9 |
| Presenting the project | Sections 7.3 (the controls table) and 11 |

**The 30-second version:**

* **Module 1** makes MiniRocket cheaper to run. Keeping half of the groups takes the test
  transform down to **0.36-0.46** of the original time, for a loss of only 0.05 to 0.27
  percentage points of accuracy. Keeping 10-20% of the groups costs about 1 to 2 points.
* **Module 2** shows where that saving comes from: groups picked **at random** do just as well as
  carefully chosen ones. The gain comes from doing less convolution work, not from finding
  "better" features. A pruned model is still 17-27% faster than a smaller fresh MiniRocket of
  the same accuracy.
* **Module 3** did not help: over all 113 datasets and 20 seeds, picking threshold examples class
  by class changed accuracy by **-0.002** points on average. I report it as a null result.

---

## 2. Quick start (5 commands)

Run these in a terminal, inside the project folder. On Windows I use Git Bash; the
`compat_windows` folder is only needed there.

```bash
cd Final_Minirocket
export PYTHONPATH="$PWD/code:$PWD/compat_windows"     # on macOS/Linux use "$PWD/code"
export NUMBA_NUM_THREADS=1

# 1. check everything works (73 tests OK, 1 skipped; the first run compiles code, so allow a minute)
python -B -m unittest discover -s tests

# 2. Module 1: prune two datasets with both scoring methods, keeping 100% / 50% / 20% of the groups
python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --datasets GunPoint ECG200 --seeds 0 --keep-fractions 1.0 0.5 0.2 --scorers ridge filter --bias-strategies original --output-dir results/quick_pruning

# 3. turn that run into tables (writes report.md and summary.csv into the same folder)
python -B -m experiments.summarise results/quick_pruning

# 4. Module 3: original vs uniform vs stratified thresholds on the same two datasets, 3 seeds
python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --datasets GunPoint ECG200 --seeds 0 1 2 --keep-fractions 1.0 --bias-strategies original uniform stratified --output-dir results/quick_bias

# 5. compare them (writes bias_analysis.md into the folder)
python -B -m experiments.analyse_bias results/quick_bias
```

The data is read from `data/Univariate_ts/`; these scripts never download anything (see
[Setup](#4-setup)). Each run needs a **new** output folder. Step 2 took about 16 seconds on my
laptop, including compilation.

To look at one dataset without writing any files, use the demo
([section 10.3](#103-experimentspruning_demopy-one-dataset-nothing-saved)):
`python -B -m experiments.pruning_demo --dataset GunPoint`.

---

## 3. Background: MiniRocket in plain words

**The task.** *Time series classification*: give a whole sequence of numbers one label, for
example "this hand movement is drawing a gun / pointing".

**How MiniRocket does it (Dempster, Schmidt & Webb, 2021):**

1. **84 fixed pattern detectors (kernels)**, each with 9 weights. Sliding one along the series
   gives a new series `C` that is high wherever the series looks like the pattern.
2. **Different widths (dilations).** Each kernel is also used "stretched out" to look for wider
   patterns (up to 32 widths).
3. **Thresholds (biases).** For every kernel and width, a few thresholds are taken from the
   kernel's output on **one random training series**.
4. **One number per threshold: PPV**, the fraction of `C` above the threshold. That gives
   about **10,000 features**.
5. **A simple linear classifier** (ridge regression) is trained on these features.

**What I call a group.** One kernel at one width, together with all of its thresholds, is a
**group**. It is a block of neighbouring features. On GunPoint there are 16 widths x 84 kernels =
**1,344 groups**, holding 9,996 features. Groups at small widths hold more thresholds (30 each at
the smallest width on GunPoint) than groups at large widths (3 each). Over the 113 datasets there
are about 1,598 groups on average.

**Arnav's contribution (already in this repository):** a faster way to compute PPV ("joint
PPV"). It reads `C` once for all thresholds instead of once per threshold, with identical results.

**Why groups are worth removing.** After Arnav's change, most of the remaining time goes into
the convolutions (one per group) and into the classifier (which grows with the number of
features):

| Part of the work | Grows with |
|---|---|
| Sliding one kernel at one width over the series | number of **groups** |
| Counting how often the output is above each threshold | number of features, but cheap after joint PPV |
| Training and using the classifier | number of **features** |

Removing a whole group saves all three. Removing a single threshold saves only the cheapest one.

**My contribution:** the three modules described below.

---

## 4. Setup

### 4.1 Requirements

* Python with `numpy`, `numba`, `scikit-learn`, `scipy`, `pandas`, `aeon` (and `matplotlib` for the
  controls figure)
* I used Python 3.14.7, numba 0.63.1, numpy 2.3.5, scikit-learn 1.8.0, scipy 1.17.1, aeon 1.5.0 on a
  Windows 11 laptop (AMD Ryzen 9 8940HX). The full package list is in `results/environment_freeze.txt`.

### 4.2 Data

Download the UCR archive as described in Arnav's [benchmark guide](./benchmarks/README.md) and
extract it to `data/Univariate_ts/`. 15 of the 128 datasets have series of different lengths or
missing values, and the loader skips them, so "all datasets" in this README means **113**.

### 4.3 Windows

The existing code was written for macOS and Linux, and three of its imports (`fcntl`, `resource`
and `os.sysconf`) do not exist on Windows. The `compat_windows/` folder holds small stand-ins for
them, so the existing tests run without editing any original file. Put it on `PYTHONPATH` on
Windows only. My own experiment scripts do not need it. Details are in
[section 19.10](#1910-windows-compatibility).

### 4.4 Check

```bash
python -B -m unittest discover -s tests               # 73 tests OK, 1 skipped (an Apple ARM64-only test)
python tools/generate_joint_ppv.py --check            # Arnav's generated kernels are current
```

### 4.5 Important: where to run commands from

Always run commands from the **project root**, with `NUMBA_NUM_THREADS=1` when you care about
timings. Close other heavy programs, and only compare timings taken on the same machine. How I
took the timings on Windows, step by step, and how that differs from Arnav's benchmark, is in
[section 19.9](#199-how-the-timings-were-taken-on-windows).

---

## 5. Folder structure

```
Final_Minirocket/                       <- Arnav's project (no existing file changed)
├── code/
│   ├── minirocket.py, minirocket_baseline.py, minirocket_joint_ppv_simd.py, ...   original + Arnav
│   ├── minirocket_pruned.py              Module 1: keep mask, pruned transform, the two ways of ranking groups
│   └── minirocket_stratified.py          Module 3: fitting with stratified (or uniform) threshold examples
├── tests/test_pruned.py                  13 tests for my code
├── tests/test_pruning_demo.py            5 tests for the demo (adapted from Shreyas's integration tests)
├── tests/test_analysis_scripts.py        4 tests for eta squared and the bootstrap interval
├── experiments/                          EVERYTHING BELOW IS MINE
│   ├── group_pruning.py                  runs the experiments (pruning and threshold choices)
│   ├── pruning_demo.py                   one dataset in memory, prints a table (adapted from Shreyas, section 14)
│   ├── summarise.py                      accuracy / time tables and break-even points
│   ├── controls.py, analyse_controls.py  Module 2: random groups and smaller models
│   ├── analyse_bias.py                   Module 3: stratified vs uniform comparison
│   ├── final_analysis.py                 the statistical tests over all runs
│   ├── equivalence.py                    bootstrap intervals for the accuracy changes (section 19.13)
│   ├── cross_platform.py                 macOS vs Windows, fit by fit (section 19.13)
│   ├── threshold_class_share.py          how much a threshold depends on the series' class (section 19.13)
│   ├── compare_to_paper.py               our MiniRocket vs the published numbers
│   ├── sanity_keepall.py                 keep-everything check against Arnav's transform
│   └── energy.py                         energy measurement (did not work on this laptop)
├── compat_windows/                       Windows stand-ins (section 4.3)
├── README_GROUP_PRUNING.md               this file
└── results/                              one folder per run (CSV, report, metadata) + logs
```

**No-touch rule:** none of the original or Arnav's files are modified. My code only imports from
them. I wrote a separate experiment runner rather than extending Arnav's benchmark, because his
benchmark is built to compare exactly two versions (original and joint PPV) and checks that their
accuracy is identical, which pruning deliberately changes.

---

## 6. Module 1: Group pruning

### 6.1 The idea

MiniRocket computes about 1,600 groups for every series, but it may not need all of them. If I
switch a group off, the transform skips its convolution and its thresholds, and the classifier gets
fewer inputs. The question is how much accuracy that costs and how much time it saves.

One thing matters a lot here: **the features that remain must be exactly the same as before.**
Then any change in accuracy comes from removing groups, not from accidentally computing something
different. I check this on every run by comparing the raw bits of the remaining features with the
same columns of the full model.

**Worked example** (GunPoint, seed 0, from my main run):

| Model | Groups | Features | Test transform time | Accuracy |
|---|---:|---:|---:|---:|
| Full (joint PPV) | 1,344 | 9,996 | 0.0231 s | 99.33% |
| Keep 50%, ridge | 672 | 5,496 | 0.0135 s | 99.33% |
| Keep 50%, filter | 672 | 7,271 | 0.0169 s | 100.00% |
| Keep 10%, ridge | 134 | 975 | 0.0032 s | 99.33% |

Half the groups, a bit more than half the time, and the same accuracy. The two scorers keep
different groups, so they keep different numbers of features (see 6.2).

### 6.2 How it works (`minirocket_pruned.py`)

1. Fit MiniRocket **once** on the full set of groups, as normal.
2. Transform the **training** series with all groups.
3. Rank the groups using the training data only, with one of two methods:
   * **`ridge`**: train the classifier, give each group a score from the size of its classifier
     weights (divided by the square root of the group's size, so big groups do not win just by being
     big), drop the weakest 20% of the remaining groups, and repeat until the target is reached.
   * **`filter`**: a quick statistical score (ANOVA F) for each feature, averaged per group. No
     classifier is trained, so it is much cheaper.
4. Keep the best groups (a boolean "keep mask" over widths x kernels), and retrain the classifier on
   their columns only.
5. At test time, only the kept groups are computed.

**Three rules keep the remaining features exact:**

* **Fit once and never refit for fewer groups.** Each threshold depends on the feature's position in
  the full layout, so refitting moves it.
* **Never renumber the groups.** MiniRocket pads every other group depending on its width number
  plus kernel number. In a small test, renumbering after dropping one width changed 1,931 of 2,016
  feature values.
* **Use training data only** for ranking and retraining.

**The cost of choosing.** Pruning needs the full model first, so training gets *more* expensive and
only prediction gets cheaper. I report a **break-even point**: how many test series you need to
classify before the cheaper predictions pay back the extra training. A break-even of 0 means the
pruned model was already cheaper to train, because its smaller classifier saves more than the
scoring costs.

### 6.3 How the design was decided

| Step | What was tried | What happened | Decision |
|---|---|---|---|
| Pilot | 5 datasets, 2 seeds, both scorers | ridge selection took 0.1-2.3 s per case; nothing broke | full run at the planned settings |
| Main run | 113 datasets, 3 seeds, drop 20% per round | half the groups: -0.05 (filter) to -0.27 (ridge) points, 0.36-0.46 of the time | kept as the main result |
| Faster run | same datasets, 1 seed, drop 30% per round | same picture; ridge breaks even sooner (fewer rounds) | kept as a robustness check |
| Comparing scorers | filter vs ridge at every level | filter was never worse, and is far cheaper to compute | filter is my default |
| Controls | random groups and smaller fresh models (Module 2) | random groups match both scorers | "the saving is the skipped work" |
| More features | 25,000 features, 10 datasets, 1 seed | ridge lost nothing at 20% of the groups | indication only |

### 6.4 How to run

```bash
# the main run (113 datasets, 3 seeds); about 9 hours on one core. Add --resume to continue after a stop.
python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir results/group_pruning_full --bias-strategies original --scorers ridge filter
python -B -m experiments.summarise results/group_pruning_full
python -B -m experiments.final_analysis      # the statistical tests over all saved runs
```

The command for every other result folder is in [section 19.12](#1912-commands-for-every-result-folder).

### 6.5 What you will see

`report.md` and `summary.csv` in the run folder, with one row per scorer and keep level. The main
run (113 datasets, 3 seeds; full model 86.08%) gives:

| Scorer | Groups kept | Features | Accuracy | Change | Time vs original | Break-even (test series) |
|---|---:|---:|---:|---:|---:|---:|
| full, joint PPV | 100% | 9,996 | 86.08% | 0 | 0.811 | - |
| filter | 50% | 5,555 | 86.03% | -0.05 | 0.463 | 25 |
| filter | 20% | 2,455 | 85.29% | -0.79 | 0.232 | 0 |
| filter | 10% | 1,268 | 84.65% | -1.43 | 0.142 | 0 |
| ridge | 50% | 4,527 | 85.81% | -0.27 | 0.362 | 1,019 |
| ridge | 20% | 1,654 | 84.78% | -1.30 | 0.172 | 993 |
| ridge | 10% | 784 | 83.85% | -2.23 | 0.112 | 1,068 |

| Column | Meaning |
|---|---|
| Groups kept / Features | how many groups survive, and how many features they hold |
| Accuracy | mean test accuracy over datasets (each averaged over seeds) |
| Change | accuracy change against the full model, in percentage points |
| Time vs original | test transform time (joint PPV) divided by the full model's time with the original transform, same laptop |
| Break-even | test series needed before the pruned model's extra training cost is paid back |

The full table (75% and 30% too, p-values, intervals) is in [section 19.2](#192-group-pruning-main-run).

### 6.6 Use it in Python

```python
import numpy as np
from minirocket_stratified import fit
from minirocket_pruned import transform, keep_all, prune_groups, kept_columns
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parameters = fit(X_train, y_train, strategy='uniform', random_state=0)     # or 'stratified' (Module 3)
full_train = transform(X_train, parameters, keep_all(parameters))           # all groups, training data

keep, history = prune_groups(full_train, y_train, parameters, keep_fraction=0.5, scorer='filter')
columns = kept_columns(parameters, keep)

classifier = make_pipeline(StandardScaler(),
                           RidgeClassifierCV(alphas=len(X_train) * np.logspace(-3, 3, 10)))
classifier.fit(full_train[:, columns], y_train)

X_test_features = transform(X_test, parameters, keep)                       # only the kept groups
predictions = classifier.predict(X_test_features)
```

`transform(..., implementation='original')` uses the original counting (one pass per threshold)
with the same keep mask.

### 6.7 Good to know

* "Keep 50%" means 50% of the **groups**, not of the features. Groups have different sizes, so the
  two scorers keep different numbers of features at the same level.
* Ridge selection retrains the classifier several times on a large matrix, so on big datasets it is
  slow (Crop took 18-24 minutes and ElectricDevices about 45 minutes per seed).
* The pruned and full models share their fitted parameters, so you can try several keep levels
  without refitting.

---

## 7. Module 2: Controls

### 7.1 The idea

Pruning is faster partly by simple arithmetic: fewer groups means fewer convolutions. So "pruning
is faster" on its own proves nothing about the scoring methods. The real question is whether
**choosing** the groups beats a naive way of shrinking the model. I compared two naive options at
every keep level, on the same datasets and seeds as the main run:

| Control | What it is |
|---|---|
| **Random groups** | the same number of groups, picked at random (no scoring at all) |
| **Smaller model** | a fresh MiniRocket built with fewer features from the start (same feature count as the pruned model), no pruning |

### 7.2 How to run

```bash
# needs the main run (results/group_pruning_full/) first
python -B -m experiments.controls --data ./data/Univariate_ts/ --output-dir results/controls
python -B -m experiments.analyse_controls
```

It writes `results/controls/controls_analysis.md` and the accuracy-vs-time figure
`results/controls/pareto.png`.

### 7.3 What you will see

Accuracy (%) and, in brackets, test transform time relative to the full model timed in the same
session. 113 datasets, 3 seeds; the full model scores 86.08%:

| Groups kept | ridge | filter | random groups | smaller model (filter-sized) |
|---:|---|---|---|---|
| 75% | 85.97 (0.672) | 86.07 (0.811) | 85.97 (0.772) | 86.00 (0.861) |
| 50% | 85.81 (0.447) | 86.03 (0.570) | 85.89 (0.541) | 86.02 (0.683) |
| 30% | 85.29 (0.288) | 85.68 (0.384) | 85.51 (0.363) | 85.68 (0.515) |
| 20% | 84.78 (0.212) | 85.29 (0.285) | 85.20 (0.267) | 85.43 (0.393) |
| 10% | 83.85 (0.138) | 84.65 (0.175) | 84.58 (0.175) | 84.86 (0.216) |

**What this means:**

1. **Choosing the groups does not improve accuracy.** Filter and random groups are never
   significantly different (all p > 0.15). Ridge is no better than random either, and at heavy
   pruning it is slightly worse (-0.42 points at 20%, -0.74 at 10%, p about 0.04), though this does
   not survive a correction for running 20 tests.
2. **The real advantage of pruning is time.** A smaller fresh model spreads its features over almost
   all groups, so it still pays for almost every convolution. Pruning keeps fewer groups. At the
   same accuracy, filter-pruned models are **17% faster** at 50% (86.03% at 0.570 vs 86.02% at 0.683)
   and **27% faster** at 20% (85.29% at 0.285 vs 85.43% at 0.393, a non-significant -0.14 points).
3. **Random groups are a free baseline.** They need no scoring (break-even 0) and land within
   0.1-0.2 points of the best chosen groups.

My reading: MiniRocket's 10,000 features are highly redundant. It matters little *which* groups
remain, as long as enough of them do.

The full table with the ridge-sized smaller model and all 20 paired tests is in
[section 19.4](#194-controls-in-full).

### 7.4 Good to know

* The random-group models pass the same bit-for-bit check as the pruned ones (0 failures).
* Ridge and filter timings come from the main run's session and the controls from their own
  session. Each is relative to its own full model, so the ratios can be compared, but they were
  not measured in one process.

---

## 8. Module 3: Class-stratified biases

### 8.1 The idea

MiniRocket sets each group's thresholds from **one random training series**. On a dataset where
one class has 80% of the training series, about 80% of the groups get their thresholds from that
class. My hypothesis was that the minority classes might then be described less well, and that
letting the classes take turns would help accuracy, or at least make results steadier from one
random seed to the next.

Analogy: polling. Asking one random person per question over-represents the biggest group; asking
each group in turn is fairer.

**Worked example** (a real draw from my code): 10 groups, training labels 8 x A and 2 x B.

| Method | Which training series sets each group's thresholds | Classes |
|---|---|---|
| uniform (random) | 8, 6, 5, 2, 3, 0, 0, 0, 1, 8 | B A A A A A A A A B (8 A, 2 B) |
| stratified | 0, 9, 7, 9, 5, 9, 9, 6, 8, 6 | A B A B A B B A B A (5 A, 5 B) |

Uniform mirrors the 80/20 imbalance; stratified gives each class half of the groups.

### 8.2 How it works (`minirocket_stratified.py`)

1. Classes are visited in a random order, round after round, so every class supplies about the same
   number of groups.
2. Within its class, the series is picked at random.
3. The thresholds are then computed exactly as in the original code. Everything else (widths,
   number of thresholds per width) comes from the original functions, so the result works with
   every transform in the repository.

I compare three versions:

| Version | How the series is picked | Why it is there |
|---|---|---|
| `original` | the authors' code and random number generator | the reference |
| `uniform` | at random from all training series, through **my** code | the fair comparison |
| `stratified` | class by class, through my code | the idea being tested |

The original code uses a different random number generator from mine. So the comparison that
counts is **stratified vs uniform** (same code, only the picking differs). Original vs uniform
shows how much the generator alone changes things.

I also split the datasets into three groups by a fixed rule, decided before running:
**imbalanced** (smallest class under 20% of the training set: 57 datasets), **small** (under 100
training series: 26) and **rest** (30).

### 8.3 How to run

```bash
# all 113 datasets, 20 seeds, 3 versions; about 9 hours on one core (supports --resume)
python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir results/group_pruning_bias_all_20seeds --bias-strategies original uniform stratified --keep-fractions 1.0 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
python -B -m experiments.analyse_bias results/group_pruning_bias_all_20seeds --subsets results/bias_study_groups_all113.csv
```

### 8.4 What you will see

`bias_analysis.md` has one table per comparison and the mean and spread of every dataset. The
result over all 113 datasets and 20 seeds (stratified minus uniform, in percentage points):

| Datasets | Number | Mean change | 95% range | Better / equal / worse | p |
|---|---:|---:|---|---|---:|
| all | 113 | -0.002 | -0.044 to +0.043 | 43 / 23 / 47 | 0.38 |
| imbalanced | 57 | -0.035 | -0.086 to +0.018 | 17 / 9 / 31 | 0.054 |
| small | 26 | +0.060 | -0.037 to +0.178 | 7 / 11 / 8 | 0.93 |
| rest | 30 | +0.007 | -0.076 to +0.082 | 19 / 3 / 8 | 0.40 |

**What this means:**

* There is no effect. Both versions average 86.10%, and any real average effect is smaller than about
  0.05 points (about a tenth of the normal seed-to-seed variation of 0.49 points).
* Results did not get steadier: the spread across seeds is 0.503 points for stratified against
  0.485 for uniform.
* On 87 of the 113 datasets the difference is smaller than one test series.
* The original generator alone makes no real difference either (+0.030 points, p = 0.18).

**One result that did not hold up.** I first ran a smaller version (30 datasets, 10 seeds). There,
stratified looked slightly *worse* on the 10 imbalanced datasets (-0.097 points, p = 0.012). With 20
seeds the same 10 datasets give -0.066 (p = 0.25), and all 57 imbalanced datasets give -0.035
(p = 0.054). The first result was noise. A small p-value on 10 datasets is weak evidence, and more
seeds and datasets are the way to test it. The two runs give identical results on the 900 cases they
share.

### 8.5 Why it probably does not work

These are my best explanations, but I did not test them:

* A threshold is taken from a whole series' output, and for most kernels that output looks similar
  across classes, so it matters little whose series is used.
* With about 1,600 groups and about six thresholds each, the classifier can work around any one
  badly placed threshold.
* Picking the threshold series is MiniRocket's only random step. Stratifying did not change the
  spread across seeds, so that spread comes from *which individual series* is picked, not from
  *which classes* are covered.

### 8.6 Use it in Python

```python
from minirocket_stratified import fit
parameters = fit(X_train, y_train, num_features=10_000, strategy='stratified', random_state=0)
# strategy='uniform' gives the fair comparison; the result works with every transform in the repo
```

### 8.7 Good to know

* `examples_per_group=m` pools m training series per group before taking the thresholds. I only
  tested m = 1.
* I also combined stratified thresholds with pruning (30 datasets, 5 seeds). Nothing survived a
  correction for the 24 comparisons; see [section 19.6](#196-stratified-thresholds-combined-with-pruning).

---

## 9. Using your own data

All modules need:

* `X`: a 2-D NumPy array of shape **(number of series, series length)**, `float32` (all series the
  same length, no missing values)
* `y`: a 1-D array of labels, one per series

**From a CSV file** where each row is one series and the first column is the label:

```python
import sys, numpy as np
sys.path.insert(0, "code")                       # run from the project root
from minirocket_stratified import fit
from minirocket_pruned import transform, keep_all, prune_groups

data = np.loadtxt("my_train.csv", delimiter=",")
y_train = data[:, 0]
X_train = np.ascontiguousarray(data[:, 1:], dtype=np.float32)

parameters = fit(X_train, y_train, strategy='uniform', random_state=0)
full_train = transform(X_train, parameters, keep_all(parameters))
keep, history = prune_groups(full_train, y_train, parameters, keep_fraction=0.5, scorer='filter')
```

Then continue as in [section 6.6](#66-use-it-in-python).

**Any UCR dataset by name:** pass `--datasets NameOfDataset` to the experiment runner. It must be
in `data/Univariate_ts/`.

---

## 10. All options (command line and Python)

### 10.1 `experiments/group_pruning.py` (Modules 1 and 3)

| Option | Default | Meaning |
|---|---|---|
| `--data PATH` | required | folder with the UCR archive |
| `--output-dir PATH` | required | a **new** folder for the results |
| `--datasets A B ...` | all | dataset names |
| `--seeds 0 1 2 ...` | 0 1 2 | random seeds (results are averaged) |
| `--bias-strategies ...` | original uniform stratified | which threshold versions to run |
| `--keep-fractions ...` | 1.0 0.75 0.5 0.3 0.2 0.1 | share of **groups** to keep |
| `--scorers ...` | ridge | `ridge` and/or `filter` |
| `--drop-fraction X` | 0.2 | share of groups ridge drops per round |
| `--feature-count N` | 10000 | features to ask MiniRocket for |
| `--examples-per-group M` | 1 | training series pooled per group's thresholds |
| `--transform-repetitions N` | 3 | how often each test transform is timed |
| `--resume` | off | continue a stopped run in the same folder |

### 10.2 Other scripts

| Script | Options |
|---|---|
| `experiments/summarise.py` | the run folder |
| `experiments/analyse_bias.py` | the run folder, `--subsets` (dataset groups file), `--by-keep-fraction`, `--output` |
| `experiments/controls.py` | `--data`, `--output-dir`, `--datasets`, `--seeds` (0 1 2), `--keep-fractions` (0.75 to 0.1), `--resume` |
| `experiments/analyse_controls.py`, `final_analysis.py`, `compare_to_paper.py`, `equivalence.py` | read the saved results; no options needed |
| `experiments/cross_platform.py` | none to read the saved results; `--recompute DATA_PATH` also refits all 339 fits (about 20 minutes) |
| `experiments/threshold_class_share.py` | `--data`, `--datasets`, `--output` (resumable CSV), `--max-series` (100), `--groups-file` |

### 10.3 `experiments/pruning_demo.py` (one dataset, nothing saved)

The demo runs the same steps as the runner on one dataset, one seed, one scorer and one threshold
version, with the joint-PPV transform only. It prints a table, how many groups survive at each width,
and (for `uniform` or `stratified`) which class set the thresholds. It writes nothing to disk.

| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | required | one dataset from the local archive |
| `--data PATH` | `./data/Univariate_ts/` | folder with the UCR archive |
| `--scorer` | ridge | `ridge` or `filter` |
| `--bias-strategy` | original | `original`, `uniform` or `stratified` |
| `--seed N` | 0 | random seed |
| `--keep-fractions ...` | 1.0 0.75 0.5 0.3 0.2 0.1 | 1.0 is always added as the reference |
| `--feature-count N` | 10000 | features to ask MiniRocket for |
| `--repetitions N` | 3 | how often each test transform is timed (mean is shown) |

On GunPoint, seed 0 (ridge, original thresholds):

```
 keep      groups features  accuracy   change   time  size KB exact
 1.00  1344/1344      9996    99.33%   +0.00pp  1.00x      234 yes
 0.50   672/1344      5496    99.33%   +0.00pp  0.55x      129 yes
 0.10   134/1344       975    99.33%   +0.00pp  0.13x       23 yes
```

Two things to keep in mind when reading it. `time` is relative to the unpruned **joint-PPV** model in
the same session, so it is not the same baseline as the 0.36-0.46 of the original time in
section 11. `size KB` is **worked out** from the feature count (ridge weights plus the scaler's mean
and scale, 8 bytes each), not measured on disk.

### 10.4 Python functions

| Function | What it does |
|---|---|
| `minirocket_stratified.fit(X, y, num_features=10_000, examples_per_group=1, strategy='stratified', random_state=0)` | fits MiniRocket with uniform or stratified threshold examples |
| `minirocket_pruned.keep_all(parameters)` | a keep mask with every group switched on |
| `minirocket_pruned.transform(X, parameters, keep, implementation='joint')` | computes only the kept groups |
| `minirocket_pruned.kept_columns(parameters, keep)` | which columns of the full model the kept groups are |
| `minirocket_pruned.prune_groups(features, y, parameters, keep_fraction, scorer='ridge', drop_fraction=0.2)` | chooses the groups from training data; returns `(keep, history)` |

---

## 11. Results summary

All numbers come from the files in `results/`. Timings are from one Windows 11 laptop, with one
Numba thread. Accuracy is the mean over datasets, with each dataset averaged over its seeds first.

### 11.1 Did I reproduce MiniRocket first?

Yes. On the 109 datasets the paper also covers, my original MiniRocket scores **85.69%** against the
authors' **85.67%** on the same split (p = 0.58, so no systematic difference). Two datasets differ by
more than 2 points (BeetleFly +3.3, Computers +2.5), none by more than two of the paper's own
standard deviations. The paper's headline 87.24% is an average over 30 different splits, so it
cannot be compared with one split.

### 11.2 One-table summary

| Question | Answer |
|---|---|
| Are the pruned features exact? | yes: identical bit for bit in all 29,304 measured rows |
| What does keeping half the groups cost? | -0.05 (filter) to -0.27 (ridge) points, for 0.36-0.46 of the original time |
| What about keeping only 10-20%? | -0.8 to -2.2 points |
| When does choosing pay for itself? | filter: after about 25 test series; ridge: after about 1,000 |
| Does choosing groups beat random? | no |
| Pruned model vs a smaller fresh model? | same accuracy, 17-27% less transform time |
| Do stratified thresholds help? | no: -0.002 points over 113 datasets and 20 seeds |
| Energy? | not measured (the meter on this laptop did not respond to load) |

### 11.3 What I would recommend

1. **Use group pruning as a speed tool.** Half the groups gives roughly half the transform time for a
   loss you cannot detect (filter) or a very small one (ridge).
2. **Do not pay for ridge selection.** Random groups match it, and it needs about 1,000 predictions to
   pay back its training. Use filter as a cheap default, or random groups when you want no training
   cost at all.
3. **Prune rather than building a smaller MiniRocket** when speed matters (17-27% faster at the same
   accuracy).
4. **Expect a 1 to 2 point loss** below 20% of the groups.
5. **Skip stratified thresholds.** They do not change accuracy or make results steadier.

---

## 12. How the work was evaluated (and why it is honest)

* **No test-set tuning.** Groups are ranked and the classifier is retrained on training data only. The
  keep levels (100, 75, 50, 30, 20, 10%) were fixed before any run, and I report all of them.
* **Exactness on every row.** Every run compares the raw bits of the kept features with the full
  model's columns. 0 differences in 29,304 rows, and in every random mask of the controls.
* **The right comparison.** "Pruning is faster" is not enough on its own, so I compared against
  random groups and smaller fresh models (Module 2). Module 3 has its own fair comparison (uniform).
* **Statistics.** Each dataset counts once (seeds averaged first). Differences are tested with the
  Wilcoxon signed-rank test over datasets. When I ran many tests at once (20 control tests, 16 and
  24 tests for stratified thresholds), I also applied the Holm correction, and say so wherever it
  matters.
* **Failed ideas are reported**, not hidden: stratified thresholds, ridge selection, and a stratified
  result that did not hold up with more data.
* **Honest corrections.** An earlier version of this README said "filter is more accurate than ridge
  at every level". The final tests showed that this is only the same sign at every level and
  significant only at 10%, so I changed the wording.
* **One lesson from the code itself.** I once tried to tidy the pruned transform by merging its two
  kernels. The tests on small synthetic series still passed, but on real data about 43% of the
  stratified thresholds changed in their last bit and the transform became 1.5-2 times slower. Numba's
  fast-math mode makes rounding depend on how loops are laid out. I restored the original layout, and
  the two kernels stay duplicated on purpose, with a comment saying why.

---

## 13. Tests

```bash
python -B -m unittest discover -s tests -p "test_pruned.py" -v      # my 13 tests
python -B -m unittest discover -s tests -p "test_pruning_demo.py"   # 5 demo tests
python -B -m unittest discover -s tests                             # whole suite: 73 OK, 1 skipped
```

`tests/test_pruned.py`:

| Test | What it checks |
|---|---|
| layout is contiguous and complete | every group is one block of neighbouring features, widths first, and no feature is missed |
| keep-all matches original and joint bit for bit | with every group switched on, the output is identical to the original and joint-PPV transforms |
| pruned output is an exact column slice | the kept features equal the full model's kept columns (the exactness promise) |
| inputs are validated, parameters untouched | bad inputs are rejected, and pruning never changes the fitted parameters |
| direct convolution oracle | a slow version written from the definition agrees, including the padding rule |
| renumbering would change padding | shows why groups are never renumbered |
| group scores: shape, empty slots, size correction | the ridge score is per group and divided by the square root of the group size |
| both scorers keep the planted group | a group made to be useful survives both ridge and filter |
| pruning is repeatable and respects a starting mask | same input, same result; already-removed groups stay removed |
| stratified table is class-balanced, repeatable, seeded | Module 3 gives each class a fair share |
| stratified fit returns original-compatible parameters | the result works with every existing transform |
| single-example loop reproduces the original fit | given the same picks, my threshold code gives the original thresholds |
| pooled quantiles match the oracle | the `examples_per_group > 1` option computes what it says |

`tests/test_pruning_demo.py` (adapted from Shreyas's tests, section 14):

| Test | What it checks |
|---|---|
| rows are consistent and exact | full model first, fewer groups at each level, every row exact, size formula right |
| every scorer and bias strategy runs | all 6 combinations run and stay exact; an unknown scorer is rejected |
| matches the runner | same features (SHA-256), feature count and accuracy as `experiments/group_pruning.py` |
| threshold sources balanced when stratified | every class sets the thresholds of the same number of groups, give or take one |
| published results are readable | `results/group_pruning_full/summary.csv` still gives the 86.08% headline |

On top of the tests, the keep-everything transform gives output identical to Arnav's transform and is
0.7% (GunPoint) to 3% (Wafer) slower (`experiments/sanity_keepall.py`).

---

## 14. Relationship to Arnav's code

```
                 Arnav's and the original code (unchanged, only imported)       My code
                 ---------------------------------------------------------       -------
seeded fit       code/minirocket_baseline.fit_univariate          ───►  the 'original' threshold version
widths, levels   code/minirocket._fit_dilations / _quantiles      ───►  minirocket_stratified.fit
fast counting    code/_joint_ppv_simd_kernels._pool_joint         ───►  minirocket_pruned.transform (joint)
exactness check  code/minirocket_baseline.transform_univariate    ───►  every experiment row
classifier       benchmarks/common.make_classifier                ───►  every model and experiment
data loading     benchmarks/common (dataset loaders)              ───►  experiments/group_pruning.py
```

| | Arnav's version | With my modules |
|---|---|---|
| Goal | the same results, faster | less work: fewer groups, a smaller classifier |
| Features computed | all of them | only the kept groups (identical values) |
| Test transform time (Windows laptop) | 0.811 of the original | 0.463 (filter, half the groups) to 0.112 (ridge, 10%) |
| Accuracy (113 datasets) | 86.08% | 86.03% (filter, half the groups) |
| Thresholds | one random training series per group | optional: class by class (no effect) |

Arnav measured joint PPV on an Apple ARM64 laptop with his own benchmark suite (46.9% less test time
at 9,996 features). On my Windows laptop, with my runner, the same code reached 0.811 of the original
time. So the speed-up depends on the hardware, and the two numbers come from different machines and
different measuring scripts ([section 19.9](#199-how-the-timings-were-taken-on-windows)).

**In the group project.** Shreyas Shinde put my files into the team repository
([minirocket-group-project](https://github.com/er-shreyas-shinde/minirocket-group-project), folder
`Arya's Additions`). My code there came from this repository; the three scripts in section 19.13 were
added here later and are not in the team repository yet. He added a launcher, `run_arya.py`
(`python "Arya's Additions/run_arya.py" test`, `demo` or `experiment ...`), an integration layer
(`pruning_service.py`) and two pages in the group's Streamlit app. `experiments/pruning_demo.py`
and `tests/test_pruning_demo.py` in this repository are my adaptations of his integration layer and
its tests. I changed them to read the local archive, reuse the runner's functions, and report the mean
time like the runner does.

---

## 15. Troubleshooting / FAQ

| Problem | Fix |
|---|---|
| `No module named 'minirocket_pruned'` (or `minirocket`) | `PYTHONPATH` is not set; run the `export` line from section 2 in the project root |
| `No module named 'fcntl'` / `'resource'` on Windows | add `compat_windows` to `PYTHONPATH` |
| "... exists; choose a new folder or pass --resume" | each run needs a new output folder; use `--resume` only to continue the same run |
| A long run was stopped | run the same command again with `--resume`; at most the case that was running is lost |
| First run is slow | Numba compiles the code once; later runs use the cache |
| A dataset is skipped | it has series of different lengths or missing values (15 of 128 datasets) |
| "Features ... are constant" warnings | harmless; they come from the filter scorer, which gives those features a score of 0 |
| No such file in `data/Univariate_ts/` | the scripts never download; extract the archive there first (section 4.2) |
| My timings differ from this README | compare ratios on one machine, not seconds; timings always differ between machines |
| I changed `minirocket_pruned.py` and the results changed slightly | see the lesson in section 12; keep the original loop layout and clear the Numba cache |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Kernel | a small pattern detector (9 weights) slid along the series |
| Dilation (width) | how stretched the kernel is |
| Threshold (bias) | a cut-off for the kernel's output, taken from a training series |
| PPV | the fraction of the kernel's output above a threshold; one feature |
| Joint PPV | Arnav's faster way to compute PPV, all thresholds in one pass |
| Group | one kernel at one width with all its thresholds; a block of features |
| Keep mask | the list of which groups are switched on |
| Pruning | switching groups off |
| Ridge / filter | my two ways of ranking groups: by classifier weights / by a quick statistical score |
| Break-even | test series needed before the cheaper predictions pay back the extra training |
| Control | a naive alternative used to check that a method really adds something |
| Stratified / uniform | picking the threshold series class by class / at random |
| Seed | a number that fixes the randomness, so results can be repeated |
| Wilcoxon test | a test of whether paired differences over datasets are systematic or chance |
| Holm correction | makes p-values stricter when many tests are run at once |
| Percentage point (pp) | a difference between two accuracies, e.g. 86.08% to 86.03% is -0.05 points |

---

## 17. Limitations and possible next steps

**Limitations**

* One laptop and one operating system, so the timings are only comparable within this machine.
* Three seeds for pruning (20 for stratified thresholds) and one train/test split per dataset, not
  the paper's 30.
* Most p-values are not corrected for multiple tests; I only corrected where many tests were run at
  once.
* The 25,000-feature run and the faster run use one seed; the combined stratified-and-pruning study
  uses 30 datasets.
* Energy was not measured: the CodeCarbon reading on this laptop stayed the same whether the CPU was
  idle or busy ([section 19.11](#1911-energy-attempt)). Memory and model size on disk were not
  measured either; the demo's size column is worked out from the feature count.
* I did not analyse which widths and kernels survive pruning across datasets. The demo shows it for
  one dataset at a time.
* Univariate, equal-length series without missing values only.

**Next steps**

* Test stratified thresholds on small datasets under heavy pruning, with new data (the only hint left
  from the combined study).
* Try pooling several training series per group (`examples_per_group > 1`).
* Measure energy on a machine with working power counters.
* Save the keep masks of the full run and look at which widths and kernels survive across datasets.
* Compare with published ROCKET-pruning methods (S-ROCKET, POCKET, Detach-ROCKET).

---

## 18. References

* Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost)
  deterministic transform for time series classification. *Proceedings of KDD 2021*, 248-257.
* Salehinejad, H. et al. (2022). S-Rocket: Selective random convolution kernels for time series
  classification. arXiv:2203.03445.
* Chen, S. et al. (2024). POCKET: Pruning random convolution kernels for time series classification
  from a feature selection perspective. *Knowledge-Based Systems*. arXiv:2309.08499.
* Uribarri, G. et al. (2024). Detach-ROCKET: Sequential feature selection for time series
  classification with random convolutional kernels. *Data Mining and Knowledge Discovery*.
  arXiv:2309.14518.
* Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of
  Machine Learning Research*, 7, 1-30.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics*, 6(2), 65-70.

Pruning ROCKET-type models has been done before (the three papers above), so I do not claim pruning
is new. What this work adds is pruning whole MiniRocket groups so the convolution itself is skipped,
keeping the remaining features exact, measuring real transform time, reporting the cost of choosing
(break-even), and checking against random groups and smaller models. I did not find earlier work on
class-stratified thresholds in a few searches, but that does not prove there is none.

---

## 19. Experiments: every run in detail

This section is for anyone who wants the full numbers behind sections 6-8. Every table is copied
from the files in `results/`.

### 19.1 What was run

| Study | Planned | What I ran | Folder |
|---|---|---|---|
| Group pruning | all datasets, 3 seeds, drop 0.2 | **as planned**; stopped once after 53 cases and resumed with `--resume` (about 9 hours) | `group_pruning_full/` |
| Faster pruning run | (extra) | all datasets, 1 seed, drop 0.3 (about 1 h 20 min) | `group_pruning_full_drop30_1seed/` |
| Controls | (extra) | 113 datasets, 3 seeds, 5,424 rows | `controls/` |
| Stratified thresholds | all datasets, 20 seeds | **as planned**: 13,560 rows, about 9 hours, stopped once and resumed | `group_pruning_bias_all_20seeds/` |
| Stratified, first check | (extra) | 30 datasets, 10 seeds, 1,800 rows | `group_pruning_bias_30ds_10seeds/` |
| Stratified x pruning | all datasets, 5 seeds | 30 datasets, 5 seeds, ridge only, drop 0.3 (the full version would take about 10 hours) | `group_pruning_combined_30ds_5seeds/` |
| 25,000 features | 10 datasets | 10 datasets, 1 seed, keep 100/50/20% | `group_pruning_25k/` |
| Pilot | small check | 5 datasets, 2 seeds, 300 rows | `group_pruning_pilot/` |
| Energy | 5 datasets | **skipped**: no valid power reading on this laptop | - |
| Bootstrap intervals | (extra, for the paper) | reads the saved runs, no fitting | `equivalence.md` |
| macOS vs Windows | (extra, for the paper) | 339 fits compared; all 339 refitted on Windows (about 20 min) | `cross_platform.md`, `cross_platform_stages.csv` |
| Class share of thresholds | (extra, for the paper) | 112 datasets, up to 100 training series each | `threshold_class_share.csv` |

**Skipped datasets.** 15 of the 128: 12 with series of different lengths (AllGestureWiimoteX/Y/Z,
GestureMidAirD1-3, GesturePebbleZ1/Z2, MelbournePedestrian, PLAID, PickupGestureWiimoteZ,
ShakeGestureWiimoteZ) and 3 with missing values (DodgerLoopDay/Game/Weekend). None was dropped for
time or memory.

**The 30-dataset subset** (used by the first stratified check and the combined study) was fixed by
rule before running: every dataset is labelled imbalanced, small or rest (section 8.2), then 10 are
taken from each label, evenly spaced in alphabetical order. The list is in
`results/bias_study_subset.csv`; the labels of all 113 datasets are in
`results/bias_study_groups_all113.csv`; dataset sizes and class shares are in `results/dataset_stats.csv`.

**Exactness by run:** pilot 300, main 7,458, faster run 2,486, first stratified check 1,800,
stratified all-113 13,560, combined 3,600, 25k 100 rows: 29,304 rows with 0 differences (the 1,800
rows shared by the two stratified runs are counted in both).

### 19.2 Group pruning, main run

113 datasets, 3 seeds, original thresholds, drop 0.2 per round. Time = joint-PPV test transform
divided by the full model with the original transform. "p" is a Wilcoxon test of the accuracy change
against the full model.

| Scorer | Groups kept | Features | Accuracy | Change | p | Time vs original | Break-even |
|---|---:|---:|---:|---:|---:|---:|---:|
| full, original | 100% (1,598) | 9,996 | 86.08% | 0 | | 1.000 | - |
| full, joint PPV | 100% | 9,996 | 86.08% | 0 | | 0.811 | - |
| filter | 75% | 7,946 | 86.07 | -0.01 | 0.91 | 0.657 | 233 |
| filter | 50% | 5,555 | 86.03 | -0.05 | 0.50 | 0.463 | 25 |
| filter | 30% | 3,538 | 85.68 | -0.40 | 0.048 | 0.311 | 0 |
| filter | 20% | 2,455 | 85.29 | -0.79 | 0.002 | 0.232 | 0 |
| filter | 10% | 1,268 | 84.65 | -1.43 | <0.0001 | 0.142 | 0 |
| ridge | 75% | 7,133 | 85.97 | -0.11 | 0.23 | 0.545 | 1,347 |
| ridge | 50% | 4,527 | 85.81 | -0.27 | 0.006 | 0.362 | 1,019 |
| ridge | 30% | 2,582 | 85.29 | -0.79 | <0.0001 | 0.233 | 1,017 |
| ridge | 20% | 1,654 | 84.78 | -1.30 | <0.0001 | 0.172 | 993 |
| ridge | 10% | 784 | 83.85 | -2.23 | <0.0001 | 0.112 | 1,068 |

* Time falls roughly in line with the groups kept, and joint PPV's saving carries over.
* Mild pruning costs nothing measurable (filter at 75% and 50%, ridge at 75%). Ridge at 50% costs 0.27
  points, small but real (p = 0.006).
* Ridge selection costs about 4-5 times the full model's training, which is why it needs about 1,000
  test series to pay back. Filter is nearly free, and at 30% or fewer groups it is cheaper to train
  than the full model.
* Filter minus ridge: +0.09, +0.23, +0.39, +0.51, +0.80 points at 75, 50, 30, 20, 10%. The sign is the
  same everywhere, but only 10% is significant (p = 0.038, not corrected), and filter keeps more
  features at the same group count. So "filter is at least as good as ridge" is supported; "filter is
  better" is only a hint.
* More detail (same-implementation ratios, bootstrap intervals, training-cost ratios) is in
  `results/group_pruning_full/report.md`.

### 19.3 Faster pruning run (1 seed, drop 0.3)

Baseline 86.07%; joint PPV alone 0.806 of the original.

| Scorer | Groups kept | Features | Accuracy % | Change | Time vs original | Median break-even |
|---|---:|---:|---:|---:|---:|---:|
| filter | 75% | 7,952 | 86.10 | +0.03 | 0.654 | 237 |
| filter | 50% | 5,558 | 85.94 | -0.14 | 0.460 | 23 |
| filter | 30% | 3,536 | 85.51 | -0.57 | 0.306 | 0 |
| filter | 20% | 2,457 | 85.20 | -0.88 | 0.230 | 0 |
| filter | 10% | 1,271 | 84.51 | -1.57 | 0.140 | 0 |
| ridge | 75% | 7,060 | 85.92 | -0.15 | 0.530 | 656 |
| ridge | 50% | 4,508 | 85.70 | -0.38 | 0.359 | 530 |
| ridge | 30% | 2,575 | 85.14 | -0.93 | 0.230 | 594 |
| ridge | 20% | 1,654 | 84.53 | -1.55 | 0.170 | 578 |
| ridge | 10% | 776 | 83.72 | -2.35 | 0.112 | 642 |

Same conclusions. Ridge pays back sooner here (about 530-650 series) because dropping 30% per round
means fewer rounds. The runs differ in seeds and session too, so this is not a controlled comparison.

### 19.4 Controls in full

Accuracy % and time ratio (relative to the full model timed in the same session):

| Groups kept | ridge | filter | random groups | smaller (ridge-sized) | smaller (filter-sized) |
|---:|---|---|---|---|---|
| 75% | 85.97 (0.672) | 86.07 (0.811) | 85.97 (0.772) | 85.98 (0.789) | 86.00 (0.861) |
| 50% | 85.81 (0.447) | 86.03 (0.570) | 85.89 (0.541) | 85.82 (0.603) | 86.02 (0.683) |
| 30% | 85.29 (0.288) | 85.68 (0.384) | 85.51 (0.363) | 85.64 (0.426) | 85.68 (0.515) |
| 20% | 84.78 (0.212) | 85.29 (0.285) | 85.20 (0.267) | 85.20 (0.294) | 85.43 (0.393) |
| 10% | 83.85 (0.138) | 84.65 (0.175) | 84.58 (0.175) | 84.47 (0.148) | 84.86 (0.216) |

Feature counts differ between columns (at 50%: ridge 4,527, filter 5,555, random 5,017); each smaller
model has the feature count of the pruned model it is compared with. A smaller fresh model at 50% of
the features still uses 1,597 groups with 3.5 features each, while pruning keeps about half the groups
with 5.7-7 features each.

Paired tests over 113 datasets (accuracy difference in points, p in brackets; 20 tests, not corrected):

| Groups kept | ridge - random | filter - random | ridge - smaller | filter - smaller |
|---:|---|---|---|---|
| 75% | +0.00 (0.27) | +0.09 (0.16) | -0.01 (0.64) | +0.06 (0.11) |
| 50% | -0.09 (0.75) | +0.14 (0.22) | -0.02 (0.70) | +0.01 (0.86) |
| 30% | -0.22 (0.32) | +0.17 (0.24) | -0.35 (0.12) | -0.01 (0.68) |
| 20% | -0.42 (**0.041**) | +0.09 (0.45) | -0.41 (**0.030**) | -0.14 (0.50) |
| 10% | -0.74 (**0.035**) | +0.06 (0.61) | -0.63 (0.14) | -0.21 (0.36) |

Three of the 20 are below 0.05 where about one is expected by chance, and none survives the Holm
correction. Accuracy differences at the same group count are also mixed up with feature count
(random keeps more features than ridge but fewer than filter).

### 19.5 Stratified thresholds in full

All 113 datasets, 20 seeds, 6,780 cases. Stratified minus uniform, with the spread across seeds:

| Datasets | Number | Mean change | 95% range | Better / equal / worse | p (non-zero) | Seed spread, stratified vs uniform |
|---|---:|---:|---|---|---|---|
| all | 113 | -0.002 | -0.044 to +0.043 | 43/23/47 | 0.377 (90) | 0.503 vs 0.485 |
| imbalanced | 57 | -0.035 | -0.086 to +0.018 | 17/9/31 | 0.054 (48) | 0.474 vs 0.483 |
| small | 26 | +0.060 | -0.037 to +0.178 | 7/11/8 | 0.934 (15) | 0.441 vs 0.444 |
| rest | 30 | +0.007 | -0.076 to +0.082 | 19/3/8 | 0.400 (27) | 0.610 vs 0.525 |

* The 95% range comes from resampling the datasets 10,000 times.
* Original minus uniform: +0.030 points (range -0.007 to +0.072), p = 0.182.
* The median dataset differs by 0.08 points; 32 of 113 differ by 0.2 or more and 7 by 0.5 or more, in
  both directions (largest: BeetleFly +1.0, Rock +0.70, DistalPhalanxOutlineCorrect -0.53,
  Computers -0.50).
* In the `rest` group stratified was *more* variable across seeds on 18 of 30 datasets (p = 0.032,
  not corrected).
* 16 tests in total (2 comparisons x 4 groups x accuracy and spread): one is below 0.05 (0.8 expected
  by chance), and none survives Holm (smallest corrected p = 0.51).
* First check (30 datasets, 10 seeds): all -0.032 (p = 0.353); imbalanced -0.097 (1/1/8, p = 0.012).
  On the same 10 imbalanced datasets with 20 seeds: -0.066 (p = 0.25). The 900 shared cases match in
  accuracy and in a hash of the test features.
* The full per-dataset table (mean, spread, minimum and maximum of every dataset) is in
  `results/group_pruning_bias_all_20seeds/bias_analysis.md`.

### 19.6 Stratified thresholds combined with pruning

The 30-dataset subset, 5 seeds, ridge only, drop 0.3, 3,600 rows:

| Groups kept | Stratified % | Uniform % | Change vs full, stratified | Change vs full, uniform | Time vs full |
|---:|---:|---:|---:|---:|---:|
| 100% | 81.56 | 81.58 | 0 | 0 | 1.000 |
| 75% | 81.60 | 81.54 | +0.03 | -0.04 | 0.69 |
| 50% | 81.45 | 81.45 | -0.12 | -0.14 | 0.475 |
| 30% | 81.13 | 80.96 | -0.43 | -0.62 | 0.31 |
| 20% | 80.60 | 80.45 | -0.97 | -1.14 | 0.22 |
| 10% | 79.76 | 79.27 | -1.81 | -2.32 | 0.14 |

Stratified minus uniform: -0.02, +0.06, 0.00, +0.17, +0.15, +0.49 points (p = 0.77, 0.31, 0.98, 0.44,
0.41, 0.046). I tested 24 slices (6 keep levels x 4 dataset groups) and corrected for all 24:

| Keep | Datasets | Number | Mean change | Better/equal/worse | p | Corrected p |
|---:|---|---:|---:|---|---:|---:|
| 100% | all | 30 | -0.019 | 11/8/11 | 0.770 | 1.000 |
| 75% | all | 30 | +0.057 | 14/6/10 | 0.310 | 1.000 |
| 50% | all | 30 | -0.001 | 9/8/13 | 0.975 | 1.000 |
| 30% | all | 30 | +0.171 | 14/3/13 | 0.441 | 1.000 |
| 20% | all | 30 | +0.149 | 15/6/9 | 0.406 | 1.000 |
| 10% | all | 30 | +0.491 | 21/3/6 | 0.046 | 1.000 |
| 10% | small | 10 | +1.354 | 9/1/0 | 0.004 | 0.094 |

Two of 24 are below 0.05 (about 1.2 expected by chance) and none survives the correction. "Small
datasets under heavy ridge pruning" is the only hint worth testing on new data.

### 19.7 25,000 features

24,948 features (1,579 groups), 10 datasets (GunPoint, ECG200, Beef, Computers, Wafer, ECG5000,
FacesUCR, Trace, Yoga, EthanolLevel, picked for a mix of sizes), 1 seed, drop 0.3, 100 rows:

| Version | Groups | Features | Accuracy % | Change | Time vs full | Break-even |
|---|---:|---:|---:|---:|---:|---:|
| full | 1,579 | 24,948 | 88.90 | 0 | 1.000 | - |
| ridge 50% | 790 | 11,559 | 89.04 | +0.14 | 0.457 | 398 |
| ridge 20% | 316 | 4,314 | 89.02 | +0.13 | 0.183 | 475 |
| filter 50% | 790 | 14,729 | 88.59 | -0.31 | 0.600 | 34 |
| filter 20% | 316 | 7,017 | 88.57 | -0.33 | 0.304 | 0 |

An indication only (10 datasets, 1 seed): with more features there is more to cut. Joint PPV was not
faster than the original counting in this sample (1.006), but the dataset mix differs from the
113-dataset runs, so the two numbers cannot be compared. On larger datasets the classifier's share of
test time becomes visible (Wafer: 0.33 s to predict vs 2.2 s to transform).

### 19.8 Pilot

5 datasets (GunPoint, ECG200, Beef, Computers, Wafer), 2 seeds, all three threshold versions, keep
100/50/20%, 300 rows, 0 differences. A smoke test, not a result: with the original thresholds, half
the groups gave -0.68 (filter) / -0.91 (ridge) points at about 0.54 of the time. Ridge selection took
0.1-2.3 s per case, far below the 48-hour limit the plan allowed, so I kept dropping 20% per round.

### 19.9 How the timings were taken on Windows

All timings in this README come from my runner (`experiments/group_pruning.py`) on the Windows laptop.
I did **not** run Arnav's benchmark suite (`benchmarks/run.py`) on Windows, only its tests.

Steps I followed:

1. Set `PYTHONPATH` (with `compat_windows`) and `NUMBA_NUM_THREADS=1`, and closed other heavy programs.
2. Started each run into a new folder; long runs were stopped once and continued with `--resume`.
3. Turned the results into tables with `summarise.py`, then ran the statistics scripts.

What the runner does for each dataset, seed and threshold version:

| Step | Timed? |
|---|---|
| Compile everything on a small random dataset first | no (so compilation never counts) |
| Fit MiniRocket once | yes |
| Transform the training data with all groups (both counting methods) | yes |
| Compute the full test features for the exactness check | no |
| Rank the groups, then retrain the classifier | yes |
| Transform the training data with the kept groups (both methods) | yes, once |
| Transform the test data with the kept groups (both methods) | yes, 3 times; the mean is reported |
| Predict | yes |

Things to keep in mind:

* I always compare **ratios** between versions timed in the same session, never raw seconds.
* My runner times everything in one process, in a fixed order (original counting, then joint PPV).
  Arnav's suite runs every case in its own process and shuffles the order of each pair
  (`benchmarks/controller.py`), so its numbers and mine come from different setups.
* scikit-learn's own threads were not limited, so classifier timings may use several cores.
* The first 53 of the 339 main-run cases were timed in an earlier session, because the run was stopped
  and resumed.

### 19.10 Windows compatibility

At first, 56 of the existing tests gave 1 failure and 2 errors on Windows, because of three things the
code assumes from macOS/Linux:

1. `import fcntl` (file locking) in `benchmarks/controller.py`, `benchmarks/run.py` and
   `tests/test_controller.py`.
2. `os.sysconf('SC_PAGE_SIZE')` in `benchmarks/run.py`, which raises an error Windows does not catch.
3. `import resource` in `benchmarks/memory.py`, which made the worker processes crash at start.

Instead of editing those files, I put small stand-in modules on `PYTHONPATH`:

| File | What it provides |
|---|---|
| `compat_windows/fcntl.py` | file locking through Windows' `msvcrt.locking` |
| `compat_windows/sitecustomize.py` | `os.sysconf` for page size and memory size (loaded automatically, also in subprocesses) |
| `compat_windows/resource.py` | peak memory from the Windows peak working set |

After this, the whole suite gives 64 tests OK and 1 skipped (a test that only applies to Apple ARM64),
73 OK since the demo and analysis tests were added.
On Windows the benchmark's memory number is the peak working set, not the macOS measure Arnav's
results use, so the two cannot be compared. The stand-ins are for Windows only.

### 19.11 Energy attempt

`experiments/energy.py` runs the test transform in a loop for at least 30 seconds under CodeCarbon and
subtracts an idle period of the same length. It works, but the readings on this laptop were not
usable. CodeCarbon 3.3.2 read about 995 J (about 46 W) for every 20-second window, idle or busy, and the
idle-corrected energy of a real transform came out **negative** (about -80 to -150 J per 30 s on
GunPoint). That is impossible, so I skipped energy as the plan said to, and I make no energy claim. A
machine with working power counters (Linux on Intel) or an external meter would be needed.

### 19.12 Commands for every result folder

All runs start with `python -B -m experiments.group_pruning --data ./data/Univariate_ts/ --output-dir <folder>`
followed by the options below, then `python -B -m experiments.summarise <folder>`. `DS30` means the 30
names in `results/bias_study_subset.csv`.

| Folder | Options |
|---|---|
| `group_pruning_pilot/` | `--datasets GunPoint ECG200 Beef Computers Wafer --seeds 0 1 --keep-fractions 1.0 0.5 0.2 --scorers ridge filter` |
| `group_pruning_full/` | `--bias-strategies original --scorers ridge filter` (resumed once with `--resume`) |
| `group_pruning_full_drop30_1seed/` | `--bias-strategies original --scorers ridge filter --drop-fraction 0.3 --seeds 0` |
| `group_pruning_bias_30ds_10seeds/` | `--datasets DS30 --bias-strategies original uniform stratified --keep-fractions 1.0 --seeds 0 1 2 3 4 5 6 7 8 9`, then `experiments.analyse_bias <folder>` |
| `group_pruning_bias_all_20seeds/` | `--bias-strategies original uniform stratified --keep-fractions 1.0 --seeds 0 1 ... 19` (all 20), then `experiments.analyse_bias <folder> --subsets results/bias_study_groups_all113.csv` |
| `group_pruning_combined_30ds_5seeds/` | `--datasets DS30 --bias-strategies uniform stratified --seeds 0 1 2 3 4 --drop-fraction 0.3 --scorers ridge`, then `experiments.analyse_bias <folder> --by-keep-fraction` |
| `group_pruning_25k/` | `--datasets GunPoint ECG200 Beef Computers Wafer ECG5000 FacesUCR Trace Yoga EthanolLevel --bias-strategies original --scorers ridge filter --keep-fractions 1.0 0.5 0.2 --seeds 0 --drop-fraction 0.3 --feature-count 25000` |
| `controls/` | `python -B -m experiments.controls --data ./data/Univariate_ts/ --output-dir results/controls`, then `python -B -m experiments.analyse_controls` |
| `reference/` | `python -B -m experiments.compare_to_paper` (the authors' CSVs come from github.com/angus924/minirocket) |
| final tests | `python -B -m experiments.final_analysis` (reads the saved CSVs) |
| `equivalence.md` | `python -B -m experiments.equivalence` (reads the saved CSVs) |
| `cross_platform.md`, `cross_platform_stages.csv` | `python -B -m experiments.cross_platform --recompute ./data/Univariate_ts/` (without `--recompute` it only re-reads the saved CSVs) |
| `threshold_class_share.csv` | `NUMBA_NUM_THREADS=1 python -B -m experiments.threshold_class_share --data ./data/Univariate_ts/ --output results/threshold_class_share.csv` |

### 19.13 Three checks added for the group paper

All three read data that is already in the repository, or fit nothing new beyond the threshold code.

**`experiments/equivalence.py`** (writes `results/equivalence.md`). Dataset-bootstrap intervals (seeds
averaged per dataset, 10,000 resamples) for every pruning level against the full model, and for stratified
against uniform thresholds. Keeping 50% of the groups with `filter` changed accuracy by -0.05 points (90%
interval -0.29 to +0.19), so a loss larger than 0.3 points is unlikely; with `ridge` it is -0.27 points
(-0.47 to -0.08), a small but real loss. Stratified against uniform: -0.002 points (90% interval -0.037 to
+0.035). The size of an acceptable loss has to be chosen **before** looking at an interval, so the script
reports the interval and leaves the margin to the reader.

**`experiments/cross_platform.py`** (writes `results/cross_platform.md`). Compares Arnav's macOS run with my
Windows run, fit by fit, for the original MiniRocket with 9,996 features and seeds 0 to 2. All 339 pairs used
the same split. 290 gave the same accuracy; 49 (on 26 datasets) differ by 1 to 17 test series, in both
directions, and the difference averages 0.02 points in favour of Windows.

To find where the two machines part ways, `--recompute` refits all 339 fits on Windows and recomputes the
macOS harness's own stage hashes (writes `results/cross_platform_stages.csv`, about 20 minutes). Without
`--recompute`, the script summarises the saved CSV again, so `results/cross_platform.md` can be rebuilt in a
second (commands in section 19.12). The input data is identical in all 339 fits, but the **fitted thresholds already differ in 333**; only 6 fits are
identical end to end. Both machines used the same MiniRocket code, NumPy 2.3.5 and numba 0.63.1 (scikit-learn,
aeon and the BLAS library differ, but they only act after the thresholds). MiniRocket's fitting code is
compiled with `fastmath=True`, which lets the compiler reorder float32 sums, so the split is consistent with
ARM and x86 rounding those sums differently (the same effect as the lesson in section 12); I did not prove it
by switching fast math off.
It is not a different random draw: on Windows, a different seed keeps accuracy identical in only 117 of 339
seed pairs, while the two machines with the same seed agree in 290 of 339. The refit differs from macOS in
exactly the same 49 fits as my committed Windows run did.

**`experiments/threshold_class_share.py`** (writes `results/threshold_class_share.csv`, resumable). Each
group takes its thresholds from one training series. For every threshold the script computes its value from
every training series (at most 100 per dataset, same code as the real fit) and measures eta squared, the
share of its variation explained by the series' class, with shuffled labels as the chance level. Fungi is
skipped (every class has one training series).

| Datasets | n | Median eta squared, true labels | Median eta squared, shuffled |
|---|---:|---:|---:|
| all | 112 | 0.26 | 0.03 |
| imbalanced | 56 | 0.41 | 0.08 |
| small | 26 | 0.14 | 0.03 |
| rest | 30 | 0.05 | 0.01 |

The true-label value is more than twice the shuffled value on 93 of 112 datasets, and on 20 datasets class
explains more than half of the variation. So the idea that thresholds "differ little from class to class"
is wrong for most datasets.
Even so, stratifying changed accuracy by -0.002 points. I did not test why; a plausible reason is that the
9,996 features overlap heavily, so the classifier does not depend on which class set a given threshold.

Please cite the original method as in the [main README](./README.md#minirocket).
