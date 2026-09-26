# IRIS — Introspective Reliability-gated Instance Selection

Human-in-the-loop deep active learning when the annotator is imperfect.

A classifier trained on a few hundred labels is confused about exactly the
examples it has not seen enough of, and its own output probabilities are a poor
guide to which those are. A human can resolve that — a few thousand times, and
not perfectly. This repository studies two separable mechanisms for that
setting:

- **Reliability-weighted label acceptance** — a per-sample exponential moving
  average of the loss down-weights annotations that behave like mistakes, inside
  the normal training loop. This is the part that **transfers**: bolted onto
  BADGE, an acquisition function built on completely different principles, it
  still gains about a point on every dataset.
- **An introspection head** — a two-layer MLP on detached penultimate features
  that learns, from the model's own error history, the probability it is about
  to be wrong. Used as the acquisition signal with a k-center diversity gate.
  It beats the prior learned-signal method (Learning-Loss) decisively, and beats
  every uncertainty and coverage baseline, but it does **not** beat BADGE.

Both results are in the table below, including the one that goes against us.

---

## Everything reproduces without training

All **285 runs** are already in `results/`. Cloning the repo and running two
scripts regenerates every figure, table and reported number in under a minute on
a laptop — **no GPU, no training**.

```bash
git clone git@github.com:samim-reza/IRIS.git
cd IRIS
pip install -r requirements.txt

python src/make_plots.py      # -> results/*.png and the summary tables
python src/make_numbers.py    # -> the reported figures, regenerated
python src/check_claims.py    # -> 22/22 claims verified against the data
```

A fresh clone reproduces the committed figures **bit-identically** — verified,
not assumed.

Prefer a notebook? `notebooks/IRIS.ipynb` carries the full implementation, the
results, the learning curves and the checkpoint verification in one file. It
does not train anything unless you set `RUN_FULL = True`.

---

## Results — final accuracy under a noisy annotator (γ = 0.2)

Mean ± sd over 6 seeds (3 for CIFAR-10). Full table: `results/summary_table.md`.

| Method | Fashion-MNIST | BloodMNIST | CIFAR-10 |
|---|---|---|---|
| Random | 83.71 ± 0.59 | 86.94 ± 0.82 | 63.24 ± 0.67 |
| Learning-Loss | 83.27 ± 1.09 | 80.53 ± 2.67 | 53.97 ± 4.63 |
| **IRIS** (ours) | 85.68 ± 0.67 | 87.82 ± 0.82 | 62.20 ± 0.37 |
| BADGE | 86.59 ± 0.33 | 88.42 ± 0.77 | 63.15 ± 1.86 |
| **BADGE + reliability** (ours) | **87.52 ± 0.43** | **89.36 ± 0.77** | **64.59 ± 1.42** |

What to take from it:

- **The reliability gate transfers.** Added to BADGE it gains +0.92 on
  Fashion-MNIST (*p* = 0.003, 6/6 seeds), +0.94 on BloodMNIST and +1.44 on
  CIFAR-10 — lifts as large as those it produces inside IRIS, on an acquisition
  function it was never designed for.
- **Our acquisition signal loses to BADGE**, and we say so. It beats
  Learning-Loss by 2.4 and 7.3 points, and beats every uncertainty and coverage
  baseline, but BADGE's gradient embedding selects better batches. Rebuilding
  our diversity gate in that same geometry (`iris-grad`) did not close the gap.
- **BALD collapses under annotator noise** — the best baseline with a clean
  oracle drops *below* random sampling once labels are noisy.
- **Cold start defeats every acquisition strategy.** On CIFAR-10 from scratch
  nothing beats random sampling; BADGE + reliability is the only configuration
  anywhere in this study that does.
- **Batch quality ≠ error-ranking quality.** Softmax entropy has a *higher*
  point-wise error-detection AUROC than the learned head on all three datasets,
  yet picks worse batches.

---

## Training

Only needed to regenerate `results/` from scratch. The full grid is 285 runs,
roughly 6 GPU-hours on an idle RTX 4090.

```bash
pip install torch torchvision          # in addition to requirements.txt

# detached, survives an SSH disconnect
setsid nohup bash src/rerun_all.sh > logs/rerun.log 2>&1 &
tail -f logs/rerun.log                 # ends with "RERUN COMPLETE"
```

**Restarts are safe.** Any (dataset, method, noise, seed) already recorded in
`results/results.csv` is skipped, so you can stop and resume freely.

Subsets:

```bash
python -u src/hitl_experiments.py --datasets bloodmnist --seeds 0 1 2
python -u src/hitl_experiments.py --datasets fmnist --seeds 0 --save-checkpoints
python -u src/hitl_experiments.py --quick        # 2-minute smoke test
```

> `--quick` appends rows to `results/results.csv`. Use a scratch copy if you
> care about the file.

### The grid

- **Datasets** — Fashion-MNIST (500 + 5×500 labels, small CNN), BloodMNIST /
  MedMNIST v2 (identical CNN and schedule, no retuning), CIFAR-10
  (1,000 + 5×1,000, ResNet-18 from scratch)
- **Methods** — `random`, `entropy`, `bald`, `coreset`, `learnloss`, `badge`,
  `badge-rel`, `iris`, `iris-grad`, plus the `iris-nodiv` / `iris-norel`
  ablations
- **Oracle** — clean (γ = 0) and noisy (γ = 0.2 symmetric label noise)
- **Seeds** — 6 on the 28×28 datasets, 3 on CIFAR-10 (285 runs)

Training is **deterministic**: cuDNN autotuning is off and the DataLoader is
seeded, so a re-run reproduces the published numbers exactly rather than
approximately. A **divergence guard** catches the occasional run where SGD never
escapes its initialisation — detected from *training* accuracy only, never the
test set, and restarted from a fresh init. Healthy runs are bit-identical with
or without it.

---

## Testing

```bash
python src/check_claims.py                 # 22 stated conclusions vs the data
python src/verify_checkpoint.py            # reload every saved model, re-measure
python src/verify_checkpoint.py --dataset bloodmnist
python src/compare_runs.py                 # this grid vs results/previous_run/
```

`--save-checkpoints` keeps the final trained model of each run in
`checkpoints/`, together with the examples it was trained on and the labels the
(possibly lying) oracle returned — the audit trail an accuracy claim needs.
`verify_checkpoint.py` reloads each one and re-measures it on the untouched test
set; a clean result is `max |delta| = 0.000000`.

`check_claims.py` exists because regenerating numbers keeps figures honest but
not *sentences*. "BADGE beats IRIS", "BALD falls below random", "the gate lifts
BADGE" are English, and English does not regenerate. Run it after any re-run: a
conclusion that silently inverts fails loudly instead. (`src/lint_prose.py` is
its companion for the write-up itself; it needs the compiled manuscript, which
is not distributed here, and reports that and exits cleanly if absent.)

### Trained models

The 285 trained models — one per run, the final model of each — are archived on
Zenodo rather than here, at 3.2 GB. Each carries the audit trail needed to
re-measure its reported accuracy independently: the indices of the examples it
saw, the labels the noisy oracle returned for them, the training configuration
and the PyTorch version. All 285 re-measure at `max |delta| = 0.000000`.

Archives are split by dataset, so checking one claim does not mean downloading
everything:

| Archive | Models | Size |
|---|---|---|
| `iris-checkpoints-fmnist.tar` | 114 | 396 MB |
| `iris-checkpoints-bloodmnist.tar` | 114 | 396 MB |
| `iris-checkpoints-cifar10.tar` | 57 | 2.4 GB |

```bash
tar -xf iris-checkpoints-bloodmnist.tar -C checkpoints/
python src/verify_checkpoint.py --dataset bloodmnist
```

Only final-round models are kept: the per-round accuracies behind the
accuracy-budget curves are already in `results/results.csv`, so storing those
models would not let you check anything the CSV does not already state.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/hitl_experiments.py` | everything: models, IRIS, the reliability gate, all 9 acquisition methods, the noisy oracle, the AL loop |
| `src/make_plots.py` | figures and summary tables from `results/results.csv` |
| `src/make_numbers.py` | reported figures and paired t-tests, regenerated from the data |
| `src/check_claims.py` | asserts the stated conclusions against the data |
| `src/verify_checkpoint.py` | reload saved models and re-measure them |
| `src/compare_runs.py` | diff this grid against a previous one |
| `src/zenodo_upload.py` | deposit the checkpoint archives and mint a DOI |
| `src/rerun_all.sh` | full grid, detached, with checkpoints |
| `results/` | **shipped**: `results.csv`, `summary.csv`, `diagnostics.csv`, figures, tables |
| `notebooks/IRIS.ipynb` | code, results and verification in one file |
| `logs/` | logs of the runs that produced `results/` |

Not in git: `data/` (~460 MB, **downloads automatically on first run** —
Fashion-MNIST and CIFAR-10 via torchvision, BloodMNIST from
[Zenodo record 10519652](https://zenodo.org/records/10519652) with its checksum
verified), `checkpoints/`
(3.2 GB — see below), `papers/` (third-party PDFs), and the manuscript sources.

---

## Requirements

Reproduction only: Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib`, `scipy`
(`requirements.txt`). Training additionally needs `torch` and `torchvision`;
rebuilding the notebook needs `nbformat` and `nbclient`.

Note `src/` targets Python 3.10+, and the results were produced with
**torch 2.12.0+cu130**. Torch 2.11 gives different accuracies for the same seed,
so do not split a grid across versions.

## Citing

```
@software{iris2026,
  author = {Samim},
  title  = {IRIS: Introspective Reliability-gated Instance Selection for
            Human-in-the-Loop Deep Active Learning under Imperfect Annotators},
  year   = {2026},
  url    = {https://github.com/samim-reza/IRIS}
}
```
