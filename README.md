# IRIS — Introspective Reliability-gated Instance Selection

**Introspective, reliability-gated instance selection for human-in-the-loop deep
active learning under imperfect annotators.**

A deep classifier trained on a small labelled set is confused about exactly the
examples it has not seen enough of, and its own output probabilities are a poor
guide to which those are. IRIS adds three things to an ordinary classifier so it
can ask a human the right questions and survive the wrong answers:

1. an **introspection head** — a two-layer MLP on detached penultimate features
   that learns, from the model's own error history, the probability that the
   classifier is about to be wrong (replacing hand-crafted entropy as the
   acquisition signal);
2. a **k-center diversity gate** over its top-ranked candidates, so a batch of
   500 labels buys 500 distinct insights instead of 500 copies of one;
3. **small-loss reliability weighting** of the labels bought from a noisy human
   annotator, so corrupted annotations are damped inside the same training loop.

It costs under 1% extra parameters and no extra forward pass at query time.

---

## Everything here reproduces without training

The full experiment grid (180 runs) is **already in `results/`**. Cloning this
repo and running two scripts regenerates every figure, table and number in the
paper — including the paper PDF itself — in well under a minute on a laptop, with
**no GPU and no training**.

```bash
git clone git@github.com:samim-reza/IRIS.git
cd IRIS
pip install -r requirements.txt

python src/make_plots.py      # -> results/*.png, results/summary_table.{tex,md}
python src/make_numbers.py    # -> paper/numbers.tex  (196 LaTeX macros)
```

That is the whole reproduction path. Re-running the experiments is optional and
covered in [Re-running the experiments](#re-running-the-experiments-optional-gpu).

### Verifying the results, not just reading them

Every run keeps the **final trained model** that produced its published
accuracy, under `checkpoints/` (180 files, ~1.9 GB, git-ignored). Each file
carries the weights, both heads, and the audit trail a claim needs: the exact
examples the model was trained on and the labels the (possibly lying) oracle
returned for them.

```bash
python src/verify_checkpoint.py                    # re-measure all 180 checkpoints
python src/verify_checkpoint.py --dataset bloodmnist
```

It reloads each model, re-evaluates it on the untouched test set, and prints the
delta against the accuracy recorded during training. A clean result is
`max |delta| = 0.000000`.

Training is **deterministic**: cuDNN autotuning is disabled and the training
DataLoader is seeded, so re-running reproduces the published numbers exactly
rather than approximately. This was verified by re-running a completed run and
getting byte-identical per-round accuracies.

The grid also carries a **divergence guard**. SGD occasionally fails to escape
its initialisation and a run sits at chance level for good; one such run
appeared here and would have silently dragged a baseline's mean down by 11
points. After training, the model is checked against *its own training set* --
never the test set -- and restarted from a different initialisation if it could
not fit the data it was just shown. Healthy runs never trigger it and are
bit-identical with or without it.

To reproduce the whole grid from scratch, with checkpoints:

```bash
tmux new-session -d -s iris 'bash src/rerun_all.sh > logs/rerun.log 2>&1'
tmux attach -t iris        # watch;  Ctrl-b d to detach
```

`results/previous_run/` holds the earlier, pre-checkpointing results, and
`python src/compare_runs.py` diffs the two per scenario and flags any change in
which method wins.

### What "no training" does and does not cover

`results/` holds the **measurements**, not model weights. Every number, figure,
table, significance test and the paper PDF regenerate from it exactly, with no
GPU — that is verified, and a fresh clone reproduces the committed figures
bit-identically.

Active learning retrains the classifier *from scratch* after every acquisition
round (see `run_al` in `src/hitl_experiments.py`), so the 180 runs train 1,080
models in total. Keeping all of them would serve nobody, so each run keeps the
**final** one — the model that produced that run's published accuracy. Those
180 checkpoints are ~1.9 GB, too large for git, so they are produced by the
grid rather than shipped; `src/verify_checkpoint.py` re-measures them.

So: to **check the paper's numbers, figures and tables**, you need no GPU and no
training. To **hold the trained models** and re-measure them yourself, rerun the
grid with `--save-checkpoints` (~3 GPU-hours for all of it, or ~3 minutes for a
single Fashion-MNIST or BloodMNIST run).

### Rebuild the paper too

```bash
cd paper
make            # pdflatex -> bibtex -> pdflatex x2  -> main_en.pdf
```

Needs a TeX distribution with `stix`, `algorithmicx`, `pgf/tikz` and `natbib`.
The Elsevier CAS class files (`cas-sc.cls`, `cas-common.sty`,
`cas-model2-names.bst`) are vendored in `paper/`, so nothing else is required.

---

## Headline results

Final accuracy after 3,000 labels (Fashion-MNIST, BloodMNIST) or 6,000 labels
(CIFAR-10), under a **noisy annotator** (γ = 0.2 symmetric label noise), mean ±
sd over seeds. Full table: `results/summary_table.md`.

| Dataset | Regime | Random | IRIS (ours) | Verdict |
|---|---|---|---|---|
| Fashion-MNIST | capable base model | 83.71 ± 0.54 | **85.68 ± 0.61** | best of 5; wins 5/6 seeds, *p* = 0.021, and saves 17% of the budget |
| BloodMNIST | capable base model | 86.94 ± 0.75 | **87.82 ± 0.75** | best of 5; wins 5/6 seeds, *p* = 0.097 — a trend, not proof |
| CIFAR-10 | cold start (ResNet-18 from scratch) | **63.24 ± 0.55** | 62.20 ± 0.30 | **IRIS loses** — see below |

Three findings the repository is built to let you check:

- **BALD collapses under annotator noise.** The strongest baseline with a perfect
  oracle drops *below* random sampling once labels are noisy, on both capable-model
  datasets.
- **Cold start defeats every acquisition strategy, ours included.** On CIFAR-10
  from scratch nothing beats random sampling. Running both ablations there
  localises the cause: removing reliability weighting changes nothing
  (*p* = 0.850) and removing the diversity gate makes things worse, so it is the
  learned *selection signal* that fails when the backbone is still at ~46%
  accuracy. The paper reports this as a condition of use, not a footnote.
- **Batch quality ≠ error-ranking quality.** Softmax entropy has a *higher*
  point-wise error-detection AUROC than the learned head on all three datasets,
  yet picks worse batches (`results/diagnostics_auroc.png`).

---

## Repository layout

| Path | Contents |
|---|---|
| `paper/main_en.tex` → `main_en.pdf` | the manuscript, Elsevier CAS single-column format |
| `paper/refs.bib` | 50-entry BibTeX database (author–year, `cas-model2-names`) |
| `paper/numbers.tex` | auto-generated LaTeX macros — never edit by hand |
| `paper/cas-*.{cls,sty,bst}` | vendored Elsevier CAS class files |
| `paper/Makefile` | `make` builds the PDF, `make clean` removes artefacts |
| `src/hitl_experiments.py` | full implementation: models, IRIS, 4 baselines, noisy oracle, AL loop |
| `src/make_plots.py` | figures + summary tables from `results/results.csv` |
| `src/make_numbers.py` | LaTeX macros + paired t-tests, so the prose cannot drift from the data |
| `src/finalize.sh` | watcher that regenerates figures/tables/PDF as results land |
| `src/build_notebook.py` | regenerates `notebooks/IRIS.ipynb` from the experiment source |
| `results/` | **shipped**: `results.csv`, `summary.csv`, `diagnostics.csv`, figures, tables |
| `notebooks/IRIS.ipynb` | self-contained notebook (code + results + figures) |
| `docs/literature_review/` | the synthesis, the gap table, and per-source metadata |
| `logs/` | logs of the actual runs that produced `results/` |

Two directories are deliberately **not** in git:

- `data/` — the datasets (~460 MB). Fashion-MNIST and CIFAR-10 download
  automatically on first run; BloodMNIST is fetched from
  [Zenodo record 10519652](https://zenodo.org/records/10519652) as
  `data/bloodmnist.npz`.
- `papers/` — the 21 reference PDFs. Third-party copyright; not redistributable.

---

## Re-running the experiments (optional, GPU)

Only needed if you want to regenerate `results/` from scratch, or if you want
trained weights (see below). The grid is 180 runs and takes roughly 3 GPU-hours
on an idle RTX 4090.

```bash
pip install torch torchvision          # in addition to requirements.txt

nohup python -u src/hitl_experiments.py > logs/experiments.log 2>&1 &
nohup bash src/finalize.sh > /dev/null 2>&1 &   # rebuilds figures/PDF as results land

tail -f logs/experiments.log           # grid ends with "ALL RUNS COMPLETE"
```

Both survive an SSH disconnect. **Restarts are safe**: any (dataset, method,
noise, seed) combination already recorded in `results/results.csv` is skipped, so
you can stop and resume freely.

To run a subset:

```bash
python -u src/hitl_experiments.py --datasets fmnist bloodmnist --seeds 3 4 5
```

### The grid

- **Datasets** — Fashion-MNIST (500 + 5×500 labels, small CNN), BloodMNIST /
  MedMNIST v2 (500 + 5×500, *the same* small CNN and recipe, untuned), CIFAR-10
  (1,000 + 5×1,000, ResNet-18 from scratch)
- **Methods** — random · entropy · BALD · core-set · **IRIS**, plus the
  `iris-nodiv` and `iris-norel` ablations on all three datasets
- **Oracle** — clean (γ = 0) and noisy (γ = 0.2 symmetric label noise)
- **Seeds** — 6 on Fashion-MNIST and BloodMNIST, 3 on CIFAR-10 (72 + 72 + 36 =
  180 runs, ablations included)
- **Metrics** — accuracy vs. budget, final accuracy, AUBC, error-prediction
  AUROC, and a paired *t*-test of IRIS against the strongest baseline on matched
  seeds

BloodMNIST inherits the Fashion-MNIST architecture, optimiser, schedule and
budget with **no retuned hyperparameter**, which is what makes it a transfer test
rather than a demonstration.

---

## Requirements

Reproduction only (no training): Python ≥ 3.9 with `numpy`, `pandas`,
`matplotlib`, `scipy` — see `requirements.txt`. Re-running the grid additionally
needs `torch` and `torchvision`; rebuilding the notebook needs `nbformat` and
`nbclient`.

## Citing

The manuscript is under preparation for submission. Until it appears, please cite
this repository.

```bibtex
@software{iris2026,
  author = {Samim},
  title  = {IRIS: Introspective Reliability-gated Instance Selection for
            Human-in-the-Loop Deep Active Learning under Imperfect Annotators},
  year   = {2026},
  url    = {https://github.com/samim-reza/IRIS}
}
```
