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

The full experiment grid (108 runs) is **already in `results/`**. Cloning this
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
| Fashion-MNIST | capable base model | 83.85 ± 0.55 | **84.82 ± 0.50** | best of 5 methods; trend not significant |
| BloodMNIST | capable base model | 86.69 ± 0.91 | **88.08 ± 0.56** | best; wins 6/6 seeds, *p* = 0.003 |
| CIFAR-10 | cold start (ResNet-18 from scratch) | **63.69 ± 0.96** | 58.28 ± 1.24 | **IRIS loses** — see below |

Three findings the repository is built to let you check:

- **BALD collapses under annotator noise.** The strongest baseline with a perfect
  oracle drops *below* random sampling once labels are noisy, on both capable-model
  datasets.
- **Cold start defeats every acquisition strategy, ours included.** On CIFAR-10
  from scratch, nothing beats random sampling, and reliability weighting actively
  hurts, because a weak model cannot tell hard-but-correct examples from corrupted
  ones. The paper reports this as a condition of use, not a footnote.
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

Only needed if you want to regenerate `results/` from scratch. The grid is 108
runs and takes hours on a single GPU.

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
  `iris-nodiv` and `iris-norel` ablations on the two 28×28 datasets
- **Oracle** — clean (γ = 0) and noisy (γ = 0.2 symmetric label noise)
- **Seeds** — 6 on Fashion-MNIST and BloodMNIST, 3 on CIFAR-10 (108 runs)
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
