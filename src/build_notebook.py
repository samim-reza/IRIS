"""Assemble the self-contained project notebook notebooks/IRIS.ipynb from the
experiment source, so the notebook and the nohup-run script never diverge."""

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "hitl_experiments.py")
OUT = os.path.join(ROOT, "notebooks", "IRIS.ipynb")

with open(SRC) as f:
    src = f.read()
# strip the CLI entry point: in a notebook __name__ == "__main__" would fire it
src = src.split('if __name__ == "__main__":')[0].rstrip()

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# IRIS: Introspective Reliability-Gated Instance Selection
## Human-in-the-Loop Deep Active Learning under Imperfect Annotators

**Self-contained project notebook** — code, experiments, and results in one file.

### The idea in three bullets
- **Introspection head**: a 2-layer MLP on the backbone's *detached* penultimate
  features, trained jointly (class-balanced BCE) to predict *the probability that the
  main classifier is wrong* on an input — a learned, bounded, calibrated acquisition
  signal replacing hand-crafted softmax entropy.
- **Diversity gate**: the top β·K introspection-ranked pool samples are filtered to K by
  greedy k-center selection in feature space (seeded with the already-labeled set), so
  each human-annotation batch is informative *and* non-redundant.
- **Reliability-weighted training**: per-sample EMA losses drive small-loss weighting
  (co-teaching principle), so labels bought from a *noisy* human annotator (γ = 0.2)
  don't poison the model.

### Experimental protocol
Pool-based batch AL on **Fashion-MNIST** (500 + 5×500 labels, small CNN) and
**CIFAR-10** (1000 + 5×1000 labels, ResNet-18) against **random / entropy / BALD /
core-set** baselines, under a **clean** and a **noisy (γ=0.2)** simulated annotator,
3 seeds each; metrics: accuracy-vs-budget curves, final accuracy, AUBC, plus an
error-prediction AUROC audit of the introspection head.

### Project layout
```
IRIS/
├── src/                 # hitl_experiments.py, make_plots.py, make_numbers.py, finalize.sh
├── results/             # results.csv, summary tables, figures  (shipped: no training needed)
├── paper/               # main_en.tex → main_en.pdf (numbers auto-filled from results)
├── notebooks/IRIS.ipynb # this notebook
├── docs/                # literature review + per-source metadata
├── logs/                # experiments.log (the nohup run), finalize.log
├── data/                # datasets (git-ignored; downloaded on first run)
└── papers/              # reference PDFs (git-ignored: not redistributable)
```
""")

md("## 1. Experiment implementation\n"
   "The complete implementation (identical to `src/hitl_experiments.py`). "
   "Running this cell only *defines* everything — no training starts.")
code(src)

md("## 2. Quick demo (~2 min on GPU)\n"
   "Set `RUN_DEMO = True` to run a miniature AL loop (Fashion-MNIST, IRIS vs entropy, "
   "1 round) and sanity-check the pipeline end-to-end.")
code("""RUN_DEMO = False
if RUN_DEMO:
    cfg = dict(init_labeled=500, query_size=500, rounds=1,
               epochs=15, batch_size=64, lr=0.02)
    for method in ("iris", "entropy"):
        run_al("fmnist", method, 0.2, 0, cfg)""")

md("## 3. Full experiment grid (long-running)\n"
   "The full grid (3 datasets × ≥5 methods × 2 oracle noise levels × 3-6 seeds) is meant "
   "to run detached from any SSH session:\n"
   "```bash\n"
   "cd <project>/HITL\n"
   "nohup /opt/miniconda/bin/python -u code/hitl_experiments.py "
   "> logs/experiments.log 2>&1 &\n"
   "nohup bash code/finalize.sh > /dev/null 2>&1 &   # auto-rebuilds figures + PDF\n"
   "tail -f logs/experiments.log                      # watch progress\n"
   "```\n"
   "Runs are resumable: completed (dataset, method, noise, seed) combinations found in "
   "`results/results.csv` are skipped on restart. Set `RUN_FULL = True` below to launch "
   "from the notebook instead (not recommended over SSH).")
code("""RUN_FULL = False
if RUN_FULL:
    import subprocess
    subprocess.Popen(
        ["nohup", "/opt/miniconda/bin/python", "-u", "code/hitl_experiments.py"],
        stdout=open("logs/experiments.log", "a"), stderr=subprocess.STDOUT,
        cwd=ROOT, start_new_session=True)
    print("launched — tail logs/experiments.log")""")

md("## 4. Results\nLoads `results/results.csv` (written incrementally by the grid run) "
   "and reproduces every figure and table used in the paper.")
code("""import pandas as pd
res_path = os.path.join(RESULTS_DIR, "results.csv")
if os.path.exists(res_path):
    df = pd.read_csv(res_path)
    # interrupted runs leave partial duplicate rows — keep the latest per key
    df = df.drop_duplicates(subset=["dataset", "method", "noise", "seed", "round"],
                            keep="last")
    print(f"{len(df)} result rows, "
          f"{df.groupby(['dataset','method','noise','seed']).ngroups} runs recorded")
    display(df.tail(10))
else:
    df = None
    print("results.csv not found — run the grid first (section 3)")""")

code("""# Learning curves: accuracy vs annotation budget (mean ± std over seeds)
import matplotlib.pyplot as plt

METHOD_ORDER = ["random", "entropy", "bald", "coreset", "iris",
                "iris-nodiv", "iris-norel"]
LAB = {"random": "Random", "entropy": "Entropy", "bald": "BALD",
       "coreset": "Coreset", "learnloss": "Learning-Loss", "badge": "BADGE", "badge-rel": "BADGE + reliability",
       "iris": "IRIS (ours)", "iris-grad": "IRIS-grad (ours)",
       "iris-nodiv": "IRIS w/o diversity", "iris-norel": "IRIS w/o reliability"}
COL = {"random": "#9e9e9e", "entropy": "#1976d2", "bald": "#7b1fa2",
       "coreset": "#00897b", "learnloss": "#5d4037", "badge": "#455a64", "badge-rel": "#0277bd",
       "iris": "#d32f2f", "iris-grad": "#ad1457",
       "iris-nodiv": "#ef6c00", "iris-norel": "#c2185b"}

if df is not None:
    combos = [(ds, nz) for ds in sorted(df.dataset.unique())
              for nz in sorted(df[df.dataset == ds].noise.unique())]
    fig, axes = plt.subplots(1, len(combos), figsize=(5.2 * len(combos), 4),
                             squeeze=False)
    for ax, (ds, nz) in zip(axes[0], combos):
        sub = df[(df.dataset == ds) & (df.noise == nz)]
        for m in METHOD_ORDER:
            g = sub[sub.method == m]
            if g.empty:
                continue
            piv = g.pivot_table(index="n_labeled", columns="seed", values="test_acc")
            mean, std = piv.mean(1), piv.std(1).fillna(0)
            ax.plot(piv.index, mean, "o-", ms=3, color=COL[m], label=LAB[m])
            ax.fill_between(piv.index, mean - std, mean + std, color=COL[m], alpha=0.15)
        ax.set_title(f"{ds} — {'clean' if nz == 0 else f'noisy γ={nz}'}")
        ax.set_xlabel("# human labels"); ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()""")

code("""# Summary table: final accuracy and AUBC per method/setting
import numpy as np

def aubc(n, a):
    n, a = np.asarray(n, float), np.asarray(a, float)
    o = np.argsort(n)
    return np.trapezoid(a[o], n[o]) / (n[o][-1] - n[o][0])

if df is not None:
    rows = []
    for (ds, nz, m), g in df.groupby(["dataset", "noise", "method"]):
        piv = g.pivot_table(index="n_labeled", columns="seed", values="test_acc")
        aubcs = [aubc(piv.index, piv[c].values) for c in piv.columns]
        finals = piv.loc[piv.index.max()].values
        rows.append(dict(dataset=ds, noise=nz, method=LAB.get(m, m),
                         final_acc=f"{100*np.mean(finals):.2f} ± {100*np.std(finals):.2f}",
                         AUBC=f"{100*np.mean(aubcs):.2f} ± {100*np.std(aubcs):.2f}"))
    display(pd.DataFrame(rows).sort_values(["dataset", "noise"]).reset_index(drop=True))""")

code("""# Diagnostic: does the introspection head out-rank entropy at error prediction?
diag_path = os.path.join(RESULTS_DIR, "diagnostics.csv")
if os.path.exists(diag_path):
    dg = pd.read_csv(diag_path)
    dg = dg[(dg.method == "iris")]
    piv = dg.groupby(["dataset", "round"])[
        ["auroc_introspection", "auroc_entropy"]].mean().round(4)
    display(piv)
else:
    print("diagnostics.csv not found yet")""")

md("## 5. Verifying the saved models\n"
   "Each run keeps the final trained model that produced its published accuracy, "
   "in `checkpoints/`, together with the examples it was trained on and the labels "
   "the (possibly lying) oracle returned. The cell below reloads a sample of them "
   "and re-measures accuracy on the untouched test set: a clean result is a delta "
   "of 0.0000 for every checkpoint.\n\n"
   "Training is deterministic (cuDNN autotuning off, DataLoader seeded), so these "
   "numbers reproduce exactly rather than approximately.")

code("""# Re-measure saved checkpoints.  Needs a GPU; set SAMPLE=None for all 174.
SAMPLE = 6
import glob, torch, random as _rnd
sys.path.insert(0, os.path.join(ROOT, "src"))
paths = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pt")))
if not paths:
    print("no checkpoints — run the grid with --save-checkpoints first")
else:
    import verify_checkpoint as V
    sel = paths if SAMPLE is None else _rnd.Random(0).sample(paths, min(SAMPLE, len(paths)))
    cache, worst = {}, 0.0
    print(f"{'checkpoint':<52}{'recorded':>10}{'reloaded':>10}{'delta':>9}")
    for cp in sel:
        ck, acc, d = V.verify(cp, cache)
        worst = max(worst, abs(d))
        print(f"{os.path.basename(cp):<52}{ck['test_acc']:>10.4f}{acc:>10.4f}{d:>+9.4f}")
    print(f"\\nmax |delta| over {len(sel)} checkpoint(s) = {worst:.6f}")""")

md("""## 6. Where everything lives
- **Paper (PDF):** `paper/main_en.pdf` — Elsevier CAS format; every number, table
  and figure is auto-filled from `results/` by `src/make_numbers.py` and
  `src/make_plots.py` (rerun together by `src/finalize.sh`).
- **Saved models:** `checkpoints/` — the final model of each of the 174 runs,
  re-checkable with `src/verify_checkpoint.py` (section 5).
- **Previous results:** `results/previous_run/` — the earlier grid, with
  `src/compare_runs.py` to diff old against new.
- **Literature review:** `docs/literature_review/literature_review.md`.
- **Reproduce from scratch:** `tmux new-session -d -s iris 'bash src/rerun_all.sh
  > logs/rerun.log 2>&1'`, wait for `RERUN COMPLETE`, then re-run section 4 here.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"}
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT)
