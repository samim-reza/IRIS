"""Generate figures and summary tables from results/results.csv.

Outputs (into results/):
  curves_<dataset>_noise<r>.png : accuracy vs. #labels, mean +/- std over seeds
  diagnostics_auroc.png         : introspection vs. entropy error-prediction AUROC
  summary_table.md / .tex       : final accuracy and AUBC per method
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")

METHOD_ORDER = ["random", "entropy", "bald", "coreset", "iris",
                "iris-nodiv", "iris-norel"]
LABELS = {
    "random": "Random", "entropy": "Entropy", "bald": "BALD",
    "coreset": "Coreset", "iris": "IRIS (ours)",
    "iris-nodiv": "IRIS w/o diversity", "iris-norel": "IRIS w/o reliability",
}
COLORS = {
    "random": "#9e9e9e", "entropy": "#1976d2", "bald": "#7b1fa2",
    "coreset": "#00897b", "iris": "#d32f2f",
    "iris-nodiv": "#ef6c00", "iris-norel": "#c2185b",
}
DS_NAME = {"fmnist": "Fashion-MNIST", "bloodmnist": "BloodMNIST",
           "cifar10": "CIFAR-10"}
# narrative order: competent-model regimes first, cold-start last
DS_ORDER = ["fmnist", "bloodmnist", "cifar10"]


def ds_sort(names):
    return sorted(names, key=lambda d: (DS_ORDER.index(d)
                                        if d in DS_ORDER else 99, d))


def aubc(n_labeled, acc):
    """Area under the accuracy-vs-budget curve, normalised by budget span."""
    n, a = np.asarray(n_labeled, float), np.asarray(acc, float)
    order = np.argsort(n)
    n, a = n[order], a[order]
    if n[-1] == n[0]:  # single budget point (partial run): AUBC degenerates to acc
        return float(a[-1])
    return np.trapezoid(a, n) / (n[-1] - n[0])


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
    # a run interrupted by OOM/restart leaves partial rows behind its complete
    # rerun — keep only the latest row per (dataset, method, noise, seed, round)
    df = df.drop_duplicates(subset=["dataset", "method", "noise", "seed", "round"],
                            keep="last")
    rows = []

    for ds in ds_sort(df.dataset.unique()):
        for noise in sorted(df[df.dataset == ds].noise.unique()):
            sub = df[(df.dataset == ds) & (df.noise == noise)]
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for m in METHOD_ORDER:
                g = sub[sub.method == m]
                if g.empty:
                    continue
                piv = g.pivot_table(index="n_labeled", columns="seed",
                                    values="test_acc")
                mean, std = piv.mean(1), piv.std(1).fillna(0)
                ax.plot(piv.index, mean, marker="o", ms=4, lw=1.8,
                        color=COLORS[m], label=LABELS[m])
                ax.fill_between(piv.index, mean - std, mean + std,
                                color=COLORS[m], alpha=0.15)
                # per-seed AUBC and final accuracy for the summary table
                aubcs = [aubc(piv.index, piv[c].values) for c in piv.columns]
                finals = piv.loc[piv.index.max()].values
                rows.append(dict(
                    dataset=ds, noise=noise, method=m, n_seeds=piv.shape[1],
                    final_acc_mean=np.mean(finals), final_acc_std=np.std(finals),
                    aubc_mean=np.mean(aubcs), aubc_std=np.std(aubcs)))
            ax.set_xlabel("Number of human-annotated labels")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"{DS_NAME.get(ds, ds)} — "
                         f"{'clean oracle' if noise == 0 else f'noisy oracle (γ={noise})'}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            out = os.path.join(RESULTS_DIR, f"curves_{ds}_noise{noise}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print("wrote", out)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    # Re-read what was just written.  make_numbers.py reads this CSV, and the
    # float round-trip can shift a value sitting exactly on a rounding
    # boundary (0.91505 -> 91.50 in memory but 91.51 after the round-trip),
    # which would print two different numbers for one quantity in the table
    # and in the prose.  Building the table from the round-tripped values
    # keeps the two in lockstep.
    summary = pd.read_csv(summary_path)

    # ---- markdown + latex tables ----
    md, tex = [], []
    tex.append("\\begin{tabular}{llcc}\n\\toprule")
    tex.append("Setting & Method & Final acc. (\\%) & AUBC (\\%) \\\\")
    settings = sorted(summary.groupby(["dataset", "noise"]).groups,
                      key=lambda k: (DS_ORDER.index(k[0])
                                     if k[0] in DS_ORDER else 99, k[1]))
    for ds, noise in settings:
        g = summary[(summary.dataset == ds) & (summary.noise == noise)]
        setting = f"{DS_NAME.get(ds, ds)}, " + \
                  ("clean" if noise == 0 else f"noisy γ={noise}")
        setting_tex = f"{DS_NAME.get(ds, ds)}, " + \
                      ("clean" if noise == 0 else f"noisy $\\gamma{{=}}{noise}$")
        md.append(f"\n### {setting}\n")
        md.append("| Method | Final acc (%) | AUBC (%) |")
        md.append("|---|---|---|")
        tex.append("\\midrule")
        best = g.loc[g.aubc_mean.idxmax(), "method"]
        for m in METHOD_ORDER:
            r = g[g.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            fa = f"{100 * r.final_acc_mean:.2f} ± {100 * r.final_acc_std:.2f}"
            au = f"{100 * r.aubc_mean:.2f} ± {100 * r.aubc_std:.2f}"
            mark = " **(best)**" if m == best else ""
            md.append(f"| {LABELS[m]}{mark} | {fa} | {au} |")
            bold = (lambda s: f"\\textbf{{{s}}}") if m == best else (lambda s: s)
            # NB: keep backslash-bearing substitutions out of the f-string
            # expressions -- that is a SyntaxError before Python 3.12.
            pm = "$\\pm$"
            fa_tex = bold(fa.replace("±", pm))
            au_tex = bold(au.replace("±", pm))
            label = LABELS[m].replace("w/o", "w/o ")
            tex.append(f"{setting_tex} & {label} & {fa_tex} & {au_tex} \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")

    with open(os.path.join(RESULTS_DIR, "summary_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS_DIR, "summary_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("wrote summary tables")

    # ---- diagnostics: introspection vs entropy AUROC ----
    diag_path = os.path.join(RESULTS_DIR, "diagnostics.csv")
    if os.path.exists(diag_path):
        dg = pd.read_csv(diag_path)
        dg = dg.drop_duplicates(subset=["dataset", "method", "noise", "seed", "round"],
                                keep="last")
        dg = dg[dg.method == "iris"]
        if not dg.empty:
            ds_list = ds_sort(dg.dataset.unique())
            fig, axes = plt.subplots(1, len(ds_list),
                                     figsize=(5.0 * len(ds_list), 4),
                                     squeeze=False)
            for ax, ds in zip(axes[0], ds_list):
                g = dg[(dg.dataset == ds) & (dg.noise == 0.0)]
                piv_i = g.pivot_table(index="round", columns="seed",
                                      values="auroc_introspection")
                piv_e = g.pivot_table(index="round", columns="seed",
                                      values="auroc_entropy")
                ax.plot(piv_i.index, piv_i.mean(1), "o-", color="#d32f2f",
                        label="Introspection head")
                ax.plot(piv_e.index, piv_e.mean(1), "s--", color="#1976d2",
                        label="Softmax entropy")
                ax.set_xlabel("AL round")
                ax.set_ylabel("Error-prediction AUROC (test)")
                ax.set_title(DS_NAME.get(ds, ds))
                ax.grid(alpha=0.3)
                ax.legend(fontsize=9)
            fig.tight_layout()
            out = os.path.join(RESULTS_DIR, "diagnostics_auroc.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print("wrote", out)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
