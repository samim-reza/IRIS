"""Emit results/summary_table_bn.tex: the Bengali edition of the main results
table, built from the same summary.csv as the English one so the two papers can
never drift apart."""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS_DIR, "summary_table_bn.tex")

DS_ORDER = ["fmnist", "bloodmnist", "cifar10"]
DS_NAME = {"fmnist": "Fashion-MNIST", "bloodmnist": "BloodMNIST",
           "cifar10": "CIFAR-10"}
METHOD_ORDER = ["random", "entropy", "bald", "coreset", "iris",
                "iris-nodiv", "iris-norel"]
LABELS_BN = {
    "random": "র‍্যান্ডম", "entropy": "এনট্রপি", "bald": "BALD",
    "coreset": "কোর-সেট", "iris": "\\method{} (আমাদের)",
    "iris-nodiv": "\\method{}, বৈচিত্র্য-গেট ছাড়া",
    "iris-norel": "\\method{}, নির্ভরযোগ্যতা-ভার ছাড়া",
}


def main():
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    tex = ["\\begin{tabular}{llcc}\n\\toprule",
           "পরিস্থিতি & পদ্ধতি & চূড়ান্ত নির্ভুলতা (\\%) & AUBC (\\%) \\\\"]

    settings = sorted(summary.groupby(["dataset", "noise"]).groups,
                      key=lambda k: (DS_ORDER.index(k[0])
                                     if k[0] in DS_ORDER else 99, k[1]))
    for ds, noise in settings:
        g = summary[(summary.dataset == ds) & (summary.noise == noise)]
        oracle = ("নির্ভুল ওরাকল" if noise == 0
                  else f"নয়েজি ওরাকল $\\gamma{{=}}{noise}$")
        setting = f"{DS_NAME.get(ds, ds)}, {oracle}"
        tex.append("\\midrule")
        best = g.loc[g.aubc_mean.idxmax(), "method"]
        for m in METHOD_ORDER:
            r = g[g.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            fa = f"{100 * r.final_acc_mean:.2f} $\\pm$ {100 * r.final_acc_std:.2f}"
            au = f"{100 * r.aubc_mean:.2f} $\\pm$ {100 * r.aubc_std:.2f}"
            bold = (lambda s: f"\\textbf{{{s}}}") if m == best else (lambda s: s)
            tex.append(f"{setting} & {LABELS_BN[m]} & "
                       f"{bold(fa)} & {bold(au)} \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")

    with open(OUT, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
