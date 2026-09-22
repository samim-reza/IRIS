"""Check the paper's qualitative claims against results/results.csv.

The numbers in the manuscript are macros, so they cannot go stale. The
*claims* around them can: "IRIS ranks last", "the single largest ablation
effect", "BALD falls below random". Those are English, and English does not
regenerate. This script asserts each one against the data, so a re-run that
quietly inverts a conclusion fails loudly instead of leaving a sentence that
contradicts the table beside it.

    python src/check_claims.py        # exit 0 = every claim still holds
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from make_plots import aubc  # noqa: E402  -- the paper's AUBC definition

BASE = ["random", "entropy", "bald", "coreset"]
WITH_IRIS = BASE + ["iris"]


def load():
    d = pd.read_csv(os.path.join(ROOT, "results", "results.csv"))
    last = d["round"].max()
    fin = d[d["round"] == last].groupby(
        ["dataset", "noise", "method"]).test_acc.mean().mul(100)
    au = (d.groupby(["dataset", "noise", "method", "seed"])
            .apply(lambda g: aubc(g.n_labeled.values, g.test_acc.values),
                   include_groups=False)
            .groupby(level=[0, 1, 2]).mean().mul(100))
    return fin, au


def main():
    fin, au = load()
    dg = pd.read_csv(os.path.join(ROOT, "results", "diagnostics.csv"))
    dg = dg[dg["round"] == dg["round"].max()]
    drop = lambda ds, nz, ab: fin.loc[ds, nz, "iris"] - fin.loc[ds, nz, ab]

    claims = [
        # --- central claim: the reliability gate transfers to another acquirer
        ("Reliability gate lifts BADGE on Fashion-MNIST",
         fin.loc["fmnist", 0.2, "badge-rel"] > fin.loc["fmnist", 0.2, "badge"]),
        ("Reliability gate lifts BADGE on BloodMNIST",
         fin.loc["bloodmnist", 0.2, "badge-rel"] > fin.loc["bloodmnist", 0.2, "badge"]),
        ("Reliability gate lifts BADGE on CIFAR-10",
         fin.loc["cifar10", 0.2, "badge-rel"] > fin.loc["cifar10", 0.2, "badge"]),
        ("BADGE+reliability is the best configuration on all three (noisy)",
         all(fin.loc[d, 0.2].idxmax() == "badge-rel"
             for d in ("fmnist", "bloodmnist", "cifar10"))),
        ("BADGE+reliability is the only thing beating random at cold start",
         fin.loc["cifar10", 0.2, "badge-rel"] > fin.loc["cifar10", 0.2, "random"]
         and all(fin.loc["cifar10", 0.2, m] < fin.loc["cifar10", 0.2, "random"]
                 for m in WITH_IRIS if m != "random")),
        # --- the honest negative, reported rather than hidden
        ("IRIS beats Learning-Loss on all three datasets (noisy)",
         all(fin.loc[d, 0.2, "iris"] > fin.loc[d, 0.2, "learnloss"]
             for d in ("fmnist", "bloodmnist", "cifar10"))),
        ("BADGE beats IRIS on both capable datasets",
         fin.loc["fmnist", 0.2, "badge"] > fin.loc["fmnist", 0.2, "iris"]
         and fin.loc["bloodmnist", 0.2, "badge"] > fin.loc["bloodmnist", 0.2, "iris"]),
        ("IRIS-grad did not close the gap to BADGE",
         fin.loc["fmnist", 0.2, "iris-grad"] < fin.loc["fmnist", 0.2, "badge"]),
        ("IRIS beats every uncertainty/coverage baseline (F-MNIST noisy)",
         all(fin.loc["fmnist", 0.2, "iris"] > fin.loc["fmnist", 0.2, m]
             for m in ("random", "entropy", "bald", "coreset", "learnloss"))),
        ("BloodMNIST: random still beats IRIS on AUBC while losing on final "
         "(the metric disagreement, now restricted to point-wise scorers)",
         au.loc["bloodmnist", 0.2, "random"] > au.loc["bloodmnist", 0.2, "iris"]
         and fin.loc["bloodmnist", 0.2, "random"] < fin.loc["bloodmnist", 0.2, "iris"]),
        ("BloodMNIST clean: IRIS has the lowest AUBC",
         au.loc["bloodmnist", 0.0][WITH_IRIS].idxmin() == "iris"),
        ("BloodMNIST clean: BALD, Entropy and IRIS all beat random",
         all(fin.loc["bloodmnist", 0.0, m] > fin.loc["bloodmnist", 0.0, "random"]
             for m in ("bald", "entropy", "iris"))),
        ("BALD drops below random under noise (Fashion-MNIST)",
         fin.loc["fmnist", 0.2, "bald"] < fin.loc["fmnist", 0.2, "random"]),
        ("BALD drops below random under noise (BloodMNIST)",
         fin.loc["bloodmnist", 0.2, "bald"] < fin.loc["bloodmnist", 0.2, "random"]),
        ("BALD is worst of all on BloodMNIST noisy",
         fin.loc["bloodmnist", 0.2][WITH_IRIS].idxmin() == "bald"),
        ("CIFAR-10 clean: random beats every acquisition method",
         fin.loc["cifar10", 0.0][WITH_IRIS].idxmax() == "random"),
        ("CIFAR-10 noisy: random beats every acquisition method",
         fin.loc["cifar10", 0.2][WITH_IRIS].idxmax() == "random"),
        ("CIFAR-10 noisy: IRIS is NOT last (paper no longer claims it is)",
         fin.loc["cifar10", 0.2][WITH_IRIS].idxmin() != "iris"),
        ("CIFAR-10 clean: removing the diversity gate makes it worse",
         fin.loc["cifar10", 0.0, "iris-nodiv"] < fin.loc["cifar10", 0.0, "iris"]),
        ("BloodMNIST clean diversity ablation is the single largest effect",
         drop("bloodmnist", 0.0, "iris-nodiv") > max(
             drop("fmnist", 0.0, "iris-nodiv"),
             drop("fmnist", 0.2, "iris-norel"),
             drop("bloodmnist", 0.2, "iris-norel"),
             drop("cifar10", 0.0, "iris-nodiv"))),
        ("Both ablations hurt on both capable datasets",
         all(drop(ds, nz, ab) > 0 for ds, nz, ab in (
             ("fmnist", 0.0, "iris-nodiv"), ("bloodmnist", 0.0, "iris-nodiv"),
             ("fmnist", 0.2, "iris-norel"), ("bloodmnist", 0.2, "iris-norel")))),
        ("Entropy out-ranks the introspection head point-wise on all three",
         all(dg[dg.dataset == x].auroc_entropy.mean()
             > dg[dg.dataset == x].auroc_introspection.mean()
             for x in ("fmnist", "bloodmnist", "cifar10"))),
    ]

    failed = 0
    for name, ok in claims:
        if not ok:
            failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{len(claims) - failed}/{len(claims)} claims hold")
    if failed:
        print("A claim in the manuscript no longer matches the data. "
              "Fix the prose, not this file.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
