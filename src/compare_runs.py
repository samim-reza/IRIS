"""Compare the re-run against the previously recorded results.

The original grid was run without checkpointing and its models were lost.  It
was therefore re-executed with deterministic cuDNN and per-round checkpoints.
This script quantifies how far the re-run moved the published numbers, so the
paper can be updated (or confirmed unchanged) on evidence rather than hope.

    python src/compare_runs.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_plots import aubc as _aubc   # noqa: E402  -- same definition as the paper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, "results", "results.csv")
OLD = os.path.join(ROOT, "results", "previous_run", "results.csv")


def aubc(g):
    """Delegate to make_plots so the comparison uses the paper's definition."""
    g = g.sort_values("n_labeled")
    return _aubc(g["n_labeled"].to_numpy(float), g["test_acc"].to_numpy(float))


def summarise(path):
    d = pd.read_csv(path)
    fin = (d[d["round"] == d.groupby(["dataset", "method", "noise", "seed"])
             ["round"].transform("max")]
           .groupby(["dataset", "noise", "method"])["test_acc"]
           .mean().mul(100).rename("final"))
    au = (d.groupby(["dataset", "noise", "method", "seed"])
            .apply(aubc, include_groups=False).rename("aubc")
            .groupby(level=[0, 1, 2]).mean().mul(100))
    return pd.concat([fin, au], axis=1)


def main():
    for p in (OLD, NEW):
        if not os.path.exists(p):
            print(f"missing: {p}")
            return 1
    o, n = summarise(OLD), summarise(NEW)
    j = o.join(n, lsuffix="_old", rsuffix="_new", how="outer")
    j["d_final"] = j["final_new"] - j["final_old"]
    j["d_aubc"] = j["aubc_new"] - j["aubc_old"]

    print(f"{'dataset':<12}{'noise':>6}{'method':<13}"
          f"{'final old':>10}{'new':>9}{'delta':>8}"
          f"{'  |  AUBC old':>14}{'new':>9}{'delta':>8}")
    print("-" * 92)
    for (ds, nz, m), r in j.iterrows():
        print(f"{ds:<12}{nz:>6}{m:<13}"
              f"{r.final_old:>10.2f}{r.final_new:>9.2f}{r.d_final:>+8.2f}"
              f"{r.aubc_old:>14.2f}{r.aubc_new:>9.2f}{r.d_aubc:>+8.2f}")
    print("-" * 92)
    print(f"final accuracy : mean |delta| = {j.d_final.abs().mean():.3f} pts, "
          f"max = {j.d_final.abs().max():.3f} pts")
    print(f"AUBC           : mean |delta| = {j.d_aubc.abs().mean():.3f} pts, "
          f"max = {j.d_aubc.abs().max():.3f} pts")

    # Did any conclusion flip?  Only the winner per (dataset, noise) matters.
    print("\nbest method per scenario (the claims the paper actually makes):")
    for metric in ("final", "aubc"):
        print(f"  by {metric}:")
        for (ds, nz), g in j.groupby(level=[0, 1]):
            wo = g[f"{metric}_old"].idxmax()[2]
            wn = g[f"{metric}_new"].idxmax()[2]
            flag = "  <-- CHANGED" if wo != wn else ""
            print(f"    {ds:<12} noise={nz:<5} {wo:<12} -> {wn:<12}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
