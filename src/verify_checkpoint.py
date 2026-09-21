"""Reload saved checkpoints and re-measure the accuracy they claim.

This is the audit trail for the paper: every number in results.csv came from a
model that was evaluated once and then thrown away.  With --save-checkpoints
the final model of each run is kept, and this script re-evaluates it on the
untouched test set and reports the difference against the accuracy recorded at
training time.  A clean run prints a max |delta| of 0.0000.

    python src/verify_checkpoint.py                    # every checkpoint
    python src/verify_checkpoint.py --dataset bloodmnist
    python src/verify_checkpoint.py --ckpt checkpoints/<one>.pt
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitl_experiments as H  # noqa: E402


def verify(path, cache):
    ck = torch.load(path, map_location=H.DEVICE, weights_only=False)
    ds = ck["dataset"]
    if ds not in cache:
        cache[ds] = H.get_datasets(ds)[2]          # test split only
    backbone, head = H.make_model(ds, 0)
    backbone.load_state_dict(ck["backbone_state_dict"])
    head.load_state_dict(ck["head_state_dict"])
    acc = H.evaluate(backbone, cache[ds])
    return ck, acc, acc - ck["test_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=H.CKPT_DIR)
    ap.add_argument("--ckpt", help="verify a single checkpoint file")
    ap.add_argument("--dataset", help="restrict to one dataset")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    paths = ([args.ckpt] if args.ckpt
             else sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt"))))
    if args.dataset:
        paths = [p for p in paths
                 if os.path.basename(p).startswith(args.dataset + "_")]
    if not paths:
        print("no checkpoints found — run hitl_experiments.py "
              "--save-checkpoints first")
        return 1

    print(f"verifying {len(paths)} checkpoint(s) on {H.DEVICE}\n")
    print(f"{'checkpoint':<52}{'recorded':>10}{'reloaded':>10}{'delta':>10}")
    print("-" * 82)
    cache, worst, bad = {}, 0.0, 0
    for p in paths:
        ck, acc, d = verify(p, cache)
        worst = max(worst, abs(d))
        flag = "" if abs(d) <= args.tol else "  <-- MISMATCH"
        if flag:
            bad += 1
        print(f"{os.path.basename(p):<52}{ck['test_acc']:>10.4f}"
              f"{acc:>10.4f}{d:>+10.4f}{flag}")
    print("-" * 82)
    print(f"max |delta| = {worst:.6f}   mismatches = {bad}/{len(paths)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
