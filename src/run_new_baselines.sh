#!/usr/bin/env bash
# Add Learning-Loss and BADGE to the grid.  Everything already recorded in
# results/results.csv is skipped, so this only runs the 90 new combinations:
#   2 methods x 2 oracles x (6 F-MNIST + 6 BloodMNIST + 3 CIFAR-10 seeds)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/miniconda/envs/py10/bin/python}"
cd "$ROOT" || exit 1
echo "=== NEW BASELINES START $(date) ==="
"$PY" -u src/hitl_experiments.py --datasets fmnist cifar10 bloodmnist \
      --seeds 0 1 2 --save-checkpoints
echo "=== seeds 0-2 done $(date) ==="
"$PY" -u src/hitl_experiments.py --datasets fmnist bloodmnist \
      --seeds 3 4 5 --save-checkpoints
echo "=== NEW BASELINES COMPLETE $(date) ==="
