#!/usr/bin/env bash
# BADGE acquisition + our reliability gate, noisy oracle only.
# Isolates the label-acceptance contribution from the acquisition signal.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/miniconda/envs/py10/bin/python}"
cd "$ROOT" || exit 1
echo "=== BADGE-REL START $(date) ==="
"$PY" -u src/hitl_experiments.py --datasets fmnist cifar10 bloodmnist \
      --seeds 0 1 2 --save-checkpoints
"$PY" -u src/hitl_experiments.py --datasets fmnist bloodmnist \
      --seeds 3 4 5 --save-checkpoints
echo "ALL_BADGEREL_DONE $(date)"
