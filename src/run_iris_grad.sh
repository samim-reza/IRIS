#!/usr/bin/env bash
# IRIS with the diversity gate moved into BADGE's gradient-embedding geometry.
# Completed runs are skipped, so this is safe to re-invoke after an interruption.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/miniconda/envs/py10/bin/python}"
cd "$ROOT" || exit 1
echo "=== IRIS-GRAD START $(date) ==="
"$PY" -u src/hitl_experiments.py --datasets fmnist cifar10 bloodmnist \
      --seeds 0 1 2 --save-checkpoints
"$PY" -u src/hitl_experiments.py --datasets fmnist bloodmnist \
      --seeds 3 4 5 --save-checkpoints
echo "ALL_GRAD_DONE $(date)"
