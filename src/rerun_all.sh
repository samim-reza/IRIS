#!/usr/bin/env bash
# Re-execute the complete 174-run grid, keeping the FINAL trained model of
# every run (the one that produced that run's published accuracy).
#   seeds 0-2 on all three datasets        -> 36 + 36 + 30 = 102 runs
#   seeds 3-5 on the two 28x28 datasets    -> 36 + 36      =  72 runs
#                                                     total = 174 runs
#
# Intermediate rounds are NOT kept: their accuracies are already in
# results.csv, and 1,044 files serve nobody.  Pass --save-all-rounds to
# hitl_experiments.py directly if you ever need them.
#
# Run it detached so it survives an SSH disconnect:
#     tmux new-session -d -s iris 'bash src/rerun_all.sh > logs/rerun.log 2>&1'
#     tmux attach -t iris          # watch
#     Ctrl-b d                     # detach again
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/miniconda/envs/py10/bin/python}"
cd "$ROOT" || exit 1

echo "=== RERUN START $(date) ==="
echo "--- environment (recorded for provenance) ---"
# Import the experiment module, not bare torch: the determinism flags are set
# at import time, so a bare-torch probe would report misleading defaults.
"$PY" - <<'PYV'
import sys
sys.path.insert(0, "src")
import torch, torchvision
import hitl_experiments  # noqa: F401  -- sets the cuDNN determinism flags
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda       :", torch.version.cuda, "| available:", torch.cuda.is_available())
print("device     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("cudnn deterministic:", torch.backends.cudnn.deterministic,
      "| benchmark:", torch.backends.cudnn.benchmark)
PYV
echo "---------------------------------------------"

"$PY" -u src/hitl_experiments.py \
    --datasets fmnist cifar10 bloodmnist --seeds 0 1 2 \
    --save-checkpoints
echo "=== seeds 0-2 done $(date) ==="

"$PY" -u src/hitl_experiments.py \
    --datasets fmnist bloodmnist --seeds 3 4 5 \
    --save-checkpoints
echo "=== seeds 3-5 done $(date) ==="
echo "=== RERUN COMPLETE $(date) ==="
