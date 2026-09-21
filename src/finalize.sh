#!/bin/bash
# Watches the experiment log; when all runs are complete (or on each hourly tick),
# regenerates figures, tables, LaTeX macros, and recompiles the PDF.
# Launched with nohup so everything self-assembles even if the SSH session is gone.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/opt/miniconda/bin/python
LOG="$ROOT/logs/experiments.log"

assemble() {
  cd "$ROOT" || exit 1
  $PY src/make_plots.py   >> "$ROOT/logs/finalize.log" 2>&1
  $PY src/make_numbers.py >> "$ROOT/logs/finalize.log" 2>&1
  cd "$ROOT/paper" && pdflatex -interaction=nonstopmode main_en.tex > /dev/null 2>&1 \
                   && pdflatex -interaction=nonstopmode main_en.tex > /dev/null 2>&1
  echo "$(date) assembled" >> "$ROOT/logs/finalize.log"
}

# the grid is complete when all 102 (dataset,method,noise,seed) runs have a
# final-round (round 5) row in results.csv — robust to how many runner
# processes produced them
done_runs() {
  awk -F, 'NR>1 && $5==5 {print $1","$2","$3","$4}' \
    "$ROOT/results/results.csv" 2>/dev/null | sort -u | wc -l
}

while true; do
  if [ -f "$ROOT/results/results.csv" ]; then
    assemble
  fi
  if [ "$(done_runs)" -ge 102 ]; then
    assemble
    echo "$(date) FINAL assembly done ($(done_runs)/102 runs)" >> "$ROOT/logs/finalize.log"
    exit 0
  fi
  sleep 1800
done
