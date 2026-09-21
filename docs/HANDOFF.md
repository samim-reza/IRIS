# Handoff — IRIS, state as of 2026-09-22

Written for whoever picks this up next (human or agent). It covers what the
project is, what was changed in the session of 2026-09-21/22, what is verified,
and what is still open.

---

## What this project is

**IRIS** (Introspective Reliability-gated Instance Selection) — a lightweight
extension to a deep classifier for human-in-the-loop active learning with an
imperfect annotator. Three parts: an **introspection head** (2-layer MLP on
detached penultimate features, predicts "am I about to be wrong?"), a
**k-center diversity gate** over its top-ranked candidates, and **small-loss
reliability weighting** of labels returned by a noisy oracle.

Deliverable is a journal manuscript in Elsevier CAS format plus a repo that
reproduces every number in it.

---

## Current state — everything below is verified, not assumed

| Thing | State |
|---|---|
| Manuscript | `paper/main_en.tex` → 20 pages, `cas-sc.cls`, 0 LaTeX errors, 0 undefined refs |
| Experiment grid | **180 runs**, 0 failures (72 F-MNIST + 72 BloodMNIST + 36 CIFAR-10, ablations included) |
| Checkpoints | 180 final models, 2.0 GB, **git-ignored**; all 180 re-verify at `max \|delta\| = 0.000000` |
| Numeric audit | all 71 distinct `NN.NN` values in the PDF trace to `results/summary.csv` |
| Claim audit | `python src/check_claims.py` → **17/17 pass** |
| Determinism | proven — re-running a completed run gives byte-identical per-round accuracies |
| Repo | `git@github.com:samim-reza/IRIS.git`, branch `main`, clean |
| Submission bundle | `prism_submission.zip` (2.2 MB), flat, test-compiled in an empty dir |

**Environment matters.** Use `/opt/miniconda/envs/py10/bin/python`
(3.10, torch 2.12.0+cu130). The other interpreter on this box is torch 2.11 and
**gives different accuracies for the same seed** — never split a grid across the
two. Note `make_plots.py` had a Python-3.12-only f-string; it is fixed, but keep
3.10 compatibility in mind when editing `src/`.

---

## What happened in this session, and why

1. **Reformatted to Elsevier CAS**, rewrote the abstract, removed every `§`,
   deleted the "How to read this paper" guide, split Datasets from Protocol
   (budget moved out of the dataset table), converted 46 manual `\bibitem`s to
   `paper/refs.bib` with natbib author-year.
2. **Restructured the repo** (`code/`→`src/`, `latex/`→`paper/`, datasets to
   `data/`), renamed the folder `HITL`→`IRIS`, wrote the README, pushed.
3. **Discovered no models had ever been saved** — active learning retrains from
   scratch each round and the code had no `torch.save` at all. So the whole grid
   was re-run with checkpointing, determinism, and a divergence guard.
4. **The re-run moved several conclusions.** They were corrected in the paper
   (see below).
5. **Audited every hand-typed number** and found three that the macro system had
   been hiding.

### Conclusions that changed, and are now corrected in the manuscript

- **Significance swapped datasets.** Fashion-MNIST is now the resolved case
  (85.68%, 5/6 seeds, *p*=0.021, 17% label saving); BloodMNIST is a trend
  (87.82%, 5/6, *p*=0.097 — was 6/6, *p*=0.003).
- **Dropped "highest nominal accuracy"** on clean BloodMNIST (IRIS is third and
  statistically tied) and **"highest AUBC"** on clean Fashion-MNIST.
- **CIFAR-10: IRIS no longer ranks last** (62.20, 2nd) and the gap to random is
  not significant (*p*=0.182).
- **The claim "reliability weighting actively hurts at cold start" was refuted.**
  It had only ever been *inferred*. Both ablations were run on CIFAR-10:
  removing reliability weighting changes nothing (*p*=0.850), removing the
  diversity gate makes things worse. The cold-start failure belongs to the
  **learned selection signal**. The paper now says that.
- **"Under 1% extra parameters" was false** (and was false before the re-run).
  The head is 0.6% of a ResNet-18 but **3.7%** of the small CNN. Both are now
  stated and auto-generated.

### Unchanged (the thesis held)

BALD still collapses below random under noise; random still wins the cold start
on both metrics; entropy still out-ranks the introspection head point-wise while
picking worse batches; both gates still earn their place on capable datasets.

---

## Tooling added — use it

```bash
python src/check_claims.py                 # 17 qualitative claims vs the data
python src/verify_checkpoint.py            # reload all 180 models, re-measure
python src/compare_runs.py                 # this grid vs results/previous_run/
bash   src/rerun_all.sh                    # full grid, in tmux, with checkpoints
```

`src/make_numbers.py` emits **214 macros** into `paper/numbers.tex`. **No number
in the manuscript should be typed by hand** — that is exactly how the stale
round-0 accuracies and the false parameter claim survived. If you need a new
figure in the prose, add a macro.

`src/check_claims.py` exists because macros protect numbers but not *sentences*.
Run it after any re-run: a conclusion that silently inverts will otherwise leave
prose contradicting the table next to it.

---

## Gotchas that cost time here

- **A `--quick` smoke test appends junk rows to `results/results.csv`.** It
  polluted the comparison baseline once. Recover a clean copy with
  `git show <sha>:results/results.csv`.
- **Never `pkill -f hitl_experiments.py`** — the pattern matches the invoking
  shell and kills it. Select PIDs with
  `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /hitl_experiments/ {print $1}'`.
- **Scan a fresh grid for collapsed runs before trusting any mean.** One run
  (BloodMNIST clean/random seed 4) diverged to 19.5% and dragged that baseline
  down 11 points — which *flattered* IRIS. The divergence guard now catches this
  from **training** accuracy only, never the test set.
- **Ollama shares this GPU** and is driven by another user (`/home/gub/.ollama`);
  its processes cannot be killed from `user1` and `sudo` needs a password. Do
  not fight it — our runs need ~1.8 GB of a 24 GB card, which is why
  `min_free_mb` is 2300 and the GPU poll is 15 s.
- **CAS floats take key-value options**: `\begin{table}[pos=t]`, never `[t]`.
- **`cas-sc.cls` v2.4 calls `\vbox_unpack_clear:N`**, removed from modern LaTeX3;
  the preamble carries a shim. The 117 pt overfull box at `\maketitle` is
  produced by Elsevier's own sample file — do not chase it.

---

## Open items

1. **Checkpoint hosting.** 2.0 GB, git-ignored. Git LFS would need a paid data
   pack (GitHub gives 1 GB free, and bandwidth is metered). Recommended:
   **Zenodo** (50 GB free, mints a citable DOI for the Data Availability
   statement) or a **GitHub Release** with just the 501 MB Fashion-MNIST +
   BloodMNIST subset. Not yet done — awaiting the author's choice.
2. **`paper/refs.bib` needs verification.** 24 of the 50 entries were written
   from model knowledge, not from PDFs. Author lists/venues/years are believed
   right; volumes and pages were mostly omitted. One entry (`jiang2025`) has no
   volume and renders with a stray space. **Verify before submission.**
3. **Front matter placeholders** in `paper/main_en.tex`: affiliation, e-mail and
   ORCID are intentionally blank (empty CAS keys print stray commas).
4. **Seed count.** Six seeds cannot resolve a sub-point margin, which is why
   BloodMNIST sits at *p*=0.097. Raising both 28×28 datasets to ~12 seeds would
   settle it — but only if the result is reported either way. Stopping once
   BloodMNIST crosses 0.05 would be p-hacking and a reviewer can detect it.

---

## Reproduce from nothing

```bash
git clone git@github.com:samim-reza/IRIS.git && cd IRIS
pip install -r requirements.txt
python src/make_plots.py && python src/make_numbers.py   # no GPU, no training
cd paper && make                                          # -> main_en.pdf
```

A fresh clone regenerates the committed figures **bit-identically**; that was
verified, not assumed.
