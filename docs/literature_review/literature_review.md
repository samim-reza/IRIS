# Human-in-the-Loop Machine Learning: Literature Review

**Focus:** recent (2026) HITL research in Q1 journals, the methodological foundations of
human-queried learning, and the gap that motivates our proposed method (IRIS).

All cited PDFs are in `../papers/`. Detailed per-paper metadata is in
`sources_nature.md`, `sources_springer_ieee.md`, and `sources_foundational.md`.

---

## 1. Scope and definitions

Human-in-the-loop (HITL) machine learning covers every paradigm in which human input is
consumed *during* the learning or decision process rather than only at dataset-creation
time (Wu et al., 2022, `wu_2022_hitl-survey.pdf`). Three sub-paradigms recur across the
literature:

1. **Active learning (AL)** — the model chooses which samples a human should annotate,
   maximizing accuracy per unit of annotation budget (Ren et al., 2021).
2. **Learning to defer (L2D)** — at deployment the system decides per instance whether
   the model or a human expert should produce the decision (Mozannar & Sontag, 2020;
   Alves et al., 2025).
3. **Learning from richer human feedback** — preferences, rankings, demonstrations, and
   explanation feedback rather than point labels (Ouyang et al., 2022 — RLHF).

Our contribution sits in (1), with an explicit bridge to (2): the component that decides
*which samples deserve human attention* is learned, and it must remain useful when the
human is imperfect.

## 2. What the 2026 Q1 literature is doing

### 2.1 HITL as expert-guided data curation and explanation (application-driven)

- **Petso et al., *Scientific Reports* 2026** (`petso_2026_hitl-wildlife-tracks.pdf`)
  put expert animal trackers in the loop of a wildlife-footprint classifier twice: experts
  curate/rank training images (mAP@50–95 +10.42%, 25% fewer training images needed), and
  XAI heatmaps assist human classification (98.67% vs 92.39% accuracy). Crucially, the
  *selection of which images deserve expert attention is manual*: quality ranking is done
  by humans over the entire pool. There is no model-driven acquisition and no treatment of
  expert disagreement/noise, which the authors themselves surface (three assessors of
  varying expertise disagreed systematically).

- **Lou et al., *npj Digital Medicine* 2026** (`lou_2026_morphoxai-hitl-pathology.pdf`)
  (MorphoXAI) fold pathologist interpretation of clustered high-attribution regions back
  into whole-slide-image model transparency. The human effort is again allocated by fixed
  clustering heuristics, not by any model-side estimate of *where the model is likely
  wrong*.

- **Feng et al., *npj Digital Medicine* 2026** (`feng_2026_hachi-codesign.pdf`) (HACHI)
  alternate AI-agent-driven statistical exploration with expert feedback cycles to build
  interpretable clinical prediction models. Expert time is the scarce resource; the paper
  reports that oversight quality determines model validity, but the framework has no
  quantitative policy for *rationing* that oversight.

- **Wang et al., *npj Digital Medicine* 2026** (`wang_2026_human-llm-collab.pdf`) — a
  PRISMA meta-analysis of human–LLM collaboration in clinical medicine — finds only
  modest, context-dependent gains from human-AI teaming (composite scores +4.88 pp) and
  *high residual factual error rates (26–36%)*. The clear message: naive human-AI
  coupling is not enough; the interface between model uncertainty and human effort has to
  be engineered.

- **Madduri et al., *Nature Machine Intelligence* 2026**
  (`madduri_2026_coadaptive-interfaces.pdf`) treat the human and the learner as a
  *co-adaptive closed-loop system* (control/game-theoretic) and show interface parameters
  deliberately shape user behaviour. This is the most formal 2026 statement that the
  human channel is a dynamic, imperfect system component to be modeled — not an oracle.

**Takeaway.** The 2026 application literature is rich in *workflows* that consume expert
time, and it consistently reports two pain points: (i) human attention is allocated by
hand-crafted or manual rules, and (ii) human input is noisy and heterogeneous. Neither
pain point is addressed by an architectural mechanism inside the learner.

### 2.2 The methodological base: deep active learning

The acquisition-function literature offers three families (Ren et al., 2021 survey):

- **Uncertainty-based:** softmax entropy; BALD with MC-dropout (Gal et al., 2017)
  approximates epistemic uncertainty via mutual information. Known failure modes:
  softmax miscalibration, preference for outliers/ambiguous samples, and redundancy
  within a batch.
- **Diversity/representativeness-based:** core-set / k-center coverage of feature space
  (Sener & Savarese, 2018). Ignores whether the model is actually wrong in a region.
- **Hybrid:** BADGE (Ash et al., 2020) combines uncertainty magnitude and diversity via
  k-means++ over hallucinated gradient embeddings.
- **Learned acquisition:** Learning-Loss (Yoo & Kweon, 2019) attaches a module that
  regresses the model's loss with a pairwise ranking objective; highest predicted loss is
  queried. This established that the *acquisition function itself can be a trained
  architectural component*, but (a) it predicts a real-valued loss, which is
  scale-unstable across training stages (hence their ranking trick), (b) it uses no
  diversity control, and (c) it assumes a perfect oracle.

### 2.3 The imperfect human: noisy oracles and deferral

- **Co-teaching** (Han et al., 2018) and the noisy-label literature exploit the
  memorization effect — deep nets fit clean patterns before noisy ones — to filter
  high-loss (likely mislabeled) samples during training. This literature is almost
  entirely *disjoint* from active learning: AL papers assume the returned label is
  correct.
- **Mozannar & Sontag (2020)** give consistent losses for learning when to defer to an
  expert whose errors are instance-dependent. **Alves et al. (2025, *Scientific Data*)**
  show with the OpenL2D/FiFAR benchmark that L2D algorithm rankings change substantially
  under realistic (noisy, capacity-limited) simulated experts — direct evidence that
  *evaluating HITL methods under an idealized human is misleading*.
- **RLHF** (Ouyang et al., 2022) scales human preference feedback, and in practice heavy
  inter-annotator disagreement is handled with reward-model ensembling and data curation —
  again, noise handled *outside* the acquisition mechanism.

### 2.4 The 2026 Springer / Elsevier / IEEE picture

- **Hussain et al., *Artificial Intelligence Review* 2026**
  (`hussain_2026_annotation_errors_survey.pdf`) — a 50-page data-centric survey of
  annotation errors in object-detection datasets — documents that human annotation error
  is pervasive, systematic, and largely unmodeled by the training pipelines that consume
  the labels. This is the strongest recent evidence for our noisy-oracle protocol.
- **Ning et al., *IJCV* 2026** (`ning_2026_openset_active_learning.pdf`) push acquisition
  design further (evidence-conflict sampling for open-set AL) but still assume a clean
  oracle.
- **Comito et al., *Neural Computing & Applications* 2026**
  (`comito_2026_dalek_active_learning.pdf`) (DALEK) combine deep AL with explanation
  methods for fake-news detection with experts in the loop — application-level evidence
  that AL + expert feedback is the operative HITL pattern in 2026 venues.
- **Piccoli et al., *Neural Computing & Applications* 2026**
  (`piccoli_2026_pytagit_annotation.pdf`) (PyTAGIT) build interactive HITL annotation
  tooling; **Pensel et al., *Knowledge-Based Systems* 2026**
  (`pensel_2026_humanguided_regression.pdf`) inject human guidance into transparent
  regression; **Jiang et al., *ACM Computing Surveys* 2025**
  (`jiang_2025_preference_learning_survey.pdf`) survey human preference learning for LLM
  alignment.
- Paywalled-but-relevant 2026 titles we could only read as abstracts (TPAMI's
  "The Value of Corrective Feedback in Online Active Learning" and "Learning From Crowds
  with Multiple-Feature Dynamic Fusion", plus Pattern Recognition work on
  evidential/noisy-oracle AL) confirm that noisy human feedback in AL is an *active,
  publishing-now* topic — i.e., a reviewer-relevant conversation our paper joins, while
  none of those combine a learned introspective acquisition head with diversity gating
  and reliability-weighted training in one architecture.

## 3. The gap

Putting 2.1–2.3 together:

| Requirement (from 2026 applications) | Uncertainty AL | Coreset | BADGE | Learning-Loss | Ours (IRIS) |
|---|---|---|---|---|---|
| Learned (not hand-crafted) estimate of where the model fails | ✗ | ✗ | ✗ | ✓ (loss regression) | ✓ (calibrated error probability) |
| Batch diversity control | ✗ | ✓ | ✓ | ✗ | ✓ (gated k-center) |
| Robust when the human annotator is noisy | ✗ | ✗ | ✗ | ✗ | ✓ (reliability-weighted training) |
| Evaluated under a noisy oracle | rarely | ✗ | ✗ | ✗ | ✓ (γ = 0.2 protocol) |

No standard deep-AL method combines a *learned* acquisition signal with *diversity*
control and *noise-robust* consumption of the labels it acquires — even though the 2026
application papers (Petso; Feng; Wang; Alves) independently establish that human
attention is scarce and human labels are imperfect. That intersection is the niche IRIS
occupies.

## 4. Our contribution in one paragraph

**IRIS (Introspective Reliability-gated Instance Selection)** modifies the classifier
architecture with a lightweight *introspection head*: a two-layer MLP on the (detached)
penultimate features, trained jointly with the backbone via class-balanced BCE to predict
the probability that the main head misclassifies the input — a bounded, calibrated,
scale-stable alternative to Learning-Loss's loss regression. At query time, the top
β·K samples by introspected error probability are filtered to K by k-center selection in
feature space (diversity gate), seeded with the already-labeled set. During training,
per-sample EMA losses drive a small-loss reliability weighting (co-teaching-style) so
that the labels bought from a noisy annotator do not poison the model — coupling, for the
first time in this line, the acquisition architecture with noise-robust consumption of
the acquired labels. The design costs <1% extra parameters and one extra scalar head
per forward pass.

## 5. References (files)

| File | Reference |
|---|---|
| `wu_2022_hitl-survey.pdf` | Wu et al., "A Survey of Human-in-the-loop for Machine Learning", Future Gener. Comput. Syst., 2022 |
| `ren_2021_dal-survey.pdf` | Ren et al., "A Survey of Deep Active Learning", ACM Comput. Surv., 2021 |
| `gal_2017_bald.pdf` | Gal, Islam, Ghahramani, ICML 2017 |
| `sener_2018_coreset.pdf` | Sener & Savarese, ICLR 2018 |
| `ash_2020_badge.pdf` | Ash et al., ICLR 2020 |
| `yoo_2019_learning-loss.pdf` | Yoo & Kweon, CVPR 2019 |
| `han_2018_coteaching.pdf` | Han et al., NeurIPS 2018 |
| `mozannar_2020_defer.pdf` | Mozannar & Sontag, ICML 2020 |
| `ouyang_2022_rlhf.pdf` | Ouyang et al., NeurIPS 2022 |
| `lou_2026_morphoxai-hitl-pathology.pdf` | Lou et al., npj Digit. Med. 9:579, 2026 |
| `petso_2026_hitl-wildlife-tracks.pdf` | Petso et al., Sci. Rep. 16:18395, 2026 |
| `madduri_2026_coadaptive-interfaces.pdf` | Madduri et al., Nat. Mach. Intell. 8:372–387, 2026 |
| `feng_2026_hachi-codesign.pdf` | Feng et al., npj Digit. Med., 2026 (in press) |
| `wang_2026_human-llm-collab.pdf` | Wang et al., npj Digit. Med. 9:195, 2026 |
| `alves_2025_learning-to-defer-benchmark.pdf` | Alves et al., Sci. Data 12:506, 2025 |
| `hussain_2026_annotation_errors_survey.pdf` | Hussain et al., Artif. Intell. Rev. 59:107, 2026 |
| `ning_2026_openset_active_learning.pdf` | Ning et al., Int. J. Comput. Vis. 134:42, 2026 |
| `comito_2026_dalek_active_learning.pdf` | Comito et al., Neural Comput. Appl. 38:19, 2026 |
| `piccoli_2026_pytagit_annotation.pdf` | Piccoli et al., Neural Comput. Appl. 38:226, 2026 |
| `pensel_2026_humanguided_regression.pdf` | Pensel et al., Knowl.-Based Syst. 350:116527, 2026 |
| `jiang_2025_preference_learning_survey.pdf` | Jiang et al., ACM Comput. Surv., 2025 |

## 6. Foundational and closely related references added for the extended survey (2026-09-18)

Added at the reviewer's request to broaden the related-work section (paper Table 1 now
summarises contribution *and* limitation for every work below). These are cited from
knowledge of the literature rather than from PDFs in `papers/` — **verify bibliographic
details before submission**.

| Group | Works | Why it is in the table |
|---|---|---|
| A. Acquisition | Lewis & Gale (SIGIR 1994); Settles (TR 1648, 2009); Houlsby et al. (arXiv 2011); Gal & Ghahramani (ICML 2016); Kirsch et al. (NeurIPS 2019, BatchBALD); Beluch et al. (CVPR 2018); Sinha et al. (ICCV 2019, VAAL) | Origins of uncertainty sampling, the Bayesian deep-AL line, and the diversity/redundancy problem our k-center gate addresses |
| A. Low-budget scepticism | Mittal et al. (arXiv 2019); Munjal et al. (CVPR 2022); Hacohen et al. (ICML 2022, TypiClust) | Independent evidence that AL loses to random at low budgets — the literature our CIFAR-10 cold-start result joins, and the basis of the competence gate |
| B. Failure prediction | Hendrycks & Gimpel (ICLR 2017); Corbière et al. (NeurIPS 2019, ConfidNet); Jiang et al. (NeurIPS 2018, trust scores); Guo et al. (ICML 2017, calibration) | The introspection head's closest architectural relatives; all are evaluated pointwise and none drives acquisition |
| C. Noisy labels | Zhang et al. (ICLR 2017); Arpit et al. (ICML 2017); Patrini et al. (CVPR 2017); Li et al. (ICLR 2020, DivideMix); Northcutt et al. (JAIR 2021); Song et al. (IEEE TNNLS 2022) | Memorisation effect underpinning the small-loss criterion, plus the robust-training alternatives we do not use and why |
| D. Crowds & deferral | Sheng et al. (KDD 2008); Raykar et al. (JMLR 2010); Yan et al. (ICML 2011); Madras et al. (NeurIPS 2018) | Annotator-aware methods that need repeated labels or defer decisions — the assumption our single-expert setting breaks |
| Dataset | Yang et al., MedMNIST v2 (Sci. Data 10:41, 2023) | Source of BloodMNIST, the third dataset |

Bibliography size: 22 → 46 entries, all cited in the text or Table 1.
