# Sources: HITL Machine Learning in Q1 Springer / Elsevier / IEEE / ACM Journals (2026, late 2025)

Compiled 2026-08-23. All papers below were verified against their publisher landing pages / publisher-served PDFs, and every listed PDF was downloaded into `/home/user1/greenresearch/samim/HITL/papers/` and validated (`%PDF` magic, size > 100 KB).

---

## 1. Quality over Quantity: A Data-Centric Survey of Annotation Errors in Object Detection Datasets

- **Authors:** Adnan Hussain, Kaleem Ullah, Muhammad Afaq, Muhammad Munsif, Altaf Hussain, Sung Wook Baik
- **Journal:** Artificial Intelligence Review (Springer), vol. 59, article 107
- **Year:** 2026 (published online 7 February 2026)
- **DOI:** 10.1007/s10462-026-11502-z
- **URL:** https://link.springer.com/article/10.1007/s10462-026-11502-z (open access, © The Authors 2026)
- **Downloaded file:** `hussain_2026_annotation_errors_survey.pdf` (4.9 MB, 50 pp., publisher PDF)
- **Summary:** A data-centric survey of annotation errors in object detection benchmarks — the label-noise problem as it actually arises from human annotators (inaccurate bounding boxes, misclassified objects, and missing labels). The authors argue that even widely used benchmarks suffer substantial annotation defects, with localization errors in particular distorting both training and evaluation of detectors. The survey systematically reviews error taxonomies, methods for detecting and correcting annotation errors, and robust-training approaches that tolerate noisy labels. Its data-centric framing (improve the labels, not just the model) connects directly to human-in-the-loop annotation quality control and label-noise-from-annotators research. Relevant here as the most recent Q1-journal survey treatment of human labeling errors in vision datasets.

## 2. PyTAGIT: A Scalable and Interactive Human-in-the-Loop Tool for Fast Image Annotation

- **Authors:** Flavio Piccoli, Claudio Rota, Rajesh Kumar, Gianluigi Ciocca
- **Journal:** Neural Computing and Applications (Springer), vol. 38, article 226
- **Year:** 2026 (accepted 2 December 2025; published online 25 March 2026)
- **DOI:** 10.1007/s00521-025-11725-1
- **URL:** https://link.springer.com/article/10.1007/s00521-025-11725-1
- **Downloaded file:** `piccoli_2026_pytagit_annotation.pdf` (3.5 MB, 24 pp., publisher PDF)
- **Summary:** Introduces PyTAGIT, an open-source human-in-the-loop annotation framework for fast, scalable multi-class image labeling. The method combines AI-assisted classification with interactive mechanisms — drag-and-drop label assignment, t-SNE-based exploration of the embedding space, and iterative refinement via confidence thresholds — balancing automation with expert supervision. It runs efficiently on mid-range laptops and scales to thousands of samples and dozens of classes. In experiments on seven diverse datasets, PyTAGIT achieved the best annotation accuracy on six (second-best on the seventh), outperforming manual tagging and vision-language baselines (CLIP, LLM+CLIP, BLIP-2), and consistently annotated full datasets within a one-hour budget. A concrete 2026 example of human-AI collaboration for annotation in a Q1 Springer journal.

## 3. DALEK: Combining Deep Active Learning and Explanations Methods for Fake News Detection on COVID-19

- **Authors:** Carmela Comito, Massimo Guarascio, Angelica Liguori, Francesco Sergio Pisani
- **Journal:** Neural Computing and Applications (Springer), vol. 38, article 19
- **Year:** 2026 (accepted 18 September 2025; published online 24 January 2026)
- **DOI:** 10.1007/s00521-025-11830-1
- **URL:** https://link.springer.com/article/10.1007/s00521-025-11830-1
- **Downloaded file:** `comito_2026_dalek_active_learning.pdf` (2.4 MB, 21 pp., publisher PDF)
- **Summary:** Proposes a neural (deep) active learning framework with explanation capabilities for detecting health misinformation, focused on COVID-19 fake news. Active learning strategically selects the most informative unlabeled instances for domain-expert labeling, addressing the scarcity of labeled examples that limits neural fake-news detectors. Explanation (XAI) methods play a dual human-in-the-loop role: they support the expert during labeling and, via Explanatory Active Learning (explanatory interactive learning, XIL), guide the query-selection strategy itself. Experiments on real health-domain datasets on COVID-19 misinformation show the framework effectively detects and mitigates fake-news spread. Illustrates the coupling of deep active learning, XAI, and expert feedback in a 2026 Q1 application paper.

## 4. Evidence Conflict Sampling for Open-set Active Learning

- **Authors:** Kun-Peng Ning, Hai-Jian Ke, Jia-Yu Yao, Yu-Yang Liu, Yong-Hong Tian, Li Yuan
- **Journal:** International Journal of Computer Vision (Springer), vol. 134, article 42
- **Year:** 2026 (accepted 24 November 2025; published online 7 January 2026)
- **DOI:** 10.1007/s11263-025-02600-6
- **URL:** https://link.springer.com/article/10.1007/s11263-025-02600-6
- **Downloaded file:** `ning_2026_openset_active_learning.pdf` (1.9 MB, 14 pp., publisher PDF)
- **Summary:** Addresses deep active learning under open-set conditions, where the unlabeled pool contains substantial noise from unknown classes and standard entropy-based query strategies cannot separate informative examples from open-set noise. The proposed Evidence COnflict (ECO) sampling uses a naive Bayes framework to quantify class-wise evidence and queries examples that have both sufficient evidence for known classes (purity) and conflicting evidence among classes (informativeness). This dual evidence-driven criterion filters out open-set noise before spending human annotation budget on it. The authors give a theoretical foundation, proving the posterior variance of the evidence distribution grows with the label space, supporting robustness in open-set scenarios. Experiments across benchmarks with varying openness ratios show superiority over state-of-the-art open-set AL methods in query efficiency and robustness — directly relevant to making human labeling loops budget-efficient.

## 5. Human Guided Learning of Transparent Regression Models

- **Authors:** Lukas Pensel, Stefan Kramer
- **Journal:** Knowledge-Based Systems (Elsevier), vol. 350, article 116527
- **Year:** 2026 (issue dated September 2026)
- **DOI:** 10.1016/j.knosys.2026.116527
- **URL:** https://doi.org/10.1016/j.knosys.2026.116527 (journal version, subscription); open preprint: https://arxiv.org/abs/2502.15992
- **Downloaded file:** `pensel_2026_humanguided_regression.pdf` (1.3 MB, 14 pp., arXiv preprint PDF — journal version is paywalled)
- **Summary:** Presents HuGuR (Human Guided Regression), a human-in-the-loop / interactive machine learning approach to permutation regression — predicting a continuous value for a given ordering of items. The model is a gradient-boosted regression model whose features are simple human-understandable order constraints of the form "item x before item y"; users interactively add, remove, and refine constraints while coefficients are recomputed on the fly, letting a human explore the space of transparent regression models. A user study compares user-built models with multiple baselines on 9 datasets: user-built models outperform the compared methods on small datasets and are generally on par elsewhere, while remaining human-understandable; on larger datasets machine-induced models begin to win. A clean 2026 Q1-journal example of interactive machine learning with genuine human model-shaping rather than mere label provision.

## 6. A Survey on Human Preference Learning for Aligning Large Language Models

- **Authors:** Ruili Jiang, Kehai Chen, Xuefeng Bai, Zhixuan He, Juntao Li, Muyun Yang, Tiejun Zhao, Liqiang Nie, Min Zhang
- **Journal:** ACM Computing Surveys
- **Year:** 2025 (published online 6 December 2025 — late 2025, included as directly on-topic for human feedback)
- **DOI:** 10.1145/3773279
- **URL:** https://doi.org/10.1145/3773279 (journal version); open preprint: https://arxiv.org/abs/2406.11191 (preprint title omits "Aligning"; identical 9-author list verified via Crossref)
- **Downloaded file:** `jiang_2025_preference_learning_survey.pdf` (0.7 MB, 18 pp., arXiv preprint PDF v2 — ACM version could not be retrieved from this environment)
- **Summary:** A survey of how human preferences — the core human-feedback signal of modern LLM alignment — are introduced into large language models, taken from a preference-centered perspective. It categorizes human feedback by data source and format, then reviews techniques for modeling preference signals (e.g., reward models and their alternatives), comparing advantages and disadvantages of the different schools. It further organizes preference-usage methods by objective (how the preference signal steers training, including RLHF-style pipelines) and summarizes evaluation approaches for measuring alignment of LLMs with human intentions. The survey closes with limitations and outlooks on human-intention alignment. Included as the most relevant late-2025 Q1-journal survey on learning from human feedback; note the downloaded PDF is the June 2024 arXiv v2 preprint of the ACM Computing Surveys article.

---

## Papers found but NOT downloadable (paywalled with no preprint, or publisher blocked automated access)

These were identified via Crossref/OpenAlex/publisher pages during the same search and are strong on-topic 2026 items, listed here for follow-up via institutional access:

- **The Value of Corrective Feedback in the Online Active Learning Paradigm** — M. Lindsey, F. Kubala, R. M. Stern, IEEE TPAMI (March 2026), DOI 10.1109/TPAMI.2025.3639522, IEEE doc 11274545. Hybrid open access per OpenAlex, but IEEE Xplore and the Computer Society DL both block non-browser downloads; no arXiv preprint exists.
- **Active Learning for Multiple Target Models** — S.-J. Huang, Y. Li, Y.-P. Tang, IEEE TPAMI (January 2026), DOI 10.1109/TPAMI.2025.3602682. Paywalled; no arXiv preprint of the journal version (an earlier NeurIPS 2022 conference version is open at proceedings.neurips.cc).
- **Learning From Crowds With Multiple Feature Dynamic Fusion-Based Annotation Generation** — J. Zhang, Z. Zheng, X. Jiang, V. S. Sheng, IEEE TPAMI (2026), DOI 10.1109/TPAMI.2026.3696385. Paywalled; no preprint found.
- **A Survey on Active Learning in Visual Semantic Segmentation: Significance, Challenges, and Prospects** — Z. Xu et al., ACM Computing Surveys (June 2026), DOI 10.1145/3816242. Open access at ACM, but dl.acm.org blocks automated retrieval; no arXiv preprint found.
- **A human-in-the-loop active learning framework for scalable wind energy potential suitability assessment** — Y. Zhang et al., Knowledge-Based Systems (June 2026), DOI 10.1016/j.knosys.2026.115923 (PII S0950705126006490). Listed hybrid-OA, but ScienceDirect blocks automated retrieval; no preprint found.
- **Uncertainty and diversity based selection for active learning in vision-language models (UnDi)** — F. Yang et al., Information Fusion (2026), DOI 10.1016/j.inffus.2026.104260. Paywalled, abstract only; no preprint found.
- **Human-in-the-loop reinforcement learning with risk-aware intervention and imitation** — Y. Zhou et al., Expert Systems with Applications (May 2026), DOI 10.1016/j.eswa.2026.131118. Paywalled; no preprint found.
- **Noise-tolerant scheme and explicit regularizer for deep active learning with noisy oracles** — Y. Li et al., Pattern Recognition (April 2026), DOI 10.1016/j.patcog.2025.112313. Paywalled; no preprint found.
- **Regularized evidential neural networks for deep active learning** — P. Wang et al., Pattern Recognition (May 2026), DOI 10.1016/j.patcog.2025.112836. Paywalled; no preprint found.
- **Confident learning-based noise correction for crowdsourcing** — B. Su, L. Jiang, S. Si, Pattern Recognition (January 2026), DOI 10.1016/j.patcog.2025.111962. Paywalled; no preprint found.
- **Combating Noisy Labels in Object Detection Datasets** — Machine Learning (Springer), March 2026, DOI 10.1007/s10994-025-06976-x. Found late in the search; not retrieved/verified (candidate for follow-up).

### Venue coverage notes

- **IEEE TNNLS:** no on-topic 2026 HITL/active-learning/label-noise paper surfaced in the Crossref sweep (closest matches were control-theoretic "human-in-the-loop" multi-agent control, out of scope).
- **Machine Learning (Springer):** no strong 2026 HITL match beyond the noisy-labels detection paper noted above.
- **Information Fusion / ESWA / Pattern Recognition (Elsevier):** relevant 2026 papers exist (listed above) but all are paywalled with no arXiv preprints, so none could be downloaded.
- **Learning to defer:** no 2026 learning-to-defer paper was found in any of the eleven target journals; the field's consolidating survey ("Learning to Defer: A Survey", Strong et al., December 2025) is a Zenodo preprint under review, not yet in a journal.
