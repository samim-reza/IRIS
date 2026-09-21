# Nature Portfolio Sources — Human-in-the-Loop Machine Learning (2026 / late 2025)

Compiled 2026-08-23. All papers below were verified by fetching their nature.com article pages, and every listed PDF was downloaded and checked (`%PDF` header, size > 100 KB, first-page text matches the title). PDFs are in `/home/user1/greenresearch/samim/HITL/papers/`.

---

## 1. A human-in-the-loop explanation framework for morphologically transparent AI predictions from whole-slide images

- **Authors:** Peiliang Lou, Yitan Zhu, Nicholas Chia, Roopa Kumari, William Yang, Yan Wang, Brenna C. Novotny, Stacey J. Winham, Ruifeng Guo, Ellen L. Goode, et al.
- **Journal:** npj Digital Medicine
- **Year:** 2026 (published 14 May 2026)
- **Volume / Article:** Volume 9, Article 579
- **DOI:** 10.1038/s41746-026-02741-z
- **URL:** https://www.nature.com/articles/s41746-026-02741-z
- **Downloaded file:** `lou_2026_morphoxai-hitl-pathology.pdf` (publisher PDF, open access)

**Summary:** The paper introduces MorphoXAI, a human-in-the-loop explanation framework for deep learning models that classify whole-slide histopathology images. The method identifies high-contribution image regions across multiple training runs, clusters them by visual characteristics, and has pathologists interpret the clusters to define a "morphologic spectrum" capturing both class-defining and transitional tissue patterns. This spectrum supports both global (model-level pattern discovery) and local (slide-level prediction justification) explanations. The framework was validated on ovarian, breast, and endometrial cancer datasets. MorphoXAI explanations accurately reflected tumor histomorphology and significantly improved pathologists' ability to understand and verify model predictions compared with attention heatmaps alone. The work is a direct example of expert-in-the-loop interpretation being folded back into model transparency for clinical AI.

---

## 2. Improving wildlife track classification through human-in-the-loop method and explainable AI

- **Authors:** Tinao Petso, Rodrigo S. Jamisola Jr., Sky Alibhai, et al.
- **Journal:** Scientific Reports
- **Year:** 2026 (published 20 April 2026)
- **Volume / Article:** Volume 16, Article 18395
- **DOI:** 10.1038/s41598-026-48229-4
- **URL:** https://www.nature.com/articles/s41598-026-48229-4
- **Downloaded file:** `petso_2026_hitl-wildlife-tracks.pdf` (full-text PDF via Europe PMC, PMC13265934; open access CC BY-NC-ND)

**Summary:** This study integrates expert animal-tracker knowledge into an AI pipeline for classifying wildlife species from footprint (track) images in Botswana, covering black rhinoceros, white rhinoceros, blue wildebeest, and giraffe. Three assessors of varying expertise ranked 3,039 track images by quality, and models were trained on expert-selected versus non-expert-selected data. Expert tracker input increased mean average precision@50–95 by 10.42% versus non-expert assessment, and the model needed 25% fewer training images when expert-curated data was used. In a second human-in-the-loop stage, 36 expert trackers evaluated model outputs with raw images alone or with explanatory heatmaps: heatmap support raised human species-classification accuracy to 98.67% versus 92.39% with raw images. The paper demonstrates a two-way HITL loop — human expertise improving data selection for training, and XAI improving human decision quality.

---

## 3. Computational framework to predict and shape human–machine interactions in closed-loop, co-adaptive neural interfaces

- **Authors:** Maneeshika M. Madduri, Momona Yamagami, Si Jia Li, Sasha Burckhardt, Samuel A. Burden, Amy L. Orsborn
- **Journal:** Nature Machine Intelligence
- **Year:** 2026 (published 23 March 2026)
- **Volume / Pages:** Volume 8, pages 372–387
- **DOI:** 10.1038/s42256-026-01194-z
- **URL:** https://www.nature.com/articles/s42256-026-01194-z
- **Preprint:** bioRxiv, DOI 10.1101/2024.05.23.595598
- **Downloaded file:** `madduri_2026_coadaptive-interfaces.pdf` (publisher PDF; this article is open access CC BY-NC-ND despite Nature Machine Intelligence being a hybrid journal)

**Summary:** The authors develop a computational framework, grounded in control theory and game theory, to model closed-loop interaction between a human user and an adaptive decoder in co-adaptive neural interfaces — a setting where both the human and the machine learner update simultaneously. They validate the framework with a myoelectric (muscle-based) interface experiment in which 14 participants learned to track targets while the decoder adapted online. The models predicted how decoder learning rates influence convergence of the joint human–machine system and how decoder penalty (regularization) parameters reshape user effort without degrading task performance. A key result is that interface algorithm properties can deliberately shape user behaviour, giving designers principled levers over the human side of the loop. This is a rigorous treatment of mutual human-machine adaptation, relevant to HITL learning dynamics beyond neural interfaces.

---

## 4. Human-AI co-design for clinical prediction models

- **Authors:** Jean Feng, Avni Kothari, Patrick Vossler, et al.
- **Journal:** npj Digital Medicine
- **Year:** 2026 (published online 06 June 2026; article in press — volume/article number not yet assigned at download time)
- **DOI:** 10.1038/s41746-026-02838-5
- **URL:** https://www.nature.com/articles/s41746-026-02838-5
- **Downloaded file:** `feng_2026_hachi-codesign.pdf` (publisher "article in press" PDF, open access CC BY 4.0)

**Summary:** This paper presents HACHI, an iterative human-in-the-loop framework in which AI agents accelerate the development of fully interpretable clinical prediction models (CPMs) from unstructured clinical notes. The system alternates between AI-driven statistical exploration (concept extraction and model fitting) and expert feedback cycles, optimizing for transparency, steerability, and reciprocal learning between the AI agent and domain experts. It is evaluated on acute kidney injury and traumatic brain injury prediction tasks. HACHI outperformed existing approaches, surfaced clinically meaningful concepts, and improved model generalization across medical sites and time periods. The authors show that human oversight is critical for directing the AI's exploration, refining concept definitions, and catching data bias or label-leakage issues — a concrete demonstration of expert feedback improving model training and validity.

---

## 5. Human–large language model collaboration in clinical medicine: a systematic review and meta-analysis

- **Authors:** Guoyong Wang, Kaijun Zhang, Jiyue Jiang, Chuan Wang, et al.
- **Journal:** npj Digital Medicine
- **Year:** 2026 (published 28 January 2026)
- **Volume / Article:** Volume 9, Article 195
- **DOI:** 10.1038/s41746-026-02382-2
- **URL:** https://www.nature.com/articles/s41746-026-02382-2
- **Downloaded file:** `wang_2026_human-llm-collab.pdf` (full-text PDF via Europe PMC, PMC12953916; open access)

**Summary:** A PRISMA 2020 systematic review and meta-analysis of human-AI collaboration (H+AI) versus human-only workflows in clinical medicine, covering 10 peer-reviewed studies retrieved from four databases through 28 June 2025. Tasks span diagnostic reasoning, triage, radiologic interpretation, documentation, and cross-disciplinary communication with LLM-based systems. Composite diagnostic/management scores improved modestly under collaboration (+4.88 percentage points, 95% CI 0.65–9.12), while diagnostic accuracy showed a positive but highly uncertain trend (RR 1.59, 95% CI 0.08–32.74) and time efficiency showed no overall difference. Factual error rates in collaborative outputs remained high (26–36%), and H+AI did not universally outperform AI-only systems. The authors conclude that benefits of human-AI teaming are preliminary, context-dependent, and in need of preregistered pragmatic trials with stronger safety metrics. Useful as the current evidence-synthesis baseline for human-AI teaming claims.

---

## 6. A benchmarking framework and dataset for learning to defer in human-AI decision-making

- **Authors:** Jean V. Alves, Diogo Leitão, Sérgio Jesus, et al.
- **Journal:** Scientific Data
- **Year:** 2025 (published 23 April 2025) — included despite being 2025 because it is the Nature-portfolio reference benchmark for learning-to-defer, a core topic of this review
- **Volume / Article:** Volume 12, Article 506
- **DOI:** 10.1038/s41597-025-04664-y
- **URL:** https://www.nature.com/articles/s41597-025-04664-y
- **Downloaded file:** `alves_2025_learning-to-defer-benchmark.pdf` (full-text PDF via Europe PMC, PMC12019285; open access)

**Summary:** This Data Descriptor introduces OpenL2D, a customizable framework for generating synthetic experts with realistic decision-making processes and work-capacity constraints, aimed at benchmarking learning-to-defer (L2D) algorithms — systems that decide whether a prediction should be made by the ML model or deferred to a human expert. Applied to a public bank-account-opening fraud detection dataset, OpenL2D produces FiFAR (Financial Fraud Alert Review), containing predictions from 50 simulated fraud analysts on 30K instances. The synthetic experts exhibit properties comparable to real professionals, including intra-expert consistency and inter-expert agreement. Benchmarking shows that the performance ranking of L2D algorithms varies substantially depending on which experts are available, demonstrating that realistic, diverse expert behavior and capacity constraints must be modeled when evaluating deferral systems. This addresses the field's chronic lack of public datasets with real or realistic human-expert predictions.
