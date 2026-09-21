# Foundational & Methodological Sources: HITL Deep Active Learning

Downloaded PDFs live in `/home/user1/greenresearch/samim/HITL/papers/`. All arXiv IDs and titles verified against the downloaded PDFs (first-page check) on 2026-08-23.

---

## 1. Learning Loss for Active Learning

- **Authors:** Donggeun Yoo, In So Kweon
- **Venue / Year:** CVPR 2019
- **arXiv:** https://arxiv.org/abs/1905.03677
- **File:** `yoo_2019_learning-loss.pdf`

Yoo and Kweon attach a small auxiliary "loss prediction module" to a target network and train it to predict the target model's loss on unlabeled inputs; samples with the highest predicted loss are sent to the human annotator. The module is task-agnostic — the same mechanism works for classification, regression, and object detection without redesigning the acquisition function. This matters for HITL pipelines because it turns uncertainty estimation into a learned, architecture-level component that scales to modern deep networks, rather than relying on hand-crafted heuristics like softmax entropy. It is a standard baseline in deep active learning comparisons and a template for learned query-selection strategies.

## 2. Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (BADGE)

- **Authors:** Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal
- **Venue / Year:** ICLR 2020
- **arXiv:** https://arxiv.org/abs/1906.03671
- **File:** `ash_2020_badge.pdf`

BADGE selects batches of points whose hallucinated last-layer gradient embeddings (computed with the model's own predicted labels) are both large in magnitude (uncertain) and diverse, using k-means++ seeding in gradient space. This elegantly trades off uncertainty and diversity in a single geometric criterion with no hand-tuned hyperparameter balancing the two. Batch-mode selection is the realistic HITL regime — annotators label in rounds, not one sample at a time — and BADGE is one of the most robust general-purpose batch acquisition strategies across architectures and datasets. It is a near-mandatory baseline for any deep active learning study.

## 3. Active Learning for Convolutional Neural Networks: A Core-Set Approach

- **Authors:** Ozan Sener, Silvio Savarese
- **Venue / Year:** ICLR 2018
- **arXiv:** https://arxiv.org/abs/1708.00489
- **File:** `sener_2018_coreset.pdf`

Sener and Savarese reframe batch active learning as a core-set selection problem: choose the subset of points such that a model trained on it performs comparably to one trained on everything, which they reduce to a k-center facility-location problem in the CNN's feature space. This was the first work to show that classical uncertainty heuristics degrade in the batch setting for CNNs and that pure geometric coverage of the representation space can outperform them. For HITL active learning it establishes the diversity/representativeness side of the acquisition literature, complementary to uncertainty-based sampling, and provides theoretical bounds connecting coverage radius to generalization.

## 4. Deep Bayesian Active Learning with Image Data

- **Authors:** Yarin Gal, Riashat Islam, Zoubin Ghahramani
- **Venue / Year:** ICML 2017
- **arXiv:** https://arxiv.org/abs/1703.02910
- **File:** `gal_2017_bald.pdf`

This paper marries Bayesian uncertainty estimation with deep networks for active learning by using Monte Carlo dropout as an approximate posterior, enabling acquisition functions such as BALD (maximizing mutual information between predictions and model parameters) on image data. It demonstrated large label-efficiency gains on MNIST and a real skin-cancer diagnosis task, showing that principled epistemic uncertainty — not just softmax confidence — should drive query selection. It effectively launched the modern deep (Bayesian) active learning literature and remains the canonical uncertainty-based reference point for HITL annotation budgeting in high-stakes domains.

## 5. Consistent Estimators for Learning to Defer to an Expert

- **Authors:** Hussein Mozannar, David Sontag
- **Venue / Year:** ICML 2020
- **arXiv:** https://arxiv.org/abs/2006.01862
- **File:** `mozannar_2020_defer.pdf`

Mozannar and Sontag formalize the learning-to-defer setting, where a model may either predict or hand an instance to a human expert, and derive the first consistent surrogate loss (a cost-sensitive softmax generalization) for jointly learning the classifier and the rejection/deferral policy. The key insight is that the model should specialize on cases where it outperforms the particular expert it is paired with, adapting to the expert's strengths and error patterns. This is the complementary half of HITL to active learning: instead of humans improving the model via labels, the human is part of the deployed decision pipeline. It anchors the human-AI complementarity and deferral literature that any HITL review must connect to query-based labeling.

## 6. A Survey of Deep Active Learning

- **Authors:** Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B. Gupta, Xiaojiang Chen, Xin Wang
- **Venue / Year:** ACM Computing Surveys, 2021 (arXiv preprint 2020, v2 Dec 2021)
- **arXiv:** https://arxiv.org/abs/2009.00236
- **File:** `ren_2021_dal-survey.pdf`

This survey systematizes the deep active learning (DAL) field, taxonomizing query strategies (uncertainty, diversity/representativeness, hybrid), batch-mode acquisition, and the integration of AL with deep training practices such as data augmentation, semi-supervised learning, and pretrained representations. It also catalogs application domains (vision, NLP, medical imaging) where annotation cost motivates HITL pipelines. For a related-work section it provides the organizing vocabulary and taxonomy against which the individual methods above (BALD, core-set, BADGE, learning-loss) can be positioned, and it identifies open problems — cold start, noisy oracles, evaluation inconsistency — that motivate current HITL research.

## 7. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. (OpenAI)
- **Venue / Year:** NeurIPS 2022
- **arXiv:** https://arxiv.org/abs/2203.02155
- **File:** `ouyang_2022_rlhf.pdf`

InstructGPT operationalizes reinforcement learning from human feedback (RLHF) at scale: human-written demonstrations seed supervised fine-tuning, human preference comparisons train a reward model, and PPO optimizes the language model against that reward. The result — a 1.3B-parameter model preferred by annotators over 175B GPT-3 — showed that carefully targeted human feedback can dominate raw model scale. For a HITL review it is the flagship demonstration that human input beyond point labels (preferences, rankings, demonstrations) can steer deep models, broadening the notion of the "oracle" in active learning and connecting annotation-efficient supervision to modern alignment practice.

## 8. Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels

- **Authors:** Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, Masashi Sugiyama
- **Venue / Year:** NeurIPS 2018
- **arXiv:** https://arxiv.org/abs/1804.06872
- **File:** `han_2018_coteaching.pdf`

Co-teaching trains two networks simultaneously, with each network selecting its small-loss (likely clean) samples in every mini-batch to teach its peer, exploiting the memorization effect that deep nets fit clean patterns before noisy ones. The cross-network exchange prevents the confirmation bias that plagues single-network self-filtering, yielding strong robustness even under extreme (e.g., 45 percent) label corruption. This matters for HITL active learning because real human annotators are noisy oracles: labels bought with the annotation budget are imperfect, and query strategies must be paired with noise-robust training. Co-teaching is the canonical bridge between the AL literature's idealized oracle and realistic crowd-sourced labeling.
