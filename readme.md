

# 🧠  LLM Representation Is All You Need for Recommendation

> **Official PyTorch Implementation**

---

## 🌐 Overview

Recent advances in **Large Language Models (LLMs)** have demonstrated strong semantic reasoning and generalization ability.
However, in recommender systems, **LLM representations consistently underperform traditional ID embeddings**, especially on collaborative filtering tasks—leading to a widely held belief that *LLMs cannot replace ID-based representations*.

In this work, we revisit this assumption from an **information-theoretic perspective** and show that:

> 🔍 **LLM representations are theoretically sufficient to subsume all collaborative signals encoded in ID embeddings.**

We further demonstrate that the empirical underperformance of LLM embeddings stems not from information loss, but from **semantic entanglement and misalignment with recommendation objectives**.

To bridge this gap, we propose **Profile-then-Embedding (PtE)** — a principled, two-stage recommendation framework that decouples **semantic reasoning** from **embedding learning**.

---

<div align="center">
  <img src="img/pipeline.png" alt="PtE Framework" width="85%">
</div>

---

## 🔑 Key Idea

PtE reframes LLM-based recommendation as a **generate-then-embed** process:

### 🧩 1. Profile Stage — Semantic Reasoning over Interactions

* 🧠 **Bidirectional LLM Reasoning** jointly generates **user and item semantic profiles**
* 🔁 User profiles are inferred from interaction histories and collaborative neighbors
* 🔁 Item profiles are induced from the semantic characteristics of their interacting users
* ♻️ Profiles are iteratively refined in a **closed-loop**, enabling mutual semantic enhancement

### 🎯 2. Personalized Embedding Stage — Task-Aligned Representation Learning

* 🧬 Multiple candidate profiles are sampled per user/item
* ⚖️ **Group Relative Policy Optimization (GRPO)** selects profiles that:

  * Align with collaborative signals
  * Preserve personalization and distinctiveness
* 📐 Final embeddings are extracted from optimized profiles for downstream recommendation

> ✨ By explicitly surfacing collaborative signals *before* embedding, PtE reconciles the theoretical sufficiency of LLMs with practical recommendation performance.

---

## 🧠 Theoretical Foundations

PtE is grounded in **statistical decision theory**.

We show that:

* **LLM embeddings Blackwell-dominate ID embeddings**
* Any fusion of ID + LLM embeddings cannot contain more predictive information than LLM embeddings alone
* Empirical fusion gains arise from *optimization convenience*, not information superiority

<div align="center">
  <img src="img/case.png" alt="Information Dominance Case Study" width="70%">
</div>

📌 **Takeaway**

> *The problem is not whether LLM embeddings are sufficient — but how to disentangle and exploit the collaborative signals they already contain.*

---

## ⚙️ Requirements

### Dependencies

```bash
Python >= 3.8
PyTorch >= 1.13
transformers
einops
numpy
scikit-learn
```

### Optional

```bash
accelerate
deepspeed
```

---

## 📦 Datasets

We evaluate PtE on widely used **Amazon recommendation benchmarks**:

| Dataset | Domain            |
| ------- | ----------------- |
| Movies  | Movies & TV       |
| Toys    | Toys & Games      |
| Sports  | Sports & Outdoors |
| Yelp    | Local Business    |

### Settings

* **Cold-start user evaluation**
* **Leave-one-out evaluation**
* **Long-tail user & item splits**

Dataset preprocessing follows prior work (LLM-ESR, DreamRec).

---

## 🚀 Training & Evaluation

### 🔹 Profile Generation

```bash
python profile/generate_user_profiles.py
python profile/generate_item_profiles.py
```

Supports:

* Prompt perturbation
* Temperature sampling
* Iterative bidirectional refinement

---

### 🔹 Embedding Optimization (GRPO)

```bash
python train/grpo_embedding.py --dataset Movies --backbone SASRec
```

Backbones:

* **SASRec** (discriminative)
* **DreamRec** (generative diffusion-based)

---

### 🔹 Evaluation

```bash
python eval/evaluate.py --setting cold_start
python eval/evaluate.py --setting leave_one_out
```

Metrics:

* NDCG@K
* Recall@K
* Hit@K

---

## 🧪 Experimental Results

### 🧊 Cold-Start Performance

* PtE consistently outperforms:

  * LLMInit
  * RLMRec
  * AlphaFuse
  * LLM-ESR
* Gains are most pronounced under **severe sparsity**

### 🌱 Long-Tail Robustness

* Up to **49% relative improvement** on tail users/items
* Stable across both discriminative and generative backbones

> 📈 PtE remains effective where direct LLM embedding and fusion-based methods collapse.

---

## 🔍 Ablation Highlights

| Variant                 | Observation            |
| ----------------------- | ---------------------- |
| No Profile              | Severe degradation     |
| User-only / Item-only   | Partial gains          |
| Single-pass profiling   | Unstable semantics     |
| No GRPO                 | Profile collapse       |
| No collaborative reward | Tail performance drops |

➡️ **Joint profiling + relative optimization is critical**

---

## 📌 Contributions Summary

* 🧠 **Theory**: Establishes information dominance of LLM embeddings over ID embeddings
* 🔄 **Method**: Introduces Profile-then-Embedding to disentangle semantic and collaborative signals
* ❄️ **Robustness**: Strong cold-start and long-tail generalization
* 🔌 **Generality**: Works across discriminative and generative recommenders

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{pte2025,
  title={LLM Representation Is All You Need for Recommendation},
  author={Anonymous},
  journal={ACL},
  year={2025}
}
```

---

## ⚠️ Limitations & Future Work

* Profile generation incurs non-trivial LLM cost
* Future work:

  * Profile distillation
  * Lightweight domain-specific LLMs
  * Formal analysis of semantic disentanglement dynamics
