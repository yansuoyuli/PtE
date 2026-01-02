# From ID to LLM: Rethinking Representation Learning for Recommendation

This repository contains the **codebase and datasets** for the paper:

> **From ID to LLM: Rethinking Representation Learning for Recommendation**

---

## 📌 Introduction

Recent studies suggest a fundamental incompatibility between **ID-based representations** and **language model (LM) representations** in recommender systems.  
ID representations primarily encode **collaborative behavioral signals**, whereas LM representations capture **semantic information**.  
As a result, LM-based representations often underperform traditional ID embeddings in recommendation tasks.

In this work, we revisit this problem from an **information-theoretic perspective** and show that **LLM representations theoretically subsume all discriminative information contained in ID embeddings**.  
Based on this observation, we propose **Profile-then-Embedding (PtE)**, a two-stage framework for recommendation:

- **Profile Stage**:  
  Semantic user and item profiles are jointly generated through **LLM-based bidirectional reasoning** over user–item interaction histories.

- **Personalized Embedding Stage**:  
  The generated profiles are encoded into **task-aligned recommendation embeddings**, optimized for downstream recommendation objectives.

Extensive experiments on multiple benchmark datasets demonstrate that PtE consistently improves performance under **cold-start** and **long-tail** settings, across both **discriminative** and **generative** recommendation models.

<img src="./img/motivation.png" width="100%" />

---

## 📝 Environment

### Base Recommendation Models

- `python==3.9.13`
- `numpy==1.23.1`
- `torch==1.11.0`
- `scipy==1.9.1`

### LLM Fine-tuning and Alignment

- `wandb==0.16.2`
- `transformers==4.36.2`
- `trl==0.7.9`
- `peft==0.7.2`

---

## 🚀 How to Run

### 1. LLM Fine-tuning with LoRA

```bash
cd ./llm/lora/
```

#### (a) Supervised Fine-tuning Loop Procedure Repeat until convergence:
We adopt an iterative collaborative fine-tuning strategy. 
Starting from a base LLM, we alternately perform user-side and item-side LoRA-based instruction tuning. 
At each iteration, the model is initialized with the LoRA parameters obtained from the previous stage. This process is repeated until convergence.

```bash
python loop_lora_train.py
```


#### (c) Reinforcement Learning for Personalized Feature Enhancement

```bash
cd ./rlhf/
```

* GRPO Optimization :

```bash
python rl_training.py
```



---

### 2. User / Item Profile Generation

* **User profile generation (knowledge distillation only)**:

```bash
python inference_base.py
```

* **User profile generation (instruction tuning + RL enhancement)**:

```bash
python inference_base_mask.py
```

* **Item profile generation**:

```bash
python inference_base_item.py
```

---

### 3. Running Recommendation Models with Generated Profiles

Example: running **BiasMF** with generated user/item profiles:

```bash
cd ./base_models/BiasMF/
python Main.py --data {dataset}
```

---

## 🎯 Experimental Results

**Performance comparison in terms of *Recall* and *NDCG***:

### Performance Comparison Across Different Backbones and Methods (Cold-Start User Settings)
Note: Boldface indicates the highest score, while underlining denotes the second-best result among the models.

| Backbone | Method       | **Movies**       |          |          | **Toys**         |          |          | **Sports**       |          |          |
|          |              | N@10             | M@20     | M@20     | N@10             | M@20     | M@20     | N@10             | M@20     | M@20     |
|   SASRec | Base         | 0.0358           | 0.0210   | 0.0429   | 0.0263           | 0.0255   | 0.0191   | 0.0221           | 0.0210   | 0.0073   |
|          | MobiRec      | 0.0154           | 0.0105   | 0.0305   | 0.0119           | 0.0114   | 0.0099   | 0.0116           | 0.0072   | 0.0098   |
|          | UniSee       | 0.0220           | 0.0160   | 0.0203   | 0.0179           | 0.0271   | 0.0169   | 0.0411           | 0.0087   | 0.0075   |
|          | WhiRec       | 0.0368           | 0.0116   | 0.0221   | 0.0219           | 0.0268   | 0.0185   | 0.0304           | 0.0194   | 0.0015   |
|          | RL-MobiRec   | 0.0460           | 0.0244   | 0.0443   | 0.0363           | 0.0256   | 0.0151   | 0.0304           | 0.0194   | 0.0085   |
|          | SASRec       | 0.0350           | 0.0224   | 0.0409   | 0.0279           | 0.0203   | 0.0246   | 0.0417           | 0.0227   | 0.0080   |
|          | RL-MLin-Gen  | 0.0755           | 0.0525   | 0.0840   | 0.0291           | 0.0375   | 0.0135   | 0.0343           | 0.0253   | 0.0060   |
|          | I-JEPA       | 0.0359           | 0.0236   | 0.0474   | 0.0255           | 0.0193   | 0.0237   | 0.0256           | 0.0237   | 0.0077   |
|          | P5           | 0.0529           | 0.0444   | 0.0789   | 0.0437           | 0.0392   | 0.0282   | 0.0611           | 0.0425   | 0.0172   |
|          | AlphaEase    | 0.0559           | 0.0236   | 0.0824   | 0.0455           | 0.0419   | 0.0397   | 0.0267           | 0.0297   | 0.0048   |
|          | Best Impr.   | +41.93%          | +54.17%  | +200.18% | +226.94%         | +395.33% | +383.83% | +44.92%          | +0.97%   | +31.75%  |
| DreamRec | Base         | 0.0131           | 0.0083   | 0.0060   | 0.0129           | 0.0183   | 0.0059   | 0.0117           | 0.0097   | 0.0135   |
|  | MobiRec      | 0.0026           | 0.0080   | 0.0062   | 0.0089           | 0.0350   | 0.0051   | 0.0373           | 0.0070   | 0.0112   |
|  | UniSee       | 0.0021           | 0.0014   | 0.0030   | 0.0076           | 0.0099   | 0.0062   | 0.0034           | 0.0020   | 0.0003   |
|  | WhiRec       | 0.0007           | 0.0006   | 0.0080   | 0.0006           | 0.0214   | 0.0081   | 0.0022           | 0.0012   | 0.0090   |
|  | RL-MobiRec   | 0.0016           | 0.0056   | 0.0019   | 0.0054           | 0.0376   | 0.0071   | 0.0184           | 0.0125   | 0.0038   |
|  | RL-MLin-Gen  | 0.0062           | 0.0013   | 0.0113   | 0.0065           | 0.0198   | 0.0329   | 0.0238           | 0.0284   | 0.0168   |
|  | SASRec       | 0.0000           | 0.0017   | 0.0044   | 0.0060           | 0.0177   | 0.0061   | 0.0053           | 0.0060   | 0.0181   |
| | I-JEPA       | 0.0059           | 0.0340   | 0.0084   | 0.0366           | 0.0517   | 0.0461   | 0.0639           | 0.0469   | 0.0245   |
| | P5           | 0.0000           | 0.0017   | 0.0084   | 0.0366           | 0.0517   | 0.0461   | 0.0639           | 0.0469   | 0.0245   |
| | AlphaEase    | 0.0245           | 0.0218   | 0.0224   | 0.0187           | 0.0208   | 0.0224   | 0.0181           | 0.0081   | 0.0181   |
|  | Best Impr.   | +45.95%          | +57.71%  | +37.65%  | +461.41%         | +26.72%  | +32.47%  | +48.48%          | +56.83%  | +27.74%  |



### Performance Comparison Across Different Methods (Long-Tail Settings)

| Dataset | Model       | **Overall**       |          | **Tail Item**     |          | **Head Item**     |          | **Tail User**     |          | **Head User**     |          |
|         |             | R@10              | N@10     | R@10              | N@10     | R@10              | N@10     | R@10              | N@10     | R@10              | N@10     |
| **Yelp**| SASRec      | 0.5940            | 0.3597   | 0.1142            | 0.0495   | 0.7353            | 0.4511   | 0.5893            | 0.3578   | 0.6122            | 0.3672   |
|         | LLM-ESR     | 0.6673            | 0.4208   | 0.1893            | 0.0845   | 0.8048            | 0.5199   | 0.6685            | 0.4239   | 0.6625            | 0.4128   |
|         | AlphaFuse   | 0.6631            | 0.4219   | 0.1815            | 0.0745   | 0.8080            | 0.5232   | 0.6617            | 0.4229   | 0.6587            | 0.4141   |
|         | PJE         | 0.6796            | 0.4263   | 0.2009            | 0.0831   | 0.8173            | 0.5466   | 0.6782            | 0.4296   | 0.6740            | 0.4295   |
|         | Best Impr.  | +1.04%            | +2.40%   | +2.13%            | +10.18%  | +1.85%            | +4.77%   | +1.65%            | +1.34%   | +1.71%            | +3.72%   |
| **Fashion** | SASRec    | 0.4956            | 0.4429   | 0.0454            | 0.0235   | 0.6748            | 0.6099   | 0.3496            | 0.3390   | 0.6129            | 0.5777   |
|         | LLM-ESR     | 0.5609            | 0.4703   | 0.1095            | 0.0560   | 0.7340            | 0.6429   | 0.4311            | 0.3769   | 0.6668            | 0.6005   |
|         | AlphaFuse   | 0.6018            | 0.5143   | 0.2601            | 0.1624   | 0.7426            | 0.6474   | 0.5852            | 0.4276   | 0.6715            | 0.6175   |
|         | PJE         | 0.6462            | 0.5429   | 0.3074            | 0.2457   | 0.7682            | 0.6618   | 0.5738            | 0.4261   | 0.7082            | 0.6493   |
|         | Best Impr.  | +3.88%            | +6.30%   | +18.19%           | +49.29%  | +3.53%            | +2.15%   | +2.59%            | +2.20%   | +8.07%            | +3.24%   |
| **Beauty** | SASRec    | 0.4598            | 0.3390   | 0.0870            | 0.0277   | 0.6174            | 0.4409   | 0.1396            | 0.1390   | 0.6239            | 0.5338   |
|         | LLM-ESR     | 0.5672            | 0.3713   | 0.2257            | 0.1108   | 0.6486            | 0.4334   | 0.5591            | 0.3643   | 0.6058            | 0.4032   |
|         | AlphaFuse   | 0.5793            | 0.4046   | 0.1625            | 0.1006   | 0.6787            | 0.4719   | 0.5692            | 0.3984   | 0.6221            | 0.4326   |
|         | PJE         | 0.6254            | 0.4395   | 0.2634            | 0.1426   | 0.7052            | 0.5127   | 0.6058            | 0.4165   | 0.6625            | 0.4736   |
|         | Best Impr.  | +7.96%            | +8.63%   | +16.70%           | +28.70%  | +3.90%            | +7.50%   | +6.43%            | +4.54%   | +5.80%            | +10.08%  |


---

## 📚 Datasets

### Dataset Statistics

| Statistics       | MIND    | Netflix   | Industrial |
| ---------------- | ------- | --------- | ---------- |
| # Users          | 57,128  | 16,835    | 117,433    |
| # Overlap Items  | 1,020   | 6,232     | 72,417     |
| # Snapshot       | Daily   | Yearly    | Daily      |
| **Training Set** |         |           |            |
| # Items          | 2,386   | 6,532     | 152,069    |
| # Interactions   | 89,734  | 1,655,395 | 858,087    |
| # Sparsity       | 99.934% | 98.495%   | 99.995%    |
| **Test Set**     |         |           |            |
| # Items          | 2,461   | 8,413     | 158,155    |
| # Interactions   | 87,974  | 1,307,051 | 876,415    |
| # Sparsity       | 99.937% | 99.077%   | 99.995%    |

---

## 📎 Notes

* This repository focuses on **representation learning**, rather than prompt-based generation.
* Profile generation and embedding learning are **decoupled by design**, enabling stable integration with both discriminative and generative recommenders.
* Code is organized to facilitate reproducibility and extension.

```


