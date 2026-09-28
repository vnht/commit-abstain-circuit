# The Commit-Abstain Circuit: Why Language Models Hallucinate Instead of Abstaining

## Overview

This repository contains the source code accompanying the paper. We identify a subset of attention heads and MLP sublayers (the Commit-Abstain Circuit, CAC) that mediate whether a language model commits to an answer or abstains. The code covers the full pipeline: computing commit-abstain scores, localising the CAC, and validating it via ablation and amplification experiments.

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Hugging Face token to `.env`:

```
HF_TOKEN=hf_...
```

---

## Pipeline

The three modules run in order. Each writes results to its own `results/` subdirectory, which the next module reads from.

### 1. Delta-Eval

Runs inference on a model and dataset pair and computes the commit-abstain score Δ(x) = s_commit - s_abstain from first-step logits.

```bash
python delta-eval/run.py --model gemma-4b --dataset kuq
python delta-eval/run.py --model gemma-4b --dataset musique
python delta-eval/run.py --model gemma-4b --dataset squad
```

Supported models: `gemma-4b`, `gemma-12b`, `gemma-27b`, `ministral-3b`, `ministral-14b`, `qwen-4b`, `qwen-9b`, `qwen-35b`, `phi-4-mini`, `phi-4`, `llama-3b`, `llama-8b`

Supported datasets: `kuq`, `musique`, `squad`, `hotpotqa`, `selfaware`

Results are written to `delta-eval/results/{model}_{dataset}.json` and `.jsonl`.

---

### 2. CAC Localisation

Fits differentiable soft gates over all attention heads and MLP sublayers using a three-phase protocol (lambda=0 initialisation, lambda>0 retention, lambda<0 removal). Run across multiple seeds on the joint dataset (KUQ + MuSiQue + SQuAD), then classify each component as abstention-promoting (A), commitment-promoting (C), or irrelevant (I).

```bash
# Optional: pre-screen components
python cac-localisation/prescreen.py --model gemma-4b --dataset joint

# Fit gates (default: 10 seeds)
python cac-localisation/run.py --model gemma-4b --dataset joint

# Classify components from seed results
python cac-localisation/identify_cac.py --model gemma-4b
```

Results are written to `cac-localisation/results/{model}_joint_seed{n}.json` and `{model}_joint_summary.json`.

---

### 3. Functional Analysis

Validates the CAC via two complementary interventions. Both scripts require `cac-localisation/results/{model}_joint_summary.json`.

**Amplification** sweeps alpha in {1.0, 1.5, 2.0, 2.5, 3.0, 10, 100, 1000} across two symmetric experiments:

- **FC on A**: scales A-component outputs on false-commitment (FC) instances; measures conversion to abstention
- **FA on C**: scales C-component outputs on false-abstention (FA) instances; measures conversion to commitment

```bash
python functional-analysis/amplification.py --model gemma-4b
```

**Ablation** zero-ablates A and C components on held-out data and measures mean delta shift and sign-flip rates (commit to abstain, abstain to commit):

```bash
python functional-analysis/ablation.py --model gemma-4b
```

---

## Models

| Shortcut | Model | Family | Parameters |
|---|---|---|---|
| `gemma-4b` | google/gemma-3-4b-it | Gemma 3 | 4B |
| `gemma-12b` | google/gemma-3-12b-it | Gemma 3 | 12B |
| `gemma-27b` | google/gemma-3-27b-it | Gemma 3 | 27B |
| `ministral-3b` | mistralai/Ministral-3-3B-Instruct-2512 | Ministral 3 | 3B |
| `ministral-14b` | mistralai/Ministral-3-14B-Instruct-2512 | Ministral 3 | 14B |
| `qwen-4b` | Qwen/Qwen3.5-4B | Qwen 3.5 | 4B |
| `qwen-9b` | Qwen/Qwen3.5-9B | Qwen 3.5 | 9B |
| `qwen-35b` | Qwen/Qwen3.5-35B-A3B | Qwen 3.5 | 35B |
| `phi-4-mini` | microsoft/Phi-4-mini-instruct | Phi-4 | 3.8B |
| `phi-4` | microsoft/phi-4 | Phi-4 | 14B |
| `llama-3b` | meta-llama/Llama-3.2-3B-Instruct | Llama 3 | 3B |
| `llama-8b` | meta-llama/Llama-3.1-8B-Instruct | Llama 3 | 8B |

---

## Example unanswerable questions

**KUQ** (closed-book, knowledge-bounded)

```
Question: Who was the first person to have a backyard patio?
Label:    unanswerable
```

**MuSiQue** (open-book, multi-hop)

```
Context:  Lemonade is an alternative dance band from San Francisco, California, now
          residing in Brooklyn, New York. The band was formed in 2005 [...] London's
          first and only cable car, known as the Emirates Air Line, opened in June 2012.
          Crossing the River Thames, linking Greenwich Peninsula and the Royal Docks [...]

Question: When did cable cars come into use in the city where Lemonade formed?
Label:    unanswerable
```

**SQuAD** (open-book, reading comprehension)

```
Context:  There are 13 natural reserves in Warsaw. About 15 kilometres from Warsaw,
          the Vistula river's environment changes strikingly and features a perfectly
          preserved ecosystem, with a habitat of animals that includes the otter,
          beaver and hundreds of bird species [...]

Question: What animals does the Czerniakow river's ecosystem include?
Label:    unanswerable
```

**HotpotQA** (open-book, multi-hop)

```
Context:  PECOTA is a sabermetric system for forecasting Major League Baseball player
          performance [...] Robert Allen Parnell (born September 8, 1984) is an American
          baseball pitcher [...] drafted by the Mets in the ninth round of the 2005
          Major League Baseball draft [...]

Question: What Cuban American baseball player was suspended in the new steroid policy
          of the 2005 Major League Baseball season?
Label:    unanswerable
```

**SelfAware** (closed-book, philosophical)

```
Question: Who's buried in Grant's Tomb under the name Ulysses S. Grant if he wasn't
          really called Ulysses S. Grant?
Label:    unanswerable
```

---

## Repository structure

```
source-code/
├── core/
│   ├── config.py              # Model shortcuts, dataset configs, abstention phrases
│   ├── data.py                # Dataset loading and prompt construction
│   ├── inference.py           # Model loading and inference (all model families)
│   ├── scoring.py             # Δ(x) = s_commit - s_abstain from logits
│   ├── metrics.py             # AUROC, decision metrics (accuracy, TC, FC, TA, FA)
│   ├── gating.py              # CausalComponentGating (differentiable soft gates)
│   └── parsing.py             # Response parsing utilities
│
├── datasets/
│   ├── kuq.json
│   ├── musique.json
│   ├── squad.json
│   ├── hotpotqa.json
│   ├── selfaware.json
│   └── sample.json            # 10-instance sample
│
├── llms/
│   ├── base.py
│   ├── gemma.py
│   ├── llama.py
│   ├── mistral.py
│   ├── phi.py
│   └── qwen.py
│
├── delta-eval/
│   └── run.py
│
├── cac-localisation/
│   ├── prescreen.py
│   ├── run.py
│   └── identify_cac.py
│
├── functional-analysis/
│   ├── ablation.py
│   └── amplification.py
│
├── requirements.txt
├── .env
└── README.md
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
