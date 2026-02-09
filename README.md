# Mistral-7B LoRA Fine-Tuning for Legal QA

Fine-tuned **mistralai/Mistral-7B-Instruct-v0.2** using **LoRA (PEFT)** with **4-bit quantization (bitsandbytes)** for English Legal Question Answering.

## Overview
This project demonstrates an efficient LLM fine-tuning pipeline:
- Load Mistral-7B-Instruct in **4-bit (NF4)** to reduce VRAM usage
- Apply **LoRA adapters** (PEFT) targeting 4-bit linear layers (excluding `lm_head`)
- Supervised fine-tuning using **TRL SFTTrainer**
- Evaluate using **F1**, **ROUGE-L**, and **BERTScore**

## Base Model
- `mistralai/Mistral-7B-Instruct-v0.2` :contentReference[oaicite:1]{index=1}

## Dataset
- `haistudy/en_law_qa` (English legal QA)
- Reformatted into `[INST] ... [/INST]` style prompts
- Split: 90% train / 10% eval :contentReference[oaicite:2]{index=2}

## Training Setup
- LoRA: `r=16`, `alpha=32`, `dropout=0.05` :contentReference[oaicite:3]{index=3}
- Trainer: `trl.SFTTrainer` :contentReference[oaicite:4]{index=4}
- Batch size: 4 (gradient accumulation = 2)
- Learning rate: 2e-4
- Epochs: 1
- FP16 enabled :contentReference[oaicite:5]{index=5}

## Evaluation Results
- F1: **0.6408**
- ROUGE-L: **0.6966**
- BERTScore (F1): **0.9337** :contentReference[oaicite:6]{index=6}

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Hugging Face Login (IMPORTANT)

**Do NOT hardcode your token in code.**

Use an environment variable instead:

```
export HF_TOKEN="YOUR_TOKEN"
```

Then in Python:

```
from huggingface_hub import login
import os
login(token=os.getenv("HF_TOKEN"))
```

## 3) Train :
Run the notebook/script to:

load model in 4-bit

attach LoRA adapters

fine-tune with SFTTrainer


## 4) Evaluate : 
The evaluation computes:

token-overlap F1 (simple baseline)

ROUGE-L

BERTScore

## Author :
**Mohammad Ali Othman**
Data Science Student — Jordan University of Science and Technology
