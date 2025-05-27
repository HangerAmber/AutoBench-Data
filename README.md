# AutoControl-Bench

A benchmark for ambiguous vehicle-control commands: parse fuzzy single-turn instructions, ask clarifications, and execute structured function calls.

## 🔍 Overview
- **3 tiers**:  
  1. Single-turn fuzzy parsing  
  2. Extreme ambiguity → clarify  
  3. Multi-turn dialogue → execute  
- **Data**: 20 000 samples (6 k single-turn, 8 k clarification, 6 k dialogues)  
- **Domains**: navigation, HVAC, media, lights, etc.

## ✨ Key Features
- **9 ambiguity types** (e.g. referential vagueness, missing parameters)  
- **Multi-turn context**: maintain history & refine intent  
- **Protocol compliance**: only  
  - `function_call(name, params)`  
  - or a formatted clarification question  
- **Extensible**: add new functions, ambiguity types, models

## 📦 Quick Start

```bash
git clone https://github.com/…/AutoControl-Bench.git
cd AutoControl-Bench
pip install -r requirements.txt
```

```python
from autocontrol_bench import load_dataset
train = load_dataset("data/train.json")
print(len(train))
```

```bash
python evaluation/evaluate.py \
  --pred my_preds.json \
  --truth data/test.json
```

## ⚙️ Metrics
- **IRA**: Intent Recognition Accuracy  
- **PEP**: Parameter Extraction Precision  
- **FDR**: Fuzzy Detection Rate  
- **CQC**: Counter-Question Coverage  
- **DC**: Dialogue Consistency  
- **FESR**: Final Execution Success Rate  

## 📑 Baseline
| Model         | IRA  | PEP  | FDR  | CQC  | DC   | FESR |
|---------------|-----:|-----:|-----:|-----:|-----:|-----:|
| Zero-shot LLM |  70% |  68% |  60% |  55% |  65% |  70% |
| **Fine-tuned**| **85%** | **80%** | **80%** | **75%** | **78%** | **84%** |

## 📜 Citation
```bibtex
@inproceedings{AutoControlBench2025,
  title     = {AutoControl-Bench: A Multi-Agent Knowledge Distillation Framework for Complex Vehicle Function Call Understanding},
  author    = {Anonymous et al.},
  booktitle = {NeurIPS 2025},
  year      = {2025}
}
```
