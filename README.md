# AutoControl-Bench 🚗🧠

**AutoControl-Bench: A Three-Tier Benchmark for Ambiguity-Rich Function Call Evaluation in Vehicle Language Understanding**

> Official benchmark release for our NeurIPS 2025 paper:
> **"AutoControl-Bench: A Multi-Agent Benchmark for Protocol-Compliant Vehicle Function Call Comprehension"**

---

## 🔍 Overview

AutoControl-Bench is a function-call benchmark designed to evaluate large language models (LLMs) in realistic, high-ambiguity vehicle control scenarios. It introduces:

- 🧩 **9 Types of Linguistic Ambiguity**  
- 🔄 **Multi-turn Dialogue Understanding**  
- ✅ **Protocol-compliant Function Call Constraints**  
- 🤖 **Multi-Agent Benchmark Construction Pipeline**

---

## 📊 Key Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| Ambiguity Coverage               | 9 types (e.g., vague references, underspecification, ellipsis, etc.)       |
| Evaluation Protocol              | Three-tier: fuzzy parsing, counter-questioning, multi-turn consistency      |
| Benchmark Size                   | 20,000 examples (6k Tier-1, 8k Tier-2, 6k Tier-3)                        |
| Scenarios                        | Safety-critical, Entertainment, Autonomous Driving, Comfort                 |
| Output Format                    | JSON-based instruction–response pairs with function call targets            |
| Agents Involved                  | Semantic Parsing, Adversarial Gen, Fuzz Injection, Multi-turn Simulation    |

---

## 📁 Repository Structure

```bash
AutoControl-Bench/
├── data/
│   ├── tier1_single_turn.json
│   ├── tier2_fuzzy_clarify.json
│   ├── tier3_multi_turn.json
│   └── protocol/
├── scripts/
│   └── create_datasets.py
├── requirements.txt
├── croissant_metadata.json
└── README.md
⚙️ Quick Start
1. Install dependencies
bash
复制
编辑
pip install -r requirements.txt
2. Download data
All data is in the data/ folder. To load the Tier-3 multi-turn benchmark:

python
复制
编辑
import json
with open('data/tier3_multi_turn.json') as f:
    samples = json.load(f)
3. Run Evaluation
bash
复制
编辑
python scripts/eval_metrics.py --input <predictions.json> --ref data/tier3_multi_turn.json
📌 Evaluation Metrics
Metric	Description
IRA	Intent Recognition Accuracy
PEP	Parameter Extraction Precision
FDR	Fuzzy Detection Rate
CQC	Counter-Question Coverage
DC	Dialogue Consistency (multi-turn)
FESR	Final Execution Success Rate

All metrics are implemented in scripts/eval_metrics.py.

📄 Dataset Format (JSON)
Each sample in the benchmark includes:

json
复制
编辑
{
  "id": "multi_001",
  "tier": "Tier-3",
  "dialogue": [
    {"user": "Turn the lights on.", "assistant": "Which lights? Headlights or interior?"},
    {"user": "Headlights, please."}
  ],
  "target_call": {
    "function": "control_lighting",
    "parameters": {"zone": "headlights", "state": "on"}
  },
  "meta": {
    "ambiguity_type": "underspecification",
    "protocol_compliant": true
  }
}
🧪 Baseline Results
Model	DQS ↑	FESR ↑	CQC ↑	DC ↑
GPT-4	0.69	90.0	85.2	87.0
Claude 3	0.70	91.5	84.1	86.5
Qwen2.5-7B-SFT 🥇	0.88	97.5	99.2	94.8

📜 License
This project is released under the Apache 2.0 License.

📬 Citation
If you use AutoControl-Bench in your research, please cite:

bibtex
复制
编辑
@inproceedings{autocontrolbench2025,
  title     = {AutoControl-Bench: A Multi-Agent Benchmark for Protocol-Compliant Vehicle Function Call Comprehension},
  author    = {Anonymous et al.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
