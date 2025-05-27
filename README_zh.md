# AutoControl-Bench 🚗🧠

一个针对模糊车辆控制命令的基准测试：解析模糊的单轮指令，提出澄清问题，并执行结构化的函数调用。

## 🔍 概述
- **3个层级**：  
  1. 单轮模糊解析  
  2. 极端模糊性 → 澄清  
  3. 多轮对话 → 执行  
- **数据**：20,000个样本（6,000个单轮，8,000个澄清，6,000个对话）  
- **领域**：导航、暖通空调、媒体、灯光等

## ✨ 关键特性
- **9种模糊类型**（例如指代模糊、缺失参数）  
- **多轮对话上下文**：维护历史记录并细化意图  
- **协议遵守**：仅限  
  - function_call(name, params)  
  - 格式化的澄清问题
- **可扩展**：添加新功能、模糊类型、模型

## 📁 仓库结构

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
├── README_zh.md
└── README.md
```

## 📄 数据集格式（JSON）
基准测试中的每个样本包括：
```json
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
```

## 📦 快速入门

```bash
git clone https://github.com/HangerAmber/AutoBench-Data.git
cd AutoBench-Data
pip install -r requirements.txt
```

```bash
python scripts/create_datasets.py
```

## ⚙️ 评估指标
- **IRA**：意图识别准确率  
- **PEP**：参数提取精度  
- **FDR**：模糊检测率  
- **CQC**：反问覆盖率  
- **DC**：对话一致性  
- **FESR**：最终执行成功率  

## 📑 基线
| 模型          | IRA  | PEP  | FDR  | CQC  | DC   | FESR |
|---------------|-----:|-----:|-----:|-----:|-----:|-----:|
| zero-shot LLM   |  70% |  68% |  60% |  55% |  65% |  70% |
| **SFT**      | **85%** | **80%** | **80%** | **75%** | **78%** | **84%** |

## 📜 引用
```bibtex
@inproceedings{AutoControlBench2025,
  title     = {AutoControl-Bench: A Multi-Agent Knowledge Distillation Framework for Complex Vehicle Function Call Understanding},
  author    = {Anonymous et al.},
  booktitle = {NeurIPS 2025},
  year      = {2025}
}
```
