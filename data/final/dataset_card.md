# Safety Dataset Card

## 概述
TikTok 内容安全多模态数据集，用于训练和评估内容审核模型。

## 运行模式
- 模式: smoke_test
- 设备: mps

## 数据文件
| 文件 | 样本数 | 用途 |
|------|--------|------|
| safety_sft_mix.jsonl | 6,772 | SFT 训练 |
| safety_dpo_pairs.jsonl | 4,225 | DPO 训练 |
| safety_eval.jsonl | 1,305 | 模型评估 |

## 数据来源
- WildGuardMix: 安全对话数据
- WildJailbreak: 越狱攻击数据
- ToxiGen: 隐式毒性数据
- SafeBench: 安全基准测试
- XSTest: Over-refusal 测试
- HarmBench: 有害内容基准
- LLaVA-Instruct: 视觉语言数据
- MM-SafetyBench: 多模态安全基准

## 风险类别（14 类）
01. 非法活动 (illegal_activity)
02. 仇恨言论 (hate_speech)
03. 恶意软件 (malware)
04. 人身伤害 (physical_harm)
05. 经济犯罪 (economic_harm)
06. 欺诈 (fraud)
07. 色情内容 (pornography)
08. 政治操纵 (political_lobbying)
09. 隐私侵犯 (privacy_violation)
10. 法律意见 (legal_opinion)
11. 金融建议 (financial_advice)
12. 医疗咨询 (health_consultation)
13. 政府决策 (gov_decision)
14. 版权侵犯 (copyright) ← TikTok 独有

## 数据格式
```json
{
    "text": "User: ...",
    "images": [],
    "meta": {
        "source": "数据来源",
        "risk_category": "风险类别",
        "severity": "严重程度",
        "attack_type": "攻击类型",
        "prompt_harm_label": "harmful/unharmful",
        "synthetic": false
    }
}
```

## 许可证
仅供研究用途。各子数据集遵循其原始许可证。
