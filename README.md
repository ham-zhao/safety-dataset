# 多模态内容安全审核系统

> 面向 TikTok 审核场景的多模态内容安全数据构建、模型训练与量化评估完整闭环

## 项目定位

本项目处于 AI 训练链路的 **Post-train 数据准备阶段**：

```
Pre-train 数据清洗 → Pre-train 训练 → [Post-train 数据准备 ← 本项目] → SFT/DPO 训练 → Chat Model
```

核心任务：构建高质量的安全对齐训练数据（SFT + DPO），训练轻量级安全分类器，并通过标准 Benchmark 量化评估 + 消融实验验证每个环节的价值。

## 完整闭环

```
数据构建（Data-Juicer 清洗+增强）
  ↓
模型训练（DistilBERT 文本分类 + CLIP 多模态分类）
  ↓
量化评估（WildGuardTest AUC / HarmBench Recall / XSTest Over-refusal）
  ↓
消融实验（验证每个数据组件的贡献）
  ↓
反馈优化（发现数据短板 → 回到数据构建）
```

## 14 类安全风险体系

| 编号 | 风险类别 | 编号 | 风险类别 |
|------|---------|------|---------|
| 01 | 非法活动 | 08 | 政治游说 |
| 02 | 仇恨言论 | 09 | 隐私侵犯 |
| 03 | 恶意软件 | 10 | 法律建议 |
| 04 | 身体伤害 | 11 | 财务建议 |
| 05 | 经济犯罪 | 12 | 健康咨询 |
| 06 | 欺诈行为 | 13 | 政府决策 |
| 07 | 色情内容 | **14** | **版权/IP侵权** |

> 第 14 类（版权/IP侵权）是针对 TikTok 视频平台场景的独立扩展，使用 CLIP embedding 余弦相似度匹配实现。

## TikTok 多模型级联架构

```
用户上传内容
  ↓
第一层：轻量分类器（DistilBERT，毫秒级）← 本项目训练
  ↓
第二层：中等模型（7B 级别，百毫秒级）
  ↓
第三层：大模型 / 人工复核（秒级）
```

## 核心产出

| 产出 | 文件 | 规模 |
|------|------|------|
| 安全 SFT 数据集 | `data/final/safety_sft_mix.jsonl` | 6,772 条 |
| 安全 DPO 偏好对 | `data/final/safety_dpo_pairs.jsonl` | 4,225 对 |
| 评估数据集 | `data/final/safety_eval.jsonl` | 1,305 条 |
| 数据集卡片 | `data/final/dataset_card.md` | — |
| 文本分类器 | `results/models/text_classifier/` | DistilBERT 微调 |
| 多模态分类器 | `results/models/multimodal_classifier/` | CLIP Head |
| 消融模型 (×6) | `results/models/text_classifier_ablation_*/` | 6 组对照实验 |
| 可视化图表 | `results/figures/` | 23 张 PNG |
| Dashboard | `results/final_summary.json` | 全项目汇总 |

## 数据集来源（8 个公开数据集）

| 数据集 | 规模 | 用途 |
|--------|------|------|
| WildGuardMix | 92K | 核心安全训练+测试 |
| WildJailbreak | 262K | 对比构造，防 over-refusal |
| MM-SafetyBench | 5,040 | 多模态安全 benchmark |
| ToxiGen | 274K | 隐式仇恨言论 |
| SafeBench | 2,300 | 辅助安全对 |
| XSTest | 250 | Over-refusal 测试 |
| HarmBench | ~500 | 标准有害 benchmark |
| LLaVA-150K | 5K 子集 | 正常数据对照 |

## 快速开始

```bash
# 1. 环境搭建
bash setup.sh
source venv/bin/activate

# 2. 下载数据 + 格式转换
python3 scripts/run_download.py

# 3. 数据清洗
python3 scripts/run_cleaning.py

# 4. 数据增强
python3 scripts/run_augmentation.py

# 5. 模型训练（DistilBERT + CLIP Head）
python3 scripts/run_training.py

# 6. Benchmark 评估
python3 scripts/run_evaluation.py

# 7. 消融实验（6 组）
python3 scripts/run_ablation.py

# 8. 生成最终输出文件
python3 scripts/generate_report.py
```

默认 `smoke_test` 模式（~30 分钟跑完全流程）。切换到完整模式：修改 `configs/run_config.yaml` 中 `run_mode: "full_run"`（~4-6 小时）。

## Notebook 目录（11 个）

| # | Notebook | 内容 |
|---|---------|------|
| 00 | project_overview | 项目概览与 7 个核心概念 |
| 01 | data_exploration | 8 个数据集统计与 7 张可视化 |
| 02 | safety_taxonomy | 14 类风险分类体系 + 覆盖度热力图 |
| 03 | datajuicer_text_pipeline | 文本清洗 Pipeline（宽松阈值） |
| 04 | datajuicer_multimodal_pipeline | OCR + CLIP + 版权 embedding 库 |
| 05 | data_augmentation | 5 种增强策略 + 对比样本演示 |
| 06 | text_classifier_training | DistilBERT 训练 + 错误分析 |
| 07 | multimodal_classification | CLIP Head 训练 + 版权检测评估 |
| 08 | benchmark_evaluation | 4 个 Benchmark 评估 + 阈值分析 |
| 09 | ablation_study | 6 组消融实验 + 雷达图 |
| 10 | dashboard_report | Dashboard + 最终报告 + 输出文件生成 |

## 运行环境

- MacBook Pro M4 Max（16核CPU, 40核GPU, 128GB统一内存）
- PyTorch MPS backend 加速
- Python 3.9+

## 项目结构

```
safety-dataset/
├── configs/                # 配置文件
│   ├── run_config.yaml         # smoke_test / full_run 两档控制
│   ├── api_config.yaml         # Anthropic API 配置
│   ├── eval_config.yaml        # Benchmark + 消融实验配置
│   ├── text_cleaning.yaml      # Data-Juicer 文本清洗配置
│   └── image_text_cleaning.yaml
├── data/                   # 数据目录
│   ├── raw/                    # 原始下载数据（8 个数据集）
│   ├── unified/                # 统一格式（text_safety + multimodal_safety）
│   ├── cleaned/                # 清洗后数据
│   ├── augmented/              # 增强数据 + IP embedding 库 + 攻击图片
│   └── final/                  # 最终交付（SFT + DPO + Eval + Card）
├── notebooks/              # 11 个 Jupyter Notebook（00-10）
├── src/                    # 源代码模块
│   ├── data_download/          # 数据下载 + 格式转换
│   ├── cleaning/               # DJ 清洗 + OCR + CLIP 跨模态验证
│   ├── augmentation/           # 对比生成 + 类别平衡 + 印刷术攻击 + 版权 + 改写
│   ├── training/               # DistilBERT + CLIP Head 分类器
│   ├── evaluation/             # Benchmark 评估 + 版权检测 + 消融实验
│   └── utils/                  # 配置加载 + 可视化
├── scripts/                # 全流程运行脚本（7 个）
├── results/                # 输出
│   ├── figures/                # 23 张可视化 PNG
│   ├── models/                 # 8 个模型 checkpoint
│   ├── training/               # 训练结果 JSON
│   ├── evaluation/             # 评估结果 JSON
│   ├── ablation/               # 消融实验结果 JSON
│   └── final_summary.json      # 全项目汇总
└── docs/                   # 文档（交互记录）
```

## 数据流水线

```
8 个公开数据集 (Raw)
  ↓ format_converter.py
统一格式 (9,480 文本 + 500 多模态)
  ↓ text_safety_pipeline.py
清洗后 (6,529 条, 保留率 68.9%)
  ↓ run_augmentation.py
增强数据 (243 条: 对比+类别+印刷术+版权+改写)
  ↓ generate_report.py
最终交付 (SFT 6,772 + DPO 4,225 + Eval 1,305)
```

## 消融实验设计

| 实验 | 去掉什么 | 预期影响 |
|------|---------|---------|
| Full | 无 | 对照基准 |
| -Safety | WildGuardMix | AUC 大幅下降 |
| -Contrastive | 对比无害数据 | Over-refusal 升高 |
| -Augmentation | 合成增强 | 稀有类别 F1 下降 |
| -Copyright | 版权数据 | 版权 Recall ≈ 0 |
| -ToxiGen | ToxiGen | 仇恨言论 F1 略降 |

## 踩坑记录

详见 `docs/interaction_log.md`，主要包括：
1. HuggingFace Gated Dataset 认证 → 合成数据替代
2. SafeBench category 字段异常 → 关键词推断
3. XSTest 多文件重复 + 标签误判 → 只读 prompts.jsonl
4. Python 3.9 f-string 限制 → 提取变量
5. Notebook JSON 中文引号 → Python 脚本生成
6. Notebook source 行分割 → `_split_source()` 函数
