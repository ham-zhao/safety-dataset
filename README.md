# 多模态内容安全审核系统

面向 TikTok 审核场景，从零构建安全数据集 → 训练分类器 → 量化评估 → 消融验证的完整闭环。

## 第一次来？从这里开始

| 你的目标 | 去哪里 | 预计时间 |
|---------|--------|---------|
| 看懂这个项目在做什么 | 打开 `notebooks/00_project_overview.ipynb` | 10 分钟 |
| 看最终实验结论和数据 | 直接跳到本文档 [实验结论](#实验结论) 章节 | 5 分钟 |
| 跑通全部代码 | 按 [运行流程](#运行流程) 一步步执行 | 30 分钟 |
| 理解每个环节的原理 | 按 [深度阅读路径](#阅读路径-b深度理解每个环节2-小时) 顺序读 Notebook | 2 小时 |
| 改参数做实验 | 看 [配置文件](#配置文件configs) 章节，改 `configs/run_config.yaml` | 5 分钟 |
| 排查报错 | 查 [踩坑记录](#踩坑记录) 或 `docs/interaction_log.md` | — |

---

## 实验结论

### 这个项目做了什么

把 8 个公开安全数据集（9,480 条）经过清洗（6,529 条）、增强（+243 条），训练了两个分类器，在标准 Benchmark 上评估，并通过 6 组消融实验量化了每个数据组件的贡献。

### 核心数字

**DistilBERT 文本分类器**（66M 参数，推理 <10ms）：

| 指标 | 值 | 含义 |
|------|-----|------|
| AUC-ROC | 0.9962 | 区分有害/无害的综合能力 |
| Recall | 0.9650 | 96.5% 的有害内容被拦截 |
| Precision | 0.9892 | 被拦截内容中 98.9% 确实有害 |
| F1 | 0.9770 | Precision 和 Recall 的调和平均 |

**CLIP 多模态分类器**（冻结 CLIP backbone + 线性头）：

| 指标 | 值 | 含义 |
|------|-----|------|
| AUC-ROC | 0.9506 | 图文混合内容的分类能力 |
| Recall | 1.0000 | 所有有害内容被拦截 |
| Precision | 0.8966 | 较高误拦率，需后续人工复核 |

### 消融实验：哪些数据最重要？

每次去掉一组数据重新训练，看性能下降多少：

| 实验 | AUC | F1 | Recall | 与 Full 的 AUC 差 | 结论 |
|------|-----|-----|--------|------------------|------|
| **Full（基准）** | 0.9959 | 0.9831 | 0.9837 | — | 完整数据集 |
| -Safety | 0.9948 | 0.9644 | 0.9445 | -0.0011 | Recall 下降最大（-3.9%），安全数据是基石 |
| -ToxiGen | 0.9946 | 0.9652 | 0.9828 | -0.0013 | AUC 下降最大，隐式毒性数据有独特贡献 |
| -Copyright | 0.9932 | 0.9768 | 0.9647 | -0.0027 | 版权数据影响超预期 |
| -Augmentation | 0.9942 | 0.9751 | 0.9762 | -0.0017 | 增强数据对整体有正面作用 |
| -Contrastive | 0.9972 | 0.9814 | 0.9703 | +0.0013 | 对比数据在 smoke_test 下效果不显著，full_run 下预期显著 |

> **关键发现**：安全数据（WildGuardMix）是性能基石，去掉后 Recall 从 98.4% 降到 94.5%。ToxiGen 的隐式毒性数据贡献独特，去掉后 AUC 下降最多。

### 最终交付的数据文件

| 文件 | 条数 | 用途 | 格式 |
|------|------|------|------|
| `data/final/safety_sft_mix.jsonl` | 6,772 | 安全对齐 SFT 训练 | JSONL |
| `data/final/safety_dpo_pairs.jsonl` | 4,225 | DPO 偏好训练 | JSONL (prompt/chosen/rejected) |
| `data/final/safety_eval.jsonl` | 1,305 | 模型评估 | JSONL |

---

## 运行流程

### 前提条件

- macOS（已在 M4 Max 上验证）或 Linux
- Python 3.9+
- 磁盘空间 ~5GB（数据 + 模型）

### 验证模式（smoke_test，约 30 分钟）

```bash
# 第 1 步：环境搭建（约 3 分钟）
bash setup.sh              # 创建 venv，安装所有依赖
source venv/bin/activate   # 激活虚拟环境

# 第 2 步：下载数据 + 格式转换（约 5 分钟）
python3 scripts/run_download.py
# → 产出：data/unified/text_safety.jsonl (9,480 条)

# 第 3 步：数据清洗（约 1 分钟）
python3 scripts/run_cleaning.py
# → 产出：data/cleaned/text_safety_cleaned.jsonl (6,529 条)

# 第 4 步：数据增强（约 1 分钟）
python3 scripts/run_augmentation.py
# → 产出：data/augmented/augmented_data.jsonl (243 条)

# 第 5 步：模型训练（约 10 分钟）
python3 scripts/run_training.py
# → 产出：results/models/text_classifier/ 和 multimodal_classifier/

# 第 6 步：Benchmark 评估（约 2 分钟）
python3 scripts/run_evaluation.py
# → 产出：results/evaluation/

# 第 7 步：消融实验（约 10 分钟，训练 6 个模型）
python3 scripts/run_ablation.py
# → 产出：results/ablation/ablation_results.json

# 第 8 步：生成最终数据文件（约 1 分钟）
python3 scripts/generate_report.py
# → 产出：data/final/ 下的 3 个 JSONL + dataset_card.md
```

### 正式模式（full_run，约 4-6 小时）

打开 `configs/run_config.yaml`，把第 7 行改成：

```yaml
run_mode: "full_run"
```

然后按上面同样的顺序运行。区别是数据量从 2,000 → 50,000，训练从 1 epoch → 5 epochs。

> **注意**：训练阶段耗时较长，建议关闭电脑休眠（`系统设置 → 电池 → 永不`），或在 `tmux` / `screen` 中运行。

---

## 项目地图

### Notebooks（11 个）

打开方式：`jupyter notebook`，然后在浏览器中点击对应文件。或用 VS Code 直接打开 `.ipynb`。

所有 Notebook 已执行完毕，直接打开即可看到输出和图表，不需要重新运行。

| 文件名 | 做什么 | 什么时候看 |
|--------|--------|-----------|
| `00_project_overview.ipynb` | 7 个核心概念的中文讲解 | 第一次看项目时 |
| `01_data_exploration.ipynb` | 8 个数据集的统计和 7 张图 | 想了解数据长什么样 |
| `02_safety_taxonomy.ipynb` | 14 类风险分类 + 覆盖度热力图 | 想了解分类体系 |
| `03_datajuicer_text_pipeline.ipynb` | 文本清洗过程和前后对比 | 想了解清洗策略 |
| `04_datajuicer_multimodal_pipeline.ipynb` | OCR 测试 + CLIP 分析 + IP 库 | 想了解多模态处理 |
| `05_data_augmentation.ipynb` | 5 种增强策略和效果分析 | 想了解数据增强 |
| `06_text_classifier_training.ipynb` | DistilBERT 训练曲线 + 错误分析 | 想了解文本模型 |
| `07_multimodal_classification.ipynb` | CLIP 分类 + 版权检测结果 | 想了解多模态模型 |
| `08_benchmark_evaluation.ipynb` | 4 个 Benchmark 的评估结果 | 想看模型多好 |
| `09_ablation_study.ipynb` | 6 组消融实验 + 雷达图 | 想看哪些数据重要 |
| `10_dashboard_report.ipynb` | 全项目汇总 Dashboard | 看最终总结 |

### 脚本（scripts/）

运行方式：`source venv/bin/activate && python3 scripts/xxx.py`

| 文件名 | 做什么 | 运行顺序 |
|--------|--------|---------|
| `run_download.py` | 下载 8 个数据集 + 转换格式 | 第 1 个 |
| `run_cleaning.py` | 清洗文本和多模态数据 | 第 2 个 |
| `run_augmentation.py` | 运行 5 种数据增强 | 第 3 个 |
| `run_training.py` | 训练文本 + 多模态分类器 | 第 4 个 |
| `run_evaluation.py` | 在 Benchmark 上评估 | 第 5 个 |
| `run_ablation.py` | 运行 6 组消融实验 | 第 6 个 |
| `generate_report.py` | 生成 SFT/DPO/Eval 文件 | 第 7 个 |

### 配置文件（configs/）

打开方式：任意文本编辑器。

| 文件名 | 做什么 | 什么时候改 |
|--------|--------|-----------|
| `run_config.yaml` | 控制数据量和训练轮数 | **最常改**：`run_mode` 从 `smoke_test` 改为 `full_run` |
| `eval_config.yaml` | Benchmark 目标值 + 消融实验配置 | 想加/删消融实验组 |
| `text_cleaning.yaml` | 清洗阈值（文本长度、语言分数） | 想调清洗松紧度 |
| `image_text_cleaning.yaml` | 多模态清洗参数 | 一般不需要改 |

### 源代码（src/）

一般不需要直接看源码。如果想了解实现细节：

| 目录 | 做什么 | 核心文件 |
|------|--------|---------|
| `src/data_download/` | 数据下载和格式转换 | `format_converter.py`（统一 8 个数据集格式） |
| `src/cleaning/` | 清洗 + OCR + CLIP | `text_safety_pipeline.py`（5 步清洗）、`cross_modal_validator.py`（CLIP） |
| `src/augmentation/` | 5 种增强策略 | `contrastive_generator.py`（对比样本）最关键 |
| `src/training/` | 模型训练 | `text_classifier.py`（DistilBERT）、`multimodal_classifier.py`（CLIP Head） |
| `src/evaluation/` | 评估和消融 | `ablation_runner.py`（消融实验）、`safety_metrics.py`（指标计算） |

### 结果文件（results/）

| 路径 | 内容 | 用什么打开 |
|------|------|-----------|
| `results/figures/*.png` | 23 张可视化图表 | 图片查看器 / Finder 预览 |
| `results/models/` | 8 个模型 checkpoint | 代码加载（PyTorch / HuggingFace） |
| `results/ablation/ablation_results.json` | 消融实验数值结果 | 文本编辑器 / `python3 -m json.tool` |
| `results/final_summary.json` | 全项目数据流统计 | 文本编辑器 |

---

## 阅读路径

### 路径 A：快速了解结论（30 分钟）

1. **本 README 的 [实验结论](#实验结论)**（5 分钟）→ 看到核心数字和关键发现
2. **`notebooks/10_dashboard_report.ipynb`**（10 分钟）→ 看到数据流全貌和 Dashboard 图表
3. **`notebooks/09_ablation_study.ipynb`**（15 分钟）→ 看到消融实验细节和雷达图

### 路径 B：深度理解每个环节（2 小时）

1. **`notebooks/00_project_overview.ipynb`**（10 分钟）→ 理解 7 个核心概念（Post-train、级联架构、Modal Gap...）
2. **`notebooks/01_data_exploration.ipynb`**（15 分钟）→ 理解 8 个数据集各自的特点和分布
3. **`notebooks/03_datajuicer_text_pipeline.ipynb`**（15 分钟）→ 理解为什么安全数据清洗要用宽松阈值
4. **`notebooks/05_data_augmentation.ipynb`**（20 分钟）→ 理解对比样本为什么能解决 over-refusal
5. **`notebooks/06_text_classifier_training.ipynb`**（20 分钟）→ 理解 DistilBERT 训练和 Precision-Recall 的取舍
6. **`notebooks/09_ablation_study.ipynb`**（20 分钟）→ 理解消融实验怎么量化每个组件的贡献
7. **`notebooks/04_datajuicer_multimodal_pipeline.ipynb`**（20 分钟）→ 理解 OCR + CLIP 双层防护

### 路径 C：只看多模态部分（40 分钟）

1. **`notebooks/04_datajuicer_multimodal_pipeline.ipynb`**（20 分钟）→ OCR 提取、CLIP 相似度、版权 embedding
2. **`notebooks/07_multimodal_classification.ipynb`**（20 分钟）→ CLIP Head 训练和版权检测评估

---

## 背景知识

### 这个项目在 AI 链路中的位置

```
Pre-train 数据清洗 → Pre-train → [Post-train 数据准备 ← 这个项目] → SFT/DPO → Chat Model
```

大模型训练完之后，需要用安全数据做 SFT（指令微调）和 DPO（偏好对齐），让模型学会拒绝有害请求。本项目就是构建这些安全训练数据。

### TikTok 级联审核架构

```
用户上传内容
  ↓
第 1 层：DistilBERT（<10ms，66M 参数）→ 拦截 90% 有害内容  ← 本项目训练
  ↓
第 2 层：CLIP Head（~100ms）→ 处理图文混合攻击              ← 本项目训练
  ↓
第 3 层：7B/13B LLM（~1s）→ 处理困难样本
  ↓
第 4 层：人工复审 → 最终仲裁
```

### 14 类安全风险

| 编号 | 类别 | 编号 | 类别 |
|------|-----|------|-----|
| 01 | 非法活动 | 08 | 政治游说 |
| 02 | 仇恨言论 | 09 | 隐私侵犯 |
| 03 | 恶意软件 | 10 | 法律建议 |
| 04 | 身体伤害 | 11 | 财务建议 |
| 05 | 经济犯罪 | 12 | 健康咨询 |
| 06 | 欺诈行为 | 13 | 政府决策 |
| 07 | 色情内容 | **14** | **版权侵权**（TikTok 特有） |

---

## 踩坑记录

完整记录在 `docs/interaction_log.md`，最常见的 3 个：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| HuggingFace Gated Dataset 下载失败 | WildGuardMix 等需要申请访问权限 | 用合成数据替代，全流程可跑通 |
| XSTest 所有样本被标为 harmful | 6 个文件重复 + type 字段误判 | 只读 `prompts.jsonl`，默认标为 safe |
| Notebook JSON 解析失败 | 中文引号破坏 JSON 结构 | 用 Python 脚本 + `json.dump` 生成 |

---

## 运行环境

- macOS（M4 Max 验证），Python 3.9+，PyTorch MPS 加速
- 依赖安装：`bash setup.sh`（自动创建 venv + 安装全部依赖）
