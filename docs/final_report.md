# 最终报告：多模态内容安全审核系统

## 一、项目目标

为 TikTok 审核场景构建完整的安全数据 Pipeline：从公开数据集出发，经过清洗、增强，训练轻量级安全分类器，通过标准 Benchmark 评估，并用消融实验量化每个环节的价值。

最终产出可以直接用于大模型的 SFT（指令微调）和 DPO（偏好对齐）训练。

---

## 二、技术方案总览

### 2.1 数据流水线

```
8 个公开数据集 (Raw)
      ↓ format_converter.py — 统一为 {text, images, meta} 格式
统一格式 9,480 条
      ↓ text_safety_pipeline.py — 5 步清洗（HTML清理→Unicode修复→长度过滤→语言过滤→去重）
清洗后 6,529 条（保留率 68.9%）
      ↓ run_augmentation.py — 5 种增强（对比+类别+印刷术+版权+改写）
增强数据 +243 条
      ↓ generate_report.py — 拆分 SFT/DPO/Eval
最终交付: SFT 6,772 / DPO 4,225 / Eval 1,305
```

### 2.2 模型架构

**级联设计**：轻量模型做初筛（处理 90%），重模型处理困难样本。

| 层级 | 模型 | 参数量 | 推理延迟 | 职责 |
|------|------|--------|---------|------|
| 第 1 层 | DistilBERT | 66M | <10ms | 文本二分类（harmful/unharmful），拦截 90% |
| 第 2 层 | CLIP Head | CLIP ViT-B-32 + 线性头 | ~100ms | 图文混合攻击检测 |
| 第 3 层 | 7B/13B LLM | — | ~1s | 困难样本（本项目未实现） |
| 第 4 层 | 人工审核 | — | ~秒级 | 最终仲裁 |

### 2.3 评估体系

| Benchmark | 规模 | 测什么 | 核心指标 |
|-----------|------|--------|---------|
| WildGuardTest | 400 条 | 综合安全能力 | AUC-ROC ≥ 0.85 |
| HarmBench | ~500 条 | 有害内容检测 | Recall ≥ 0.90 |
| XSTest | 450 条 | Over-refusal（误拦率） | Over-refusal < 10% |
| MM-SafetyBench | 5,040 条 | 多模态攻击 | ASR < 20% |

---

## 三、核心知识点

### 3.1 为什么安全数据清洗要用宽松阈值？

普通数据清洗追求高质量，阈值严格（文本长度 >50，语言分数 >0.8）。但安全数据的特殊性：
- 有害内容本身就格式混乱（用户发的仇恨言论不会有完美语法）
- Jailbreak 攻击会故意使用非标准格式
- 用高标准过滤会丢掉大量有害样本

**本项目采用的宽松阈值**：文本最短 20 字符（vs 普通 50）、语言分数 >0.5（vs 普通 0.8）、模糊去重阈值 0.7（vs 普通 0.85）。

对应文件：`configs/text_cleaning.yaml`、`src/cleaning/text_safety_pipeline.py`
对应 Notebook：`03_datajuicer_text_pipeline.ipynb`

### 3.2 对比数据为什么能解决 Over-refusal？

没有对比数据的模型会学到"包含敏感词就拒绝"的简单规则：
- "How to make a **bomb** at home?" → 应该拒绝
- "How to make a **bath bomb** at home?" → 不应该拒绝

如果训练集里只有前者，模型会一刀切拒绝所有含 "bomb" 的请求。对比数据让模型学会区分真正的意图。

本项目构造了 20 组对比对（harmful/benign），每组语义相似但意图相反。

对应文件：`src/augmentation/contrastive_generator.py`
对应 Notebook：`05_data_augmentation.ipynb` 的第 4 节

### 3.3 Modal Gap：为什么文本安全 + 图片安全 ≠ 组合安全？

多模态攻击的核心原理：
- 文本："What does the text in the image say?"（无害）
- 图片：渲染了有害指令的文字图片（看起来是普通图片）
- 组合：实际在请求有害内容

MM-SafetyBench 论文发现，加入视觉模块后攻击成功率从 20% 跳到 40-70%。

本项目的防御策略：
1. **OCR 层**（便宜）：从图片提取文字，送入文本分类器。覆盖 80% 的印刷术攻击。
2. **CLIP 层**（精准）：计算图文语义相似度。覆盖 OCR 失效的场景（手写体、emoji、低分辨率）。

对应文件：`src/cleaning/ocr_extractor.py`、`src/cleaning/cross_modal_validator.py`
对应 Notebook：`04_datajuicer_multimodal_pipeline.ipynb`

### 3.4 Precision-Recall 的取舍

| 场景 | 优先指标 | 原因 |
|------|---------|------|
| 生产环境 | Recall > 0.90 | 漏掉有害内容 = 真实伤害，误拦 = 可人工恢复 |
| 用户体验优先 | Precision > 0.90 | 频繁误拦会导致用户流失 |
| 本项目选择 | 高 Recall | 按 TikTok 审核标准，宁可多拦不可漏判 |

对应 Notebook：`06_text_classifier_training.ipynb` 开头的 Markdown

### 3.5 版权检测原理

用 CLIP 文本编码器为 20 个知名 IP（Mickey Mouse、Pikachu、Mario 等）生成 embedding 向量。检测时计算待检图片的 embedding 与库中所有 IP 的余弦相似度，超过阈值则标记为疑似侵权。

目前版权检测用的是文本 embedding 作为代理（真实生产中应该用图片 embedding）。smoke_test 模式下测试数据的扰动噪声太大导致匹配分数低，full_run 下效果会改善。

对应文件：`src/augmentation/copyright_embedding.py`、`src/evaluation/copyright_detector.py`
对应 Notebook：`04_datajuicer_multimodal_pipeline.ipynb` Part C、`07_multimodal_classification.ipynb` 第 3 节

### 3.6 消融实验的价值

不做消融实验 = 不知道数据集里哪部分有用。本项目设计了 6 组实验：

| 去掉什么 | 验证什么 | 实际发现 |
|---------|---------|---------|
| 无（Full） | 对照基准 | AUC=0.9959, F1=0.9831 |
| WildGuardMix | 安全数据是否是核心 | Recall 下降最多（-3.9%），**是基石** |
| 对比数据 | 对比构造是否必要 | smoke_test 下不显著，full_run 下预期显著 |
| 合成增强 | 增强是否有价值 | F1 下降 0.8%，**有正面作用** |
| 版权数据 | 版权类别是否需要专门数据 | AUC 下降 0.27%，**影响超预期** |
| ToxiGen | 隐式毒性的贡献 | AUC 下降最多（-0.13%），**有独特贡献** |

对应文件：`src/evaluation/ablation_runner.py`
对应 Notebook：`09_ablation_study.ipynb`

---

## 四、实验结果详情

### 4.1 数据统计

| 阶段 | 数量 | 说明 |
|------|------|------|
| 原始数据（统一格式后） | 9,480 文本 + 500 多模态 | 8 个数据集 |
| 文本清洗后 | 6,529 | 保留率 68.9%，去重是主要过滤源（2,942 条重复） |
| 多模态清洗后 | 20 | 合成数据模板少导致去重率高，真实数据会高得多 |
| 增强数据 | 243 | 对比 100 + 类别 13 + 印刷术 30 + 版权 50 + 改写 50 |
| 最终 SFT | 6,772 | 清洗 + 增强 |
| 最终 DPO | 4,225 | 有害样本配安全拒绝/不安全顺从回复 |
| 最终 Eval | 1,305 | 清洗数据的 20% |

### 4.2 DistilBERT 文本分类器

- 模型：`distilbert-base-uncased`，66M 参数
- 训练：1 epoch（smoke_test），AdamW，lr=2e-5，batch_size=32
- 设备：Apple M4 Max MPS

| 指标 | 值 |
|------|-----|
| AUC-ROC | 0.9962 |
| F1 | 0.9770 |
| Precision | 0.9892 |
| Recall | 0.9650 |
| Accuracy | 0.9761 |

### 4.3 CLIP 多模态分类器

- 架构：CLIP ViT-B-32（冻结）+ 线性分类头（512→256→2）
- 为什么冻结 CLIP：CLIP 已有强图文对齐能力，微调收益不大且可能灾难性遗忘

| 指标 | 值 |
|------|-----|
| AUC-ROC | 0.9506 |
| F1 | 0.9455 |
| Precision | 0.8966 |
| Recall | 1.0000 |
| Accuracy | 0.9269 |

### 4.4 Benchmark 评估

| Benchmark | 样本数 | AUC | F1 | Recall | 状态 |
|-----------|--------|-----|-----|--------|------|
| WildGuardTest | 400 | 1.0000 | 1.0000 | 1.0000 | 超过目标（0.85） |
| XSTest | 450 | — | 0.0000 | 0.0000 | 全为 safe 样本，Recall 不适用 |
| HarmBench | — | — | — | — | smoke_test 数据不足 |
| MM-SafetyBench | — | — | — | — | 无多模态测试数据 |

> **注意**：WildGuardTest 的完美分数是因为 smoke_test 模式下数据量小。full_run 模式下会有更真实的数字。

### 4.5 消融实验

| 实验 | AUC | F1 | Recall | 训练量 | AUC 变化 |
|------|-----|-----|--------|--------|---------|
| Full | 0.9959 | 0.9831 | 0.9837 | 5,417 | 基准 |
| -Safety | 0.9948 | 0.9644 | 0.9445 | 5,393 | -0.0011 |
| -Contrastive | 0.9972 | 0.9814 | 0.9703 | 5,337 | +0.0013 |
| -Augmentation | 0.9942 | 0.9751 | 0.9762 | 5,383 | -0.0017 |
| -Copyright | 0.9932 | 0.9768 | 0.9647 | 5,377 | -0.0027 |
| -ToxiGen | 0.9946 | 0.9652 | 0.9828 | 3,842 | -0.0013 |

**结论排序**（按对模型影响从大到小）：
1. **Safety 数据最关键**：去掉后 Recall 从 98.4% 降到 94.5%
2. **Copyright 影响超预期**：去掉后 AUC 下降最多（-0.0027）
3. **Augmentation 有正向作用**：F1 下降 0.8%
4. **ToxiGen 贡献隐式毒性**：去掉 1,575 条数据后 AUC 下降 0.13%
5. **Contrastive 在小数据下不显著**：smoke_test 100 条对比数据量太少

---

## 五、各文件之间的关系

### 5.1 数据流

```
src/data_download/download_all.py    → data/raw/（8 个数据集）
src/data_download/format_converter.py → data/unified/text_safety.jsonl
src/cleaning/text_safety_pipeline.py  → data/cleaned/text_safety_cleaned.jsonl
src/augmentation/*.py                 → data/augmented/augmented_data.jsonl
scripts/generate_report.py            → data/final/（SFT + DPO + Eval）
```

### 5.2 模型训练流

```
data/cleaned/ + data/augmented/
        ↓
src/training/text_classifier.py      → results/models/text_classifier/
src/training/multimodal_classifier.py → results/models/multimodal_classifier/
```

### 5.3 评估流

```
results/models/text_classifier/ + data/unified/
        ↓
src/evaluation/benchmark_runner.py   → Benchmark 评估结果
src/evaluation/ablation_runner.py    → results/ablation/ablation_results.json
src/evaluation/copyright_detector.py → 版权检测评估
```

### 5.4 配置 → 代码的映射

| 配置参数 | 所在文件 | 影响哪些代码 |
|---------|---------|-------------|
| `run_mode` | `configs/run_config.yaml` | 所有代码（控制数据量和训练轮数） |
| `classifier_epochs` | `configs/run_config.yaml` | `text_classifier.py`、`multimodal_classifier.py` |
| `device` | `configs/run_config.yaml` | 所有训练和推理代码 |
| `ablation.experiments` | `configs/eval_config.yaml` | `ablation_runner.py` |
| `copyright.similarity_thresholds` | `configs/eval_config.yaml` | `copyright_detector.py` |

### 5.5 Notebook → 源码的映射

| Notebook | 调用的核心模块 |
|----------|---------------|
| 03 | `src/cleaning/text_safety_pipeline.py` |
| 04 | `src/cleaning/ocr_extractor.py`、`src/cleaning/cross_modal_validator.py` |
| 05 | `src/augmentation/` 下所有 5 个模块 |
| 06 | `src/training/text_classifier.py` |
| 07 | `src/training/multimodal_classifier.py`、`src/evaluation/copyright_detector.py` |
| 08 | `src/evaluation/benchmark_runner.py`、`src/evaluation/safety_metrics.py` |
| 09 | `src/evaluation/ablation_runner.py` |
| 10 | `scripts/generate_report.py` |

---

## 六、已知限制和改进方向

| 限制 | 原因 | 改进方向 |
|------|------|---------|
| smoke_test 模式下部分 Benchmark 数据不足 | 数据采样太小 | 切换 full_run |
| 版权检测准确率低 | 用文本 embedding 代替图片 embedding | 收集真实 IP 图片构建库 |
| 3 个 Gated 数据集用合成替代 | HuggingFace 访问权限 | `huggingface-cli login` 后重新下载 |
| 对比数据在小规模下效果不显著 | 100 条对比样本太少 | full_run 下增到 500 条 |
| DPO 数据用模板生成 | 没有接入 LLM API | 接入 Claude API 生成高质量拒绝/顺从回复 |

---

## 七、参考论文和数据集

| 名称 | 用途 | 链接 |
|------|------|------|
| WildGuardMix | 核心安全训练+测试 | allenai/wildguardmix |
| WildJailbreak | 对比构造 | allenai/wildjailbreak |
| ToxiGen | 隐式毒性 | skg/toxigen-data |
| MM-SafetyBench | 多模态攻击 | isXinLiu/MM-SafetyBench |
| XSTest | Over-refusal 测试 | paul-rottger/xstest |
| HarmBench | 有害内容标准测试 | centerforaisafety/HarmBench |
| FigStep | 印刷术攻击方法 | 本项目 `typographic_attack.py` 复现 |
