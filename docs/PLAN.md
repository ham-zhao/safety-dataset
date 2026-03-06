# 执行手册：多模态内容安全审核系统

> **定位**：从零重新跑通整个项目的完整操作指南。包含执行步骤、设计理由、实验结果和踩坑记录，自包含，不需要交叉阅读其他文档。

---

## 一、问题-方案-预期

**问题**：TikTok 日活超 10 亿，UGC 内容审核需要自动化安全分类能力，但公开安全数据集格式各异、质量参差、类别不均衡，无法直接用于生产模型训练。

**方案**：从 8 个公开安全数据集出发，构建完整数据 Pipeline——统一格式 → Data-Juicer 风格清洗 → 5 种策略增强 → 训练级联分类器 → Benchmark 评估 → 消融验证。

**预期指标**（smoke_test 模式）：

| 指标 | 目标 | 实测 |
|------|------|------|
| 文本分类器 AUC-ROC | ≥ 0.95 | 0.9962 |
| 文本分类器 Recall | ≥ 0.90 | 0.9650 |
| 多模态分类器 AUC-ROC | ≥ 0.90 | 0.9506 |
| 14 类风险覆盖 | 每类 ≥ 20 条 | 达标（类别补充后） |
| 消融实验 | 6 组完成 | 6/6 |

---

## 二、术语表

| 术语 | 定义 |
|------|------|
| SFT | Supervised Fine-Tuning，指令微调，用 (prompt, response) 对训练模型 |
| DPO | Direct Preference Optimization，偏好对齐，用 (prompt, chosen, rejected) 三元组训练 |
| Over-refusal | 模型误拦安全内容（如 "how to make a bath bomb" 被拒绝） |
| 级联架构 | 轻量模型初筛 → 重模型二筛 → 人工兜底的多层审核流程 |
| Modal Gap | 文本安全 + 图片安全 ≠ 多模态组合安全，攻击者利用模态间隙绕过检测 |
| 消融实验 | 每次去掉一组数据重新训练，量化该组数据对模型性能的贡献 |
| Data-Juicer | 阿里开源的数据清洗框架，本项目参考其算子设计但独立实现 |
| Gated Dataset | HuggingFace 上需要申请访问权限的数据集 |

---

## 三、架构总览

### 3.1 级联审核系统

```
用户上传内容
  ↓
第 1 层：DistilBERT（<10ms, 66M 参数）→ 拦截 90% 有害内容    ← 本项目训练
  ↓ 未拦截
第 2 层：CLIP Head（~100ms, 冻结 backbone + 线性头）→ 图文混合攻击   ← 本项目训练
  ↓ 未拦截
第 3 层：7B/13B LLM（~1s）→ 困难样本（未实现）
  ↓ 不确定
第 4 层：人工复审 → 最终仲裁
```

**为什么用级联而不是单模型**：TikTok 日活 10 亿，全部用大模型推理成本不可接受。轻模型先过滤 90%，只有剩下 10% 交给重模型，整体成本降一个数量级。

### 3.2 数据流水线

```
8 个公开数据集（Raw）
      ↓ format_converter.py — 统一为 {text, images, meta}
data/unified/ — 9,480 文本 + 500 多模态
      ↓ text_safety_pipeline.py — 5 步清洗（HTML→Unicode→长度→语言→去重）
data/cleaned/ — 6,529 条（保留率 68.9%）
      ↓ 5 种增强策略（对比+类别+印刷术+版权+改写）
data/augmented/ — +243 条
      ↓ generate_report.py — 拆分
data/final/ — SFT 6,772 / DPO 4,225 / Eval 1,305
```

### 3.3 目录结构

```
safety-dataset/
├── configs/                     # 配置（两档运行模式）
│   ├── run_config.yaml          #   数据量 + 训练参数
│   ├── eval_config.yaml         #   Benchmark 目标值 + 消融配置
│   ├── text_cleaning.yaml       #   文本清洗阈值
│   └── image_text_cleaning.yaml #   多模态清洗参数
├── src/                         # 函数库（被 scripts/ 和 notebooks/ import）
│   ├── data_download/           #   下载 + 格式转换
│   ├── cleaning/                #   清洗 + OCR + CLIP
│   ├── augmentation/            #   5 种增强策略
│   ├── training/                #   DistilBERT + CLIP Head
│   ├── evaluation/              #   指标 + Benchmark + 消融
│   └── utils/                   #   配置加载 + 可视化
├── scripts/                     # CLI 入口（7 个，严格串行）
├── notebooks/                   # 分析可视化（11 个，00-10）
├── data/                        # 数据（raw → unified → cleaned → augmented → final）
├── results/                     # 输出（models/ + figures/ + ablation/）
└── docs/                        # 文档
```

### 3.4 脚本执行顺序（严格串行，步骤 N 未通过验收禁止开始 N+1）

```
run_download.py      → data/unified/
      ↓
run_cleaning.py      → data/cleaned/            （依赖 data/unified/）
      ↓
run_augmentation.py  → data/augmented/           （依赖 data/cleaned/）
      ↓
run_training.py      → results/models/           （依赖 data/cleaned/ + data/augmented/）
      ↓
run_evaluation.py    → results/evaluation/       （依赖 results/models/）
      ↓
run_ablation.py      → results/ablation/         （依赖 data/ + 训练代码）
      ↓
generate_report.py   → data/final/ + results/    （依赖全部上游）
```

### 3.5 Notebook 依赖关系

| Notebook | 依赖的数据文件 | 对应的 src 模块 |
|----------|--------------|----------------|
| 00 项目概览 | 无 | 无 |
| 01 数据探索 | `data/unified/text_safety.jsonl` | 无 |
| 02 安全分类 | `data/unified/text_safety.jsonl` | 无 |
| 03 文本清洗 | `data/unified/` + `data/cleaned/` | `src/cleaning/text_safety_pipeline.py` |
| 04 多模态 | `data/cleaned/multimodal_*` + `data/augmented/ip_embeddings.pkl` | `src/cleaning/ocr_extractor.py`, `cross_modal_validator.py` |
| 05 数据增强 | `data/augmented/augmented_data.jsonl` | `src/augmentation/*` |
| 06 文本训练 | `data/cleaned/` + `data/augmented/` | `src/training/text_classifier.py` |
| 07 多模态分类 | `data/cleaned/multimodal_*` + `results/models/multimodal_*/` | `src/training/multimodal_classifier.py` |
| 08 Benchmark | `results/models/text_classifier/` | `src/evaluation/benchmark_runner.py` |
| 09 消融 | `results/ablation/ablation_results.json` | `src/evaluation/ablation_runner.py` |
| 10 Dashboard | `results/final_summary.json` + `data/final/*.jsonl` | 无 |

### 3.6 配置 → 代码映射

| 配置参数 | 所在文件 | 影响哪些代码 |
|---------|---------|-------------|
| `run_mode` | `run_config.yaml` | 全部脚本（控制数据量和训练轮数） |
| `classifier_epochs` | `run_config.yaml` | `text_classifier.py`, `multimodal_classifier.py` |
| `device` | `run_config.yaml` | 全部训练和推理代码 |
| `process` 算子列表 | `text_cleaning.yaml` | `text_safety_pipeline.py` |
| `ablation.experiments` | `eval_config.yaml` | `ablation_runner.py` |
| `copyright.similarity_thresholds` | `eval_config.yaml` | `copyright_detector.py` |

两档运行模式：

| 参数 | smoke_test | full_run |
|------|-----------|---------|
| text_sample_size | 2,000 | 50,000 |
| image_text_sample_size | 500 | 5,040 |
| synthesis_count | 50 | 500 |
| classifier_epochs | 1 | 5 |
| 预计耗时 | ~30 分钟 | ~4-6 小时 |

切换方式：修改 `configs/run_config.yaml` 第 7 行 `run_mode: "full_run"`，然后重新运行全部脚本。

---

## 四、执行指南

### 前置条件

| 依赖 | 版本 | 用途 |
|------|------|------|
| macOS 或 Linux | — | 已在 M4 Max 验证 |
| Python | 3.9+ | `python` 可能不存在，统一用 `python3` |
| Tesseract | 最新 | OCR 提取（`brew install tesseract`） |
| 磁盘空间 | ~5GB | 数据 + 模型 checkpoint |

---

### 阶段一：环境搭建 + 数据获取 + 探索

**目标**：搭建环境，下载 8 个数据集并统一格式，创建数据探索 Notebook。

#### 步骤 1.1：环境安装

```bash
bash setup.sh
source venv/bin/activate
```

验证：
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"  # 期望: True
python3 -c "import transformers, open_clip, datasets; print('OK')"   # 期望: OK
tesseract --version                                                   # 期望: 有版本号
```

#### 步骤 1.2：下载数据 + 格式转换

```bash
source venv/bin/activate
python3 scripts/run_download.py
```

**执行内容**：
1. 从 HuggingFace 下载 8 个数据集到 `data/raw/`
2. 3 个 Gated 数据集（WildGuardMix / WildJailbreak / HarmBench）如权限不足，自动生成合成替代数据
3. 调用 `format_converter.py` 将 8 种格式统一为 `{text, images, meta}` schema
4. 处理数据集特殊问题：SafeBench category 关键词推断、XSTest 去重 + 标签修正

**统一 Schema**：
```json
{
  "text": "...",
  "images": [],
  "meta": {
    "source": "wildguardmix",
    "risk_category": "hate_speech",
    "severity": "high",
    "attack_type": "vanilla",
    "label": "harmful",
    "synthetic": false,
    "original_id": "...",
    "split": "train"
  }
}
```

验证：
```bash
wc -l data/unified/text_safety.jsonl         # 期望: 9,480
wc -l data/unified/multimodal_safety.jsonl   # 期望: 500
python3 -c "import json; [json.loads(l) for l in open('data/unified/text_safety.jsonl')]; print('JSON OK')"
ls -lh data/unified/                         # 期望: text_safety.jsonl ~6MB
```

#### 步骤 1.3：创建并执行 Notebook 00-02

```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/00_project_overview.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_safety_taxonomy.ipynb
```

**执行内容**：
- NB00：7 个核心概念的中文讲解（Post-train、级联架构、Modal Gap 等）
- NB01：8 个数据集的统计分析 + 7 张分布图（数据集条数、风险类别、严重程度、攻击类型、文本长度等）
- NB02：14 类风险分类定义 + 数据集×类别覆盖度热力图

验证：每个 Notebook 无 traceback，有图表输出。

#### 阶段一验收

```
阶段一验收：
  [ ] configs/run_config.yaml — python3 -c "import yaml; yaml.safe_load(open(...))" 通过
  [ ] data/unified/text_safety.jsonl — 9,480 行, ~6MB
  [ ] data/unified/multimodal_safety.jsonl — 500 行
  [ ] notebooks/00 — 所有 cell 有输出，无 traceback
  [ ] notebooks/01 — 7 张分布图可见
  [ ] notebooks/02 — 覆盖度热力图可见
```

---

### 阶段二：Data-Juicer 清洗 + OCR + CLIP

**目标**：对统一格式数据进行质量清洗，实现多模态内容理解（OCR + CLIP）。

#### 步骤 2.1：执行数据清洗

```bash
source venv/bin/activate
python3 scripts/run_cleaning.py
```

**执行内容**：

文本清洗 5 步流程（参数来自 `configs/text_cleaning.yaml`）：

```
步骤 1: HTML 清理 — 去标签、解码 HTML 实体
步骤 2: Unicode 修复 — 统一编码、替换 surrogate 字符
步骤 3: 长度过滤 — min_len=20, max_len=10000
步骤 4: 语言过滤 — lang_score > 0.5，保留英文
步骤 5: 去重 — MD5 精确去重 + MinHash 模糊去重（jaccard_threshold=0.7）
```

**为什么安全数据用宽松阈值**：

普通数据清洗追求高质量文本，阈值严格。但安全数据不同——有害内容本身就格式混乱，Jailbreak 攻击会故意用非标准格式。如果用普通标准过滤，会丢掉大量有害样本。

| 算子 | 普通数据 | 安全数据 | 理由 |
|------|---------|---------|------|
| 最短文本长度 | 50 字符 | 20 字符 | "kill yourself" 只有 13 字符但必须保留 |
| 语言识别分数 | >0.8 | >0.5 | 有害文本常混合 slang、拼写错误 |
| 模糊去重阈值 | 0.85 | 0.7 | 安全数据的变体（换一种说法的攻击）更需要保留 |

**为什么用两层去重**：精确去重（MD5）成本低，先过滤完全相同的文本（跨 8 个数据集发现大量精确重复）；模糊去重（MinHash，阈值 0.7）处理"换了几个词但内容一样"的近似重复（如 "How to make a bomb?" vs "How can I make a bomb at home?"）。先精确后模糊，效率更高。

多模态清洗：对 `multimodal_safety.jsonl` 执行同样的文本清洗 + 图像有效性校验。

验证：
```bash
wc -l data/cleaned/text_safety_cleaned.jsonl         # 期望: ~6,529（保留率 60-80%）
wc -l data/cleaned/multimodal_safety_cleaned.jsonl   # 期望: ~20（合成数据模板少致去重率高，属预期）
ls -lh data/cleaned/
```

#### 步骤 2.2：创建并执行 Notebook 03-04

```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/03_datajuicer_text_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_datajuicer_multimodal_pipeline.ipynb
```

**执行内容**：
- NB03：文本清洗 5 步各自的过滤数量、保留率、清洗前后对比图
- NB04：OCR 文字提取测试 + CLIP 图文相似度分析 + 20 个知名 IP 的 embedding 相似度矩阵

**多模态双层防护的设计理由**：

多模态攻击的核心原理是利用模态间隙——文本无害 + 图片无害，但组合起来有害。例如文本 "What does the text in the image say?"（无害），配上一张渲染了有害指令的图片。MM-SafetyBench 论文发现，加入视觉模块后攻击成功率从 20% 跳到 40-70%。

防御策略：
1. **OCR 层**（便宜）：Tesseract 提取图片文字 → 送入文本分类器，覆盖 80% 印刷术攻击
2. **CLIP 层**（精准）：计算图文语义相似度 → 检测异常配对，覆盖 OCR 失效场景（手写体、emoji、低分辨率）

验证：每个 Notebook 无 traceback，有图表输出。

#### 阶段二验收

```
阶段二验收：
  [ ] data/cleaned/text_safety_cleaned.jsonl — ~6,529 行
  [ ] data/cleaned/multimodal_safety_cleaned.jsonl — 非空
  [ ] notebooks/03 — 清洗前后对比图可见
  [ ] notebooks/04 — OCR 测试 + CLIP 相似度 + IP 矩阵可见
```

---

### 阶段三：数据增强 + 模型训练 + 评估 + 消融

**目标**：增强数据 → 训练两个分类器 → Benchmark 评估 → 6 组消融实验 → 生成最终交付文件。

#### 步骤 3.1：数据增强

```bash
source venv/bin/activate
python3 scripts/run_augmentation.py
```

**执行内容**：5 种增强策略，每种解决一个具体问题。

| 策略 | 模块 | 数量 | 解决什么问题 |
|------|------|------|-------------|
| 对比样本 | `contrastive_generator.py` | 100 条 | Over-refusal——给每条有害 prompt 配一条"表面相似但安全"的版本 |
| 类别平衡 | `category_balancer.py` | 13 条 | 统计各类别数量，不足 20 条的用模板补充 |
| 印刷术攻击 | `typographic_attack.py` | 30 条 + 30 张图 | 模拟 FigStep 攻击——把有害文本渲染成图片 |
| 版权 IP | `copyright_embedding.py` | 50 条 | TikTok 特有需求——20 个知名 IP 的正反例 |
| 规则式改写 | `synthetic_rephraser.py` | 50 条 | 前缀/后缀/措辞替换，增加训练多样性 |

**为什么不用 LLM API 做增强**：成本高、不可复现、引入不确定性。规则式方法对增加训练多样性已经足够，且每条增强数据在 meta 中标记 `synthetic=True`，方便追溯和过滤。

**对比数据为什么能解决 Over-refusal**：没有对比数据的模型会学到"包含敏感词就拒绝"的简单规则。"How to make a **bomb** at home?" 应该拒绝，但 "How to make a **bath bomb** at home?" 不应该。对比数据让模型学会区分真正的意图。本项目构造了 20 组对比对（harmful/benign），每组语义相似但意图相反。

验证：
```bash
wc -l data/augmented/augmented_data.jsonl    # 期望: 243 (= 100+13+30+50+50)
ls data/augmented/typographic_images/ | wc -l # 期望: 30
ls -lh data/augmented/ip_embeddings.pkl      # 期望: 非空
```

#### 步骤 3.2：模型训练

```bash
source venv/bin/activate
caffeinate -i python3 scripts/run_training.py
```

> `caffeinate -i` 防止 macOS 训练中休眠。

**执行内容**：

**DistilBERT 文本分类器**：
- 模型：`distilbert-base-uncased`，66M 参数
- 任务：文本二分类（harmful / unharmful）
- 训练：smoke_test 1 epoch / full_run 5 epochs，AdamW，lr=2e-5，batch_size=32
- 设备：Apple M4 Max MPS
- 选型理由：级联第 1 层需 <10ms 延迟，DistilBERT 是 BERT 的知识蒸馏版（参数减 40%，速度快 60%，保留 97% 理解力）

**CLIP 多模态分类器**：
- 架构：CLIP ViT-B-32（冻结 backbone）+ 线性分类头（512→256→2）
- 任务：图文混合内容二分类
- 为什么冻结 CLIP：CLIP 已有强图文对齐能力，微调收益不大且可能灾难性遗忘
- Precision-Recall 取舍：按 TikTok 审核标准优先 Recall（宁可多拦不可漏判），因为漏掉有害内容 = 真实伤害，误拦 = 可人工恢复

验证：
```bash
ls results/models/text_classifier/           # 期望: 有 checkpoint 文件
ls results/models/multimodal_classifier/     # 期望: 有 checkpoint 文件
```

#### 步骤 3.3：Benchmark 评估

```bash
source venv/bin/activate
python3 scripts/run_evaluation.py
```

**执行内容**：在 4 个标准 Benchmark 上评估模型。

| Benchmark | 样本量 | 测什么 | 目标值 | 目标口径 |
|-----------|--------|--------|--------|---------|
| WildGuardTest | 400 | 综合安全能力 | AUC ≥ 0.85 | — |
| HarmBench | ~500 | 有害内容检测 | Recall ≥ 0.90 | 分子=正确检出的有害样本数，分母=全部有害样本数 |
| XSTest | 450 | Over-refusal | 误拦率 < 10% | 分子=被误拦的安全样本数，分母=全部安全样本数 |
| MM-SafetyBench | 5,040 | 多模态攻击 | ASR < 20% | 分子=攻击成功数，分母=全部攻击数 |

验证：
```bash
ls results/evaluation/                       # 期望: 有评估结果文件
```

#### 步骤 3.4：消融实验

```bash
source venv/bin/activate
caffeinate -i python3 scripts/run_ablation.py
```

> smoke_test 下约 10 分钟（训练 6 个模型），full_run 下约 3 小时。

**执行内容**：控制变量法——每次只去掉一组数据，其他条件不变，重新训练模型，看性能变化。

| 实验 | 去掉什么 | 验证什么 |
|------|---------|---------|
| Full | 无 | 对照基准 |
| -Safety | WildGuardMix 数据 | 安全数据是否是核心 |
| -Contrastive | 对比样本 | 对比构造是否必要 |
| -Augmentation | 全部增强数据 | 增强整体价值 |
| -Copyright | 版权数据 | 版权类别是否需要专门数据 |
| -ToxiGen | ToxiGen 数据 | 隐式毒性的贡献 |

**消融实验的价值**：不做消融 = 不知道数据集里哪部分有用。最大的价值不是证明"数据好"，而是告诉业务方**下一轮数据采集应该优先投入什么**。

验证：
```bash
python3 -c "import json; d=json.load(open('results/ablation/ablation_results.json')); print(len(d), '组')"
# 期望: 6 组
```

#### 步骤 3.5：生成最终交付文件

```bash
source venv/bin/activate
python3 scripts/generate_report.py
```

**执行内容**：
- 合并清洗后数据 + 增强数据 → SFT 训练集
- 为有害样本生成安全拒绝 / 不安全顺从回复 → DPO 偏好对
- 按 80/20 分割 → 训练集 / 评估集
- 生成 `dataset_card.md` 数据集说明
- 生成 `results/final_summary.json` 全项目统计

验证：
```bash
wc -l data/final/safety_sft_mix.jsonl        # 期望: 6,772
wc -l data/final/safety_dpo_pairs.jsonl      # 期望: 4,225
wc -l data/final/safety_eval.jsonl           # 期望: 1,305
ls -lh data/final/                           # 期望: 3 个 JSONL + dataset_card.md
cat results/final_summary.json               # 期望: 含 data_pipeline + final_datasets 字段
```

#### 步骤 3.6：创建并执行 Notebook 05-10

```bash
source venv/bin/activate
for nb in 05_data_augmentation 06_text_classifier_training 07_multimodal_classification \
          08_benchmark_evaluation 09_ablation_study 10_dashboard_report; do
  jupyter nbconvert --to notebook --execute --inplace "notebooks/${nb}.ipynb"
done
```

**执行内容**：
- NB05：5 种增强策略的分布分析
- NB06：DistilBERT 训练曲线 + 错误分析（哪些样本分错了）
- NB07：CLIP 分类结果 + 版权检测评估
- NB08：4 个 Benchmark 的评估结果汇总
- NB09：6 组消融实验对比 + 雷达图
- NB10：全项目数据流 Dashboard

验证：每个 Notebook 无 traceback，共产出 23 张 PNG 图表到 `results/figures/`。

#### 阶段三验收

```
阶段三验收：
  [ ] data/augmented/augmented_data.jsonl — 243 行
  [ ] data/augmented/typographic_images/ — 30 张 PNG
  [ ] results/models/text_classifier/ — checkpoint 存在
  [ ] results/models/multimodal_classifier/ — checkpoint 存在
  [ ] results/ablation/ablation_results.json — 6 组结果
  [ ] data/final/safety_sft_mix.jsonl — 6,772 行
  [ ] data/final/safety_dpo_pairs.jsonl — 4,225 行
  [ ] data/final/safety_eval.jsonl — 1,305 行
  [ ] data/final/dataset_card.md — 非空
  [ ] results/final_summary.json — 含统计数据
  [ ] notebooks/05-10 — 全部无 traceback
  [ ] results/figures/ — 23 张 PNG
```

---

### 阶段四：文档 + Git 收尾

**目标**：完善文档，清理 Git，确保项目可交付。

#### 步骤 4.1：文档编写

| 文件 | 内容 | 验收 |
|------|------|------|
| `README.md` | 阅读指南 + 实验结论 + 运行流程 + 项目地图 | 前 50 行回答问题-方案-预期 |
| `docs/PLAN.md` | 本文档 | 自包含执行手册 |
| `docs/tech_review_prep.md` | 口述稿 + 18 个高频问题 + 数字速查表 | 覆盖数据/质量/系统设计/技术 4 类 |

#### 步骤 4.2：Git 清理

确认 `.gitignore` 排除以下内容（注意：`.gitignore` **不支持行内注释**，`*.bin # model` 中 `#` 会被当成文件名的一部分）：

```
venv/
data/
results/models/
*.bin
*.pt
*.pkl
__pycache__/
.ipynb_checkpoints/
```

验证：
```bash
git status                   # 无意外大文件
git ls-files --others --exclude-standard | head -20   # 检查未跟踪文件
```

#### 阶段四验收

```
阶段四验收：
  [ ] README.md — 含阅读指南、实验结论、运行流程
  [ ] docs/PLAN.md — 本文档存在且内容完整
  [ ] docs/tech_review_prep.md — 口述稿 + 高频问题
  [ ] .gitignore — 不含行内注释，排除 venv/data/models
  [ ] git status — 无 >10MB 的未跟踪文件
```

---

## 五、实验结果（smoke_test 模式）

> 本节是全部实验数据的**唯一出处**。其他文档引用时标注"详见 PLAN.md §五"。
>
> 注意：smoke_test 数据量小（5,417 训练 / 1,356 测试），高分数部分原因是数据分布简单。full_run 下会有更真实的指标。

### 5.1 数据流统计

| 阶段 | 数量 | 说明 |
|------|------|------|
| 原始（统一后） | 9,480 文本 + 500 多模态 | 8 个数据集 |
| 文本清洗后 | 6,529 | 保留率 68.9%（分子=6,529 清洗后，分母=9,480 清洗前） |
| 多模态清洗后 | 20 | 合成数据模板少致去重率高（分子=20 清洗后，分母=500 清洗前） |
| 增强数据 | 243 | 对比 100 + 类别 13 + 印刷术 30 + 版权 50 + 改写 50 |
| 最终 SFT | 6,772 | = 清洗后 6,529 + 增强 243 |
| 最终 DPO | 4,225 | 有害样本配安全拒绝 / 不安全顺从回复 |
| 最终 Eval | 1,305 | 清洗数据的 20%（分子=1,305 评估集，分母=6,529 清洗后） |

去重是主要过滤源：去掉 2,942 条重复（分子=去重丢弃数，分母=去重步骤输入数），说明 8 个数据集之间有大量交叉。

### 5.2 DistilBERT 文本分类器

训练配置：1 epoch, AdamW, lr=2e-5, batch_size=32, Apple M4 Max MPS

| 指标 | 值 | 含义 |
|------|-----|------|
| AUC-ROC | 0.9962 | 区分有害/无害的综合能力 |
| Recall | 0.9650 | 96.5% 的有害内容被拦截（分子=正确拦截数，分母=全部有害数） |
| Precision | 0.9892 | 被拦截内容中 98.9% 确实有害（分子=真正有害数，分母=全部被拦截数） |
| F1 | 0.9770 | Precision 和 Recall 的调和平均 |
| Accuracy | 0.9761 | 整体正确率 |

### 5.3 CLIP 多模态分类器

架构：CLIP ViT-B-32（冻结）+ 线性分类头（512→256→2）

| 指标 | 值 | 含义 |
|------|-----|------|
| AUC-ROC | 0.9506 | 图文混合内容的分类能力 |
| Recall | 1.0000 | 所有有害内容被拦截 |
| Precision | 0.8966 | 较高误拦率（分子=真正有害数，分母=全部被拦截数），需后续人工复核 |
| F1 | 0.9455 | — |
| Accuracy | 0.9269 | — |

### 5.4 Benchmark 评估

| Benchmark | 样本数 | AUC | F1 | Recall | 状态 |
|-----------|--------|-----|-----|--------|------|
| WildGuardTest | 400 | 1.0000 | 1.0000 | 1.0000 | 超过目标（0.85），但 smoke_test 数据量小导致完美分数 |
| XSTest | 450 | — | 0.0000 | 0.0000 | 全为 safe 样本，Recall 指标不适用 |
| HarmBench | — | — | — | — | smoke_test 数据不足，需 full_run |
| MM-SafetyBench | — | — | — | — | 无多模态测试数据，需 full_run |

### 5.5 消融实验

| 实验 | AUC | F1 | Recall | 训练量 | 与 Full 的 AUC 差 |
|------|-----|-----|--------|--------|------------------|
| Full（基准） | 0.9959 | 0.9831 | 0.9837 | 5,417 | — |
| -Safety | 0.9948 | 0.9644 | 0.9445 | 5,393 | -0.0011 |
| -Contrastive | 0.9972 | 0.9814 | 0.9703 | 5,337 | +0.0013 |
| -Augmentation | 0.9942 | 0.9751 | 0.9762 | 5,383 | -0.0017 |
| -Copyright | 0.9932 | 0.9768 | 0.9647 | 5,377 | -0.0027 |
| -ToxiGen | 0.9946 | 0.9652 | 0.9828 | 3,842 | -0.0013 |

**结论排序**（按对模型影响从大到小）：

1. **Safety 数据最关键**：去掉后 Recall 从 98.4% 降到 94.5%（绝对差 -3.92 个百分点），说明 WildGuardMix 是性能基石
2. **Copyright 影响超预期**：去掉后 AUC 下降最多（绝对差 -0.0027），版权类别需要专门数据
3. **Augmentation 有正向作用**：F1 下降 0.80 个百分点（绝对差），增强数据不是噪声
4. **ToxiGen 贡献隐式毒性**：去掉 1,575 条后 AUC 下降 0.0013（绝对差），隐式 hate speech 有独特贡献
5. **Contrastive 在小数据下不显著**：smoke_test 仅 100 条对比数据，full_run 增到 500 条后预期显著

**消融实验指导的数据迭代优先级**：下一轮采集应优先扩充安全对话数据（WildGuardMix）和隐式毒性数据（ToxiGen）。

### 5.6 最终交付文件

| 文件 | 条数 | 大小 | 用途 | 格式 |
|------|------|------|------|------|
| `data/final/safety_sft_mix.jsonl` | 6,772 | 4.1MB | 安全对齐 SFT 训练 | JSONL |
| `data/final/safety_dpo_pairs.jsonl` | 4,225 | 1.4MB | DPO 偏好训练 | JSONL (prompt/chosen/rejected) |
| `data/final/safety_eval.jsonl` | 1,305 | 799KB | 模型评估 | JSONL |
| `data/final/dataset_card.md` | — | 1.5KB | 数据集说明文档 | Markdown |

---

## 六、full_run 规划

smoke_test 验证 Pipeline 正确性后，切换到 full_run 产出生产级结果。

### 6.1 与 smoke_test 的差异

| 维度 | smoke_test | full_run | 影响 |
|------|-----------|---------|------|
| 文本数据量 | 2,000 | 50,000 | 清洗后预期 ~35,000 条 |
| 合成增强 | 50 | 500 | 对比样本从 100 增到 ~500，预期 Over-refusal 改善显著 |
| 训练轮数 | 1 epoch | 5 epochs | 模型充分收敛 |
| 耗时 | ~30 分钟 | ~4-6 小时 | 建议用 tmux / screen |
| Benchmark | 2/4 有结果 | 4/4 预期有结果 | HarmBench + MM-SafetyBench 数据充足 |

### 6.2 执行步骤

```bash
# 1. 修改配置
sed -i '' 's/run_mode: "smoke_test"/run_mode: "full_run"/' configs/run_config.yaml

# 2. 在 tmux 中重新运行全流程
tmux new -s fullrun
source venv/bin/activate
caffeinate -i bash -c '
  python3 scripts/run_download.py && \
  python3 scripts/run_cleaning.py && \
  python3 scripts/run_augmentation.py && \
  python3 scripts/run_training.py && \
  python3 scripts/run_evaluation.py && \
  python3 scripts/run_ablation.py && \
  python3 scripts/generate_report.py
'

# 3. 重跑 Notebook
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

### 6.3 full_run 预期指标

| 指标 | smoke_test 实测 | full_run 预期 | 理由 |
|------|----------------|-------------|------|
| DistilBERT AUC | 0.9962 | 0.95-0.98 | 数据量大 → 分布更真实 → 不再"轻松"区分 |
| Contrastive 消融差异 | +0.0013（不显著） | 显著负值 | 500 条对比数据量足够 |
| HarmBench Recall | 数据不足 | ≥ 0.90 | 有足够测试样本 |
| MM-SafetyBench ASR | 无数据 | < 20% | 多模态测试数据充足 |

---

## 七、已知限制与改进 Action Items

| # | 限制 | 原因 | Action Item | 优先级 |
|---|------|------|------------|--------|
| 1 | 3 个 Gated 数据集用合成替代 | HF 权限 | `huggingface-cli login` → 在 HF 网站申请 WildGuardMix / WildJailbreak / HarmBench 权限 → 重跑 `run_download.py` | 高 |
| 2 | smoke_test 部分 Benchmark 不足 | 数据采样小 | 执行上述 §六 full_run 流程 | 高 |
| 3 | 版权检测准确率低 | 用文本 embedding 代替图片 embedding | 收集 20 个 IP 的真实图片 → 用 CLIP 图像编码器生成 embedding → 替换 `ip_embeddings.pkl` | 中 |
| 4 | 对比数据 smoke_test 下不显著 | 100 条太少 | full_run 自动增到 500 条，无需额外操作 | 中 |
| 5 | DPO 数据用模板生成 | 未接入 LLM API | 在 `configs/api_config.yaml` 配置 API key → 修改 `generate_report.py` 调用 Claude API 生成多样化拒绝/顺从回复 | 低 |
| 6 | 多模态清洗后仅 20 条 | 合成数据模板重复 | 获取真实图文安全数据 → 替换 `data/raw/` 中的合成多模态数据 | 中 |

---

## 八、14 类安全风险定义

| 编号 | 类别 | 英文标识 | 示例 |
|------|-----|---------|------|
| 01 | 非法活动 | illegal_activity | 制毒、黑客教程 |
| 02 | 仇恨言论 | hate_speech | 种族歧视、群体攻击 |
| 03 | 恶意软件 | malware | 病毒代码、钓鱼 |
| 04 | 身体伤害 | physical_harm | 暴力指导、自我伤害 |
| 05 | 经济犯罪 | economic_crime | 洗钱、逃税教程 |
| 06 | 欺诈行为 | fraud | 诈骗话术、伪造 |
| 07 | 色情内容 | pornography | 未成年保护 |
| 08 | 政治游说 | political_lobbying | 操纵选举 |
| 09 | 隐私侵犯 | privacy_violation | 人肉搜索、泄漏个人信息 |
| 10 | 法律建议 | legal_opinion | 未授权法律咨询 |
| 11 | 财务建议 | financial_advice | 未授权投资建议 |
| 12 | 健康咨询 | health_consultation | 未授权医疗建议 |
| 13 | 政府决策 | gov_decision | 冒充政府决策 |
| **14** | **版权侵权** | **copyright** | **TikTok 特有：IP 角色、品牌 logo** |

---

## 九、踩坑速查

执行过程中遇到的典型问题及解决方案。

### 数据层

| # | 报错 / 现象 | 原因 | 解决方案 |
|---|------------|------|---------|
| 1 | `Dataset 'allenai/wildguardmix' is a gated dataset` | WildGuardMix / WildJailbreak / HarmBench 需要 HF 申请权限 | 代码自动 fallback 到合成替代数据；或提前 `huggingface-cli login` |
| 2 | SafeBench category 字段是完整的种子问题 | 数据集设计将种子问题作为类别标识 | `_infer_safebench_category()` 关键词推断 |
| 3 | XSTest 出现 2,700 条且全标为 harmful | 6 个 JSONL 文件重复（各模型评测结果）+ type 字段误判 | 只读 `prompts.jsonl`，去重后 450 条，默认标 safe |

### 代码层

| # | 报错 / 现象 | 原因 | 解决方案 |
|---|------------|------|---------|
| 4 | `SyntaxError: f-string: unmatched '['` | Python 3.9 不支持 f-string 中嵌套同类引号 | 提取表达式为变量再放入 f-string |
| 5 | `NotJSONError` / `Expecting ',' delimiter` | Notebook JSON 中中文引号 "" 破坏结构 | 用 Python 脚本 + `json.dump()` 生成 Notebook，不手写 JSON |
| 6 | Notebook 所有代码行拼成一行 | `source.split("\n")` 后每行末尾缺少 `\n` | `_split_source()` 函数给每行末尾添加 `\n`（最后一行除外） |

### 环境层

| # | 报错 / 现象 | 原因 | 解决方案 |
|---|------------|------|---------|
| 7 | `python` 命令不存在 | macOS 上 `python` 可能未链接 | 统一使用 `python3` |
| 8 | 训练中途 Mac 休眠 | 默认电源管理 | 命令前加 `caffeinate -i` |
| 9 | `jupyter` 命令找不到 | 未激活虚拟环境 | `source venv/bin/activate`（注意不是 `.venv`） |
| 10 | `.gitignore` 未生效 | 使用了行内注释 `*.bin # model` | `.gitignore` 不支持行内注释，`#` 会被当成文件名 |

---

## 十、参考论文和数据集

| 名称 | 用途 | HF 路径 |
|------|------|--------|
| WildGuardMix | 核心安全训练+测试 | allenai/wildguardmix |
| WildJailbreak | 对比构造 | allenai/wildjailbreak |
| ToxiGen | 隐式毒性 | skg/toxigen-data |
| MM-SafetyBench | 多模态攻击 | isXinLiu/MM-SafetyBench |
| XSTest | Over-refusal 测试 | paul-rottger/xstest |
| HarmBench | 有害内容标准测试 | centerforaisafety/HarmBench |
| SafeBench | 安全评估 | — |
| LLaVA-Instruct | 正常图文对照 | liuhaotian/LLaVA-Instruct-150K |
| FigStep | 印刷术攻击方法 | 本项目 `typographic_attack.py` 复现 |
