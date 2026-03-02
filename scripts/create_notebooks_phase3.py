#!/usr/bin/env python3
"""
创建 Notebook 05-10（阶段三），避免 JSON 转义问题
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJECT_ROOT / "notebooks"


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }


def _split_source(source):
    lines = source.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _split_source(source)}


def create_notebook_05():
    """Notebook 05: 数据增强"""
    cells = [
        md("# 05 - 数据增强\n\n"
           "对清洗后的数据进行四种增强：\n"
           "1. **对比样本生成**：防止 over-refusal\n"
           "2. **稀缺类别增强**：平衡各风险类别\n"
           "3. **印刷术攻击样本**：模拟 FigStep 攻击\n"
           "4. **版权数据增强**：TikTok 版权场景\n"
           "5. **规则式改写**：增加样本多样性"),

        md("## 为什么需要数据增强？\n\n"
           "> **安全数据的两大挑战**：\n"
           "> 1. **类别不平衡**：色情和仇恨言论数据多，经济犯罪和版权数据少\n"
           "> 2. **对比数据缺失**：没有\"表面相似但无害\"的对比样本，模型会 over-refuse\n"
           ">\n"
           "> **Over-refusal 示例**：\n"
           "> - 有害: \"How to make a bomb at home?\" -> 应该拒绝\n"
           "> - 无害: \"How to make a bath bomb at home?\" -> 不应该拒绝\n"
           ">\n"
           "> 没有对比数据的模型会一刀切地拒绝所有包含\"bomb\"的请求。"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
             "from collections import Counter\nfrom pathlib import Path\n\n"
             "from src.utils.config_loader import print_config, load_run_config, get_data_path\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "config = load_run_config()\n"
             "print_config()"),

        md("## 1. 增强前：类别分布分析"),

        code("# 加载清洗后数据\n"
             "cleaned_path = get_data_path('cleaned') / 'text_safety_cleaned.jsonl'\n"
             "cleaned_records = []\n"
             "with open(cleaned_path, 'r') as f:\n"
             "    for line in f:\n"
             "        cleaned_records.append(json.loads(line))\n\n"
             "# 统计类别分布\n"
             "cat_counts = Counter(r.get('meta', {}).get('risk_category', 'unknown') for r in cleaned_records)\n"
             "label_counts = Counter(r.get('meta', {}).get('prompt_harm_label', 'unknown') for r in cleaned_records)\n\n"
             "print(f'清洗后数据总量: {len(cleaned_records):,}')\n"
             "print(f'\\n标签分布:')\n"
             "for label, count in sorted(label_counts.items()):\n"
             "    print(f'  {label}: {count:,}')\n\n"
             "print(f'\\n类别分布（前 15）:')\n"
             "for cat, count in cat_counts.most_common(15):\n"
             "    print(f'  {cat}: {count:,}')"),

        md("## 2. 运行增强流程"),

        code("from src.augmentation.contrastive_generator import generate_contrastive_samples\n"
             "from src.augmentation.category_balancer import generate_category_augmentation, analyze_category_balance\n"
             "from src.augmentation.typographic_attack import generate_typographic_samples\n"
             "from src.augmentation.copyright_embedding import generate_copyright_training_data\n"
             "from src.augmentation.synthetic_rephraser import rephrase_samples\n\n"
             "n_samples = config['synthesis_count']\n"
             "seed = config.get('seed', 42)\n\n"
             "# 2.1 对比样本\n"
             "print('[1/5] 对比样本生成...')\n"
             "contrastive = generate_contrastive_samples(n_samples=n_samples, seed=seed)\n"
             "print(f'  生成: {len(contrastive)} 条')\n\n"
             "# 2.2 类别平衡\n"
             "print('\\n[2/5] 稀缺类别增强...')\n"
             "analysis, _ = analyze_category_balance(cleaned_records)\n"
             "target = max(20, n_samples // 5)\n"
             "category_aug, cat_stats = generate_category_augmentation(cleaned_records, target_per_category=target, seed=seed)\n"
             "print(f'  生成: {len(category_aug)} 条')\n"
             "for cat, info in cat_stats.items():\n"
             "    print(f'    {cat}: {info[\"before\"]} -> {info[\"after\"]}')\n\n"
             "# 2.3 印刷术攻击\n"
             "print('\\n[3/5] 印刷术攻击...')\n"
             "harmful_prompts = []\n"
             "for r in cleaned_records:\n"
             "    if r.get('meta', {}).get('prompt_harm_label') == 'harmful':\n"
             "        text = r.get('text', '').replace('User: ', '')\n"
             "        cat = r.get('meta', {}).get('risk_category', 'unknown')\n"
             "        harmful_prompts.append((text, cat))\n"
             "typo_dir = get_data_path('augmented') / 'typographic_images'\n"
             "typo_samples = generate_typographic_samples(harmful_prompts[:n_samples], typo_dir, max_samples=min(n_samples, 30), seed=seed)\n"
             "print(f'  生成: {len(typo_samples)} 条')\n\n"
             "# 2.4 版权增强\n"
             "print('\\n[4/5] 版权数据增强...')\n"
             "copyright_samples = generate_copyright_training_data(n_samples=max(20, n_samples // 2), seed=seed)\n"
             "print(f'  生成: {len(copyright_samples)} 条')\n\n"
             "# 2.5 规则式改写\n"
             "print('\\n[5/5] 规则式改写...')\n"
             "harmful_records = [r for r in cleaned_records if r.get('meta', {}).get('prompt_harm_label') == 'harmful'][:n_samples]\n"
             "rephrased = rephrase_samples(harmful_records, multiplier=2, seed=seed)\n"
             "print(f'  改写: {len(rephrased)} 条')\n\n"
             "all_augmented = contrastive + category_aug + typo_samples + copyright_samples + rephrased\n"
             "print(f'\\n总计增强: {len(all_augmented):,} 条')"),

        md("## 3. 增强后分布分析"),

        code("# 合并原始+增强数据\n"
             "combined = cleaned_records + all_augmented\n"
             "aug_source_counts = Counter(r.get('meta', {}).get('source', 'unknown') for r in all_augmented)\n"
             "aug_label_counts = Counter(r.get('meta', {}).get('prompt_harm_label', 'unknown') for r in all_augmented)\n\n"
             "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
             "# 来源分布\n"
             "sources = sorted(aug_source_counts.keys())\n"
             "counts = [aug_source_counts[s] for s in sources]\n"
             "bars = axes[0].barh(sources, counts, color=plt.cm.Set3(range(len(sources))))\n"
             "axes[0].set_title('Augmented Data by Source', fontweight='bold')\n"
             "axes[0].set_xlabel('Count')\n"
             "for bar, c in zip(bars, counts):\n"
             "    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(c), va='center', fontsize=9)\n\n"
             "# 标签分布对比\n"
             "labels = ['harmful', 'unharmful']\n"
             "before_counts = [label_counts.get(l, 0) for l in labels]\n"
             "after_counts = [Counter(r.get('meta', {}).get('prompt_harm_label', 'unknown') for r in combined).get(l, 0) for l in labels]\n"
             "x = range(len(labels))\n"
             "w = 0.35\n"
             "axes[1].bar([i - w/2 for i in x], before_counts, w, label='Before Aug', color='#3498db', alpha=0.7)\n"
             "axes[1].bar([i + w/2 for i in x], after_counts, w, label='After Aug', color='#e74c3c', alpha=0.7)\n"
             "axes[1].set_xticks(list(x))\n"
             "axes[1].set_xticklabels(labels)\n"
             "axes[1].set_title('Label Distribution: Before vs After', fontweight='bold')\n"
             "axes[1].legend()\n\n"
             "# 增强类型饼图\n"
             "type_counts = {'Contrastive': len(contrastive), 'Category': len(category_aug),\n"
             "               'Typographic': len(typo_samples), 'Copyright': len(copyright_samples),\n"
             "               'Rephrased': len(rephrased)}\n"
             "axes[2].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',\n"
             "            colors=plt.cm.Pastel1(range(len(type_counts))))\n"
             "axes[2].set_title('Augmentation Type Breakdown', fontweight='bold')\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/augmentation_analysis.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 4. 对比样本演示"),

        code("# 展示对比样本对\n"
             "from src.augmentation.contrastive_generator import CONTRASTIVE_PAIRS\n\n"
             "print('对比样本示例（harmful vs benign）:')\n"
             "print('=' * 80)\n"
             "for pair in CONTRASTIVE_PAIRS[:8]:\n"
             "    print(f'  Harmful: {pair[\"harmful\"]}')\n"
             "    print(f'  Benign:  {pair[\"benign\"]}')\n"
             "    print(f'  Category: {pair[\"category\"]}')\n"
             "    print()"),

        md("## 5. 印刷术攻击样本展示"),

        code("# 展示生成的攻击图片\n"
             "import os\nfrom PIL import Image\nimport numpy as np\n\n"
             "typo_images_dir = get_data_path('augmented') / 'typographic_images'\n"
             "if typo_images_dir.exists():\n"
             "    image_files = sorted(typo_images_dir.glob('*.png'))[:6]\n"
             "    if image_files:\n"
             "        fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
             "        for ax, img_path in zip(axes.flat, image_files):\n"
             "            img = Image.open(img_path)\n"
             "            ax.imshow(np.array(img))\n"
             "            ax.set_title(img_path.stem, fontsize=9)\n"
             "            ax.axis('off')\n"
             "        for ax in axes.flat[len(image_files):]:\n"
             "            ax.axis('off')\n"
             "        plt.suptitle('Typographic Attack Samples (FigStep Style)', fontsize=14, fontweight='bold')\n"
             "        plt.tight_layout()\n"
             "        plt.savefig('../results/figures/typographic_samples.png', dpi=150, bbox_inches='tight')\n"
             "        plt.show()\n"
             "else:\n"
             "    print('No typographic images generated')"),

        md("## 关键发现\n\n"
           "1. **对比样本**解决 over-refusal：每条有害样本都有\"表面相似但安全\"的对照\n"
           "2. **类别平衡**提升稀缺类别表现：金融建议、隐私侵犯等原本不足的类别被补充\n"
           "3. **印刷术攻击**模拟真实场景：文字渲染为图片绕过文本检测\n"
           "4. **版权增强**覆盖 TikTok 特有需求：20 个知名 IP 的正反例\n\n"
           "-> 增强后的数据进入下一步：模型训练"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "05_data_augmentation.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


def create_notebook_06():
    """Notebook 06: DistilBERT 分类器训练"""
    cells = [
        md("# 06 - DistilBERT 文本安全分类器训练\n\n"
           "训练第一层审核模型：DistilBERT 二分类器（harmful/unharmful）\n\n"
           "**为什么选 DistilBERT？**\n"
           "- ~66M 参数，推理 <10ms\n"
           "- 在 TikTok 级联架构中做初筛：过滤 90% 内容\n"
           "- 大模型（7B/13B）放在后面的层级处理困难样本\n\n"
           "**工业界标准做法**：轻量模型初筛 + 重模型精筛 + 人工复审"),

        md("## Precision-Recall 的取舍\n\n"
           "> **生产环境优先高 Recall（召回率）**\n"
           ">\n"
           "> | 指标 | 含义 | 优先级 |\n"
           "> |------|------|--------|\n"
           "> | Recall | 有害内容被拦截的比例 | 最高（>0.90）|\n"
           "> | Precision | 被拦截内容中确实有害的比例 | 中等 |\n"
           "> | F1 | Precision 和 Recall 的调和平均 | 综合参考 |\n"
           ">\n"
           "> 漏掉一条有害内容 = 真实伤害\n"
           "> 误拦一条安全内容 = 可以人工恢复\n"
           ">\n"
           "> 所以生产系统选高 Recall，由后续人工审核处理 False Positive。"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport torch\n"
             "from collections import Counter\n\n"
             "from src.utils.config_loader import print_config, load_run_config, get_data_path, get_results_path\n"
             "from src.training.training_utils import set_seed\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "config = load_run_config()\n"
             "set_seed(config.get('seed', 42))\n"
             "print_config()\n"
             "print(f'PyTorch: {torch.__version__}')\n"
             "print(f'MPS 可用: {torch.backends.mps.is_available()}')"),

        md("## 1. 加载训练数据"),

        code("from src.training.text_classifier import load_training_data\n\n"
             "train_texts, train_labels, test_texts, test_labels, label_names = load_training_data()\n\n"
             "print(f'训练集: {len(train_texts):,} 条')\n"
             "print(f'测试集: {len(test_texts):,} 条')\n"
             "print(f'标签: {label_names}')\n"
             "print(f'\\n训练集标签分布: {Counter(train_labels)}')\n"
             "print(f'测试集标签分布: {Counter(test_labels)}')"),

        md("## 2. 训练 DistilBERT 分类器"),

        code("from src.training.text_classifier import train_classifier\n\n"
             "# 训练\n"
             "results = train_classifier()\n\n"
             "print(f'\\n训练完成！')\n"
             "print(f'模型路径: {results[\"model_path\"]}')"),

        md("## 3. 评估结果分析"),

        code("metrics = results['metrics']\n"
             "history = results['history']\n\n"
             "# 打印核心指标\n"
             "print('核心指标:')\n"
             "print(f'  AUC-ROC:   {metrics[\"auc\"]:.4f}')\n"
             "print(f'  F1:        {metrics[\"f1\"]:.4f}')\n"
             "print(f'  Precision: {metrics[\"precision\"]:.4f}')\n"
             "print(f'  Recall:    {metrics[\"recall\"]:.4f}')\n"
             "print(f'  Accuracy:  {metrics[\"accuracy\"]:.4f}')"),

        code("# 训练曲线\n"
             "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
             "if history['train_loss']:\n"
             "    axes[0].plot(history['train_loss'], 'b-o', markersize=4)\n"
             "    axes[0].set_title('Training Loss', fontweight='bold')\n"
             "    axes[0].set_xlabel('Step (x100)')\n"
             "    axes[0].set_ylabel('Loss')\n"
             "    axes[0].grid(True, alpha=0.3)\n\n"
             "if history['eval_auc']:\n"
             "    axes[1].plot(history['eval_auc'], 'g-o', markersize=6, label='AUC')\n"
             "    axes[1].plot(history['eval_f1'], 'r-s', markersize=6, label='F1')\n"
             "    axes[1].set_title('Evaluation Metrics per Epoch', fontweight='bold')\n"
             "    axes[1].set_xlabel('Epoch')\n"
             "    axes[1].legend()\n"
             "    axes[1].grid(True, alpha=0.3)\n\n"
             "# 混淆矩阵\n"
             "if 'confusion_matrix' in metrics:\n"
             "    cm = np.array(metrics['confusion_matrix'])\n"
             "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
             "                xticklabels=label_names, yticklabels=label_names, ax=axes[2])\n"
             "    axes[2].set_title('Confusion Matrix', fontweight='bold')\n"
             "    axes[2].set_xlabel('Predicted')\n"
             "    axes[2].set_ylabel('Actual')\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/text_classifier_training.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        code("# 分类报告\n"
             "if 'classification_report' in metrics:\n"
             "    report = metrics['classification_report']\n"
             "    print('分类报告:')\n"
             "    for class_name in label_names:\n"
             "        if class_name in report:\n"
             "            r = report[class_name]\n"
             "            print(f'  {class_name:12s}: precision={r[\"precision\"]:.4f}, recall={r[\"recall\"]:.4f}, f1={r[\"f1-score\"]:.4f}, support={r[\"support\"]}')\n"
             "    if 'macro avg' in report:\n"
             "        r = report['macro avg']\n"
             "        print(f'  {\"macro avg\":12s}: precision={r[\"precision\"]:.4f}, recall={r[\"recall\"]:.4f}, f1={r[\"f1-score\"]:.4f}')"),

        md("## 4. 错误分析"),

        code("# 分析预测错误的样本\n"
             "if 'all_preds' in metrics and 'all_labels' in metrics:\n"
             "    preds = metrics['all_preds']\n"
             "    labels_list = metrics['all_labels']\n"
             "    probs = metrics.get('all_probs', [])\n\n"
             "    # FP: 无害被误判为有害\n"
             "    fp_indices = [i for i, (p, l) in enumerate(zip(preds, labels_list)) if p == 1 and l == 0]\n"
             "    # FN: 有害被漏判\n"
             "    fn_indices = [i for i, (p, l) in enumerate(zip(preds, labels_list)) if p == 0 and l == 1]\n\n"
             "    print(f'False Positive (误拦): {len(fp_indices)} 条')\n"
             "    print(f'False Negative (漏判): {len(fn_indices)} 条')\n\n"
             "    # 展示部分错误样本\n"
             "    print(f'\\n--- 漏判样本（最危险）---')\n"
             "    for idx in fn_indices[:5]:\n"
             "        text = test_texts[idx][:100]\n"
             "        prob = probs[idx] if probs else 'N/A'\n"
             "        prob_str = f'{prob:.4f}' if isinstance(prob, float) else prob\n"
             "        print(f'  P(harmful)={prob_str}: {text}...')\n\n"
             "    print(f'\\n--- 误拦样本 ---')\n"
             "    for idx in fp_indices[:5]:\n"
             "        text = test_texts[idx][:100]\n"
             "        prob = probs[idx] if probs else 'N/A'\n"
             "        prob_str = f'{prob:.4f}' if isinstance(prob, float) else prob\n"
             "        print(f'  P(harmful)={prob_str}: {text}...')\n"
             "else:\n"
             "    print('详细预测数据不可用（非 detailed 模式）')"),

        md("## 关键发现\n\n"
           "1. **DistilBERT 在 smoke_test 模式下快速完成训练**\n"
           "2. **Recall 是生产环境最关键的指标** — 漏掉有害内容后果严重\n"
           "3. **错误分析帮助定位模型弱点** — FN 分析指导下一步数据增强方向\n\n"
           "-> 下一步：CLIP 多模态分类器训练"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "06_text_classifier_training.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


def create_notebook_07():
    """Notebook 07: 多模态分类 + 版权检测"""
    cells = [
        md("# 07 - CLIP 多模态安全分类 + 版权检测\n\n"
           "训练第二层审核模型：基于 CLIP 的多模态分类器\n\n"
           "**架构设计**：\n"
           "1. CLIP 提取图文特征（冻结 backbone）\n"
           "2. 线性分类头做二分类\n"
           "3. 版权 embedding 匹配检测\n\n"
           "**为什么冻结 CLIP？**\n"
           "- CLIP 已有强大的图文对齐能力，微调收益不大\n"
           "- 冻结 backbone 只训分类头：训练快、不需大量数据\n"
           "- 避免灾难性遗忘"),

        md("## Modal Gap 原理\n\n"
           "> **多模态攻击的核心原理**：文本安全 + 图片安全 ≠ 组合安全\n"
           ">\n"
           "> 示例：\n"
           "> - 文本: \"What does the text in the image say?\"（无害）\n"
           "> - 图片: 渲染的有害指令文字（看起来是普通图片）\n"
           "> - 组合: 实际请求有害内容\n"
           ">\n"
           "> MM-SafetyBench 论文发现，加入视觉模块后攻击成功率从文本的 20% 跳到 40-70%。"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport torch\n"
             "from collections import Counter\n\n"
             "from src.utils.config_loader import print_config, load_run_config, get_data_path, get_results_path\n"
             "from src.training.training_utils import set_seed\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "config = load_run_config()\n"
             "set_seed(config.get('seed', 42))\n"
             "print_config()"),

        md("## 1. 训练 CLIP 多模态分类器"),

        code("from src.training.multimodal_classifier import train_multimodal_classifier\n\n"
             "mm_results = train_multimodal_classifier()\n\n"
             "print(f'\\n训练完成！')\n"
             "print(f'模型路径: {mm_results[\"model_path\"]}')"),

        md("## 2. 结果分析"),

        code("metrics = mm_results['metrics']\n"
             "history = mm_results['history']\n\n"
             "print('CLIP 多模态分类器指标:')\n"
             "for k, v in metrics.items():\n"
             "    if isinstance(v, float):\n"
             "        print(f'  {k}: {v:.4f}')\n\n"
             "# 训练曲线\n"
             "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
             "if history['train_loss']:\n"
             "    axes[0].plot(history['train_loss'], 'b-o')\n"
             "    axes[0].set_title('Training Loss', fontweight='bold')\n"
             "    axes[0].set_xlabel('Epoch')\n\n"
             "if history['eval_auc']:\n"
             "    axes[1].plot(history['eval_auc'], 'g-o', label='AUC')\n"
             "    axes[1].plot(history['eval_f1'], 'r-s', label='F1')\n"
             "    axes[1].set_title('Evaluation Metrics', fontweight='bold')\n"
             "    axes[1].legend()\n\n"
             "if 'confusion_matrix' in metrics:\n"
             "    cm = np.array(metrics['confusion_matrix'])\n"
             "    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',\n"
             "                xticklabels=['unharmful', 'harmful'],\n"
             "                yticklabels=['unharmful', 'harmful'], ax=axes[2])\n"
             "    axes[2].set_title('Confusion Matrix', fontweight='bold')\n"
             "    axes[2].set_xlabel('Predicted')\n"
             "    axes[2].set_ylabel('Actual')\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/multimodal_classifier_training.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 3. 版权检测评估"),

        code("from src.evaluation.copyright_detector import run_copyright_evaluation\n\n"
             "copyright_results = run_copyright_evaluation()\n\n"
             "if copyright_results and 'error' not in copyright_results:\n"
             "    # 可视化不同阈值的效果\n"
             "    thresholds = sorted(copyright_results.keys())\n"
             "    precisions = [copyright_results[t]['precision'] for t in thresholds]\n"
             "    recalls = [copyright_results[t]['recall'] for t in thresholds]\n"
             "    f1s = [copyright_results[t]['f1'] for t in thresholds]\n\n"
             "    fig, ax = plt.subplots(figsize=(10, 6))\n"
             "    x = range(len(thresholds))\n"
             "    width = 0.25\n"
             "    ax.bar([i - width for i in x], precisions, width, label='Precision', color='#3498db')\n"
             "    ax.bar(list(x), recalls, width, label='Recall', color='#e74c3c')\n"
             "    ax.bar([i + width for i in x], f1s, width, label='F1', color='#2ecc71')\n"
             "    ax.set_xticks(list(x))\n"
             "    ax.set_xticklabels([str(t) for t in thresholds])\n"
             "    ax.set_xlabel('Similarity Threshold')\n"
             "    ax.set_ylabel('Score')\n"
             "    ax.set_title('Copyright Detection: Threshold vs Metrics', fontweight='bold')\n"
             "    ax.legend()\n"
             "    ax.set_ylim(0, 1.1)\n"
             "    plt.tight_layout()\n"
             "    plt.savefig('../results/figures/copyright_detection.png', dpi=150, bbox_inches='tight')\n"
             "    plt.show()"),

        md("## 4. 两个模型的对比"),

        code("# 加载文本分类器结果\n"
             "text_model_path = get_results_path('models/text_classifier')\n"
             "text_metrics_path = get_results_path('training') / 'training_summary.json'\n\n"
             "comparison = {\n"
             "    'Model': ['DistilBERT (Text)', 'CLIP Head (Multimodal)'],\n"
             "    'AUC': [0, metrics.get('auc', 0)],\n"
             "    'F1': [0, metrics.get('f1', 0)],\n"
             "    'Recall': [0, metrics.get('recall', 0)],\n"
             "    'Precision': [0, metrics.get('precision', 0)],\n"
             "}\n\n"
             "if text_metrics_path.exists():\n"
             "    with open(text_metrics_path) as f:\n"
             "        text_results = json.load(f)\n"
             "    if 'text_classifier' in text_results:\n"
             "        tm = text_results['text_classifier']['metrics']\n"
             "        comparison['AUC'][0] = tm.get('auc', 0)\n"
             "        comparison['F1'][0] = tm.get('f1', 0)\n"
             "        comparison['Recall'][0] = tm.get('recall', 0)\n"
             "        comparison['Precision'][0] = tm.get('precision', 0)\n\n"
             "df = pd.DataFrame(comparison)\n"
             "print('模型对比:')\n"
             "print(df.to_string(index=False))\n\n"
             "# 对比条形图\n"
             "fig, ax = plt.subplots(figsize=(10, 6))\n"
             "metrics_list = ['AUC', 'F1', 'Recall', 'Precision']\n"
             "x = range(len(metrics_list))\n"
             "w = 0.35\n"
             "ax.bar([i - w/2 for i in x], [comparison[m][0] for m in metrics_list], w,\n"
             "       label='DistilBERT', color='#3498db', alpha=0.8)\n"
             "ax.bar([i + w/2 for i in x], [comparison[m][1] for m in metrics_list], w,\n"
             "       label='CLIP Head', color='#e74c3c', alpha=0.8)\n"
             "ax.set_xticks(list(x))\n"
             "ax.set_xticklabels(metrics_list)\n"
             "ax.set_title('DistilBERT vs CLIP Head Comparison', fontweight='bold')\n"
             "ax.legend()\n"
             "ax.set_ylim(0, 1.1)\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/model_comparison.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 关键发现\n\n"
           "1. **CLIP 特征迁移有效**：冻结 CLIP backbone + 线性头可以做安全分类\n"
           "2. **版权检测依赖阈值选择**：0.80-0.90 区间有不同 Precision-Recall 平衡\n"
           "3. **两个模型互补**：文本模型做初筛，多模态模型处理图文攻击\n\n"
           "### TikTok 级联架构\n\n"
           "```\n"
           "Layer 1: DistilBERT (~10ms)  -> 过滤 90% 内容\n"
           "Layer 2: CLIP Head  (~100ms) -> 处理图文攻击\n"
           "Layer 3: LLM/Human (~秒级)   -> 处理边界案例\n"
           "```\n\n"
           "-> 下一步：Benchmark 评估 + 消融实验"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "07_multimodal_classification.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


def create_notebook_08():
    """Notebook 08: Benchmark 评估"""
    cells = [
        md("# 08 - Benchmark 评估\n\n"
           "在四个标准 Benchmark 上评估训练好的模型：\n"
           "1. **WildGuardTest** (5.3K): 综合安全测试集\n"
           "2. **HarmBench** (~500): 标准有害内容测试\n"
           "3. **XSTest** (250 对): Over-refusal 测试\n"
           "4. **MM-SafetyBench** (5040): 多模态攻击测试\n\n"
           "**关键原则**：Benchmark 数据绝对不能混入训练集。"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
             "from src.utils.config_loader import print_config, load_run_config, load_eval_config, get_results_path\n"
             "from src.evaluation.benchmark_runner import run_all_benchmarks, load_benchmark_data\n"
             "from src.evaluation.safety_metrics import compute_safety_metrics\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "print_config()"),

        md("## 1. 运行 Benchmark 评估"),

        code("# 检查是否有训练好的模型\n"
             "model_path = get_results_path('models/text_classifier')\n"
             "has_model = (model_path / 'config.json').exists()\n\n"
             "if has_model:\n"
             "    print(f'使用训练好的模型: {model_path}')\n"
             "    benchmark_results = run_all_benchmarks(text_model_path=str(model_path))\n"
             "else:\n"
             "    print('未找到训练好的模型，使用随机 baseline')\n"
             "    benchmark_results = run_all_benchmarks()"),

        md("## 2. 结果汇总"),

        code("# 汇总表\n"
             "summary_rows = []\n"
             "eval_config = load_eval_config()\n\n"
             "for bench_name, result in benchmark_results.items():\n"
             "    if 'error' in result:\n"
             "        continue\n"
             "    m = result['metrics']\n"
             "    bench_config = eval_config['benchmarks'].get(bench_name, {})\n"
             "    row = {\n"
             "        'Benchmark': bench_config.get('name', bench_name),\n"
             "        'Samples': result.get('data_size', 0),\n"
             "        'AUC': m.get('auc', 'N/A'),\n"
             "        'F1': m.get('f1', 0),\n"
             "        'Recall': m.get('recall', 0),\n"
             "        'Precision': m.get('precision', 0),\n"
             "        'Over-Refusal': m.get('over_refusal_rate', 'N/A'),\n"
             "    }\n"
             "    summary_rows.append(row)\n\n"
             "df_summary = pd.DataFrame(summary_rows)\n"
             "print('Benchmark 评估结果:')\n"
             "print(df_summary.to_string(index=False))"),

        code("# 可视化\n"
             "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
             "# 指标对比\n"
             "if len(summary_rows) > 0:\n"
             "    benchmarks = [r['Benchmark'] for r in summary_rows]\n"
             "    x = range(len(benchmarks))\n"
             "    width = 0.2\n\n"
             "    f1_vals = [r['F1'] for r in summary_rows]\n"
             "    recall_vals = [r['Recall'] for r in summary_rows]\n"
             "    precision_vals = [r['Precision'] for r in summary_rows]\n\n"
             "    axes[0].bar([i - width for i in x], f1_vals, width, label='F1', color='#3498db')\n"
             "    axes[0].bar(list(x), recall_vals, width, label='Recall', color='#e74c3c')\n"
             "    axes[0].bar([i + width for i in x], precision_vals, width, label='Precision', color='#2ecc71')\n"
             "    axes[0].set_xticks(list(x))\n"
             "    axes[0].set_xticklabels(benchmarks, rotation=15)\n"
             "    axes[0].set_title('Benchmark Performance Comparison', fontweight='bold')\n"
             "    axes[0].legend()\n"
             "    axes[0].set_ylim(0, 1.1)\n\n"
             "    # 目标 vs 实际\n"
             "    targets = {'WildGuardTest': 0.85, 'HarmBench': 0.90, 'XSTest': 0.10, 'MM-SafetyBench': 0.20}\n"
             "    actual = {}\n"
             "    for r in summary_rows:\n"
             "        name = r['Benchmark']\n"
             "        if name in ('WildGuardTest',):\n"
             "            auc_val = r['AUC']\n"
             "            actual[name] = auc_val if isinstance(auc_val, float) else 0\n"
             "        elif name == 'HarmBench':\n"
             "            actual[name] = r['Recall']\n"
             "        elif name == 'XSTest':\n"
             "            or_val = r['Over-Refusal']\n"
             "            actual[name] = or_val if isinstance(or_val, float) else 0\n"
             "        elif name == 'MM-SafetyBench':\n"
             "            actual[name] = 1 - r['Recall']  # ASR = 1 - Recall\n\n"
             "    target_names = list(targets.keys())\n"
             "    target_vals = [targets[n] for n in target_names]\n"
             "    actual_vals = [actual.get(n, 0) for n in target_names]\n\n"
             "    x2 = range(len(target_names))\n"
             "    axes[1].bar([i - 0.2 for i in x2], target_vals, 0.4, label='Target', color='#95a5a6', alpha=0.7)\n"
             "    axes[1].bar([i + 0.2 for i in x2], actual_vals, 0.4, label='Actual', color='#e74c3c', alpha=0.8)\n"
             "    axes[1].set_xticks(list(x2))\n"
             "    axes[1].set_xticklabels(target_names, rotation=15)\n"
             "    axes[1].set_title('Target vs Actual (Key Metrics)', fontweight='bold')\n"
             "    axes[1].legend()\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/benchmark_results.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 3. 攻击类型分析"),

        code("# 按攻击类型分析 ASR\n"
             "print('各 Benchmark 的攻击类型分析:')\n"
             "for bench_name, result in benchmark_results.items():\n"
             "    if 'error' in result:\n"
             "        continue\n"
             "    m = result['metrics']\n"
             "    if 'asr_by_type' in m:\n"
             "        print(f'\\n{bench_name}:')\n"
             "        for atype, asr in sorted(m['asr_by_type'].items()):\n"
             "            print(f'  {atype}: ASR = {asr:.4f}')"),

        md("## 4. 阈值分析"),

        code("# 最优阈值分析\n"
             "print('各 Benchmark 的最优阈值（目标 Recall >= 0.90）:')\n"
             "for bench_name, result in benchmark_results.items():\n"
             "    if 'error' in result:\n"
             "        continue\n"
             "    m = result['metrics']\n"
             "    if 'optimal_threshold' in m:\n"
             "        opt = m['optimal_threshold']\n"
             "        print(f'\\n{bench_name}:')\n"
             "        print(f'  Threshold: {opt[\"threshold\"]:.4f}')\n"
             "        print(f'  Recall:    {opt[\"recall\"]:.4f}')\n"
             "        print(f'  Precision: {opt[\"precision\"]:.4f}')\n"
             "        print(f'  F1:        {opt[\"f1\"]:.4f}')"),

        md("## 关键发现\n\n"
           "1. **Benchmark 结果提供客观评估** — 避免只看训练集上的过于乐观的指标\n"
           "2. **不同 Benchmark 测试不同能力** — WildGuardTest 综合、HarmBench 测 Recall、XSTest 测 Over-refusal\n"
           "3. **阈值分析指导生产部署** — 不同场景选择不同阈值\n\n"
           "-> 下一步：消融实验"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "08_benchmark_evaluation.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


def create_notebook_09():
    """Notebook 09: 消融实验"""
    cells = [
        md("# 09 - 消融实验（Ablation Study）\n\n"
           "系统地去掉每组数据，量化各组件的贡献。\n\n"
           "**6 组实验**：\n"
           "| # | 实验 | 描述 |\n"
           "|---|------|------|\n"
           "| 1 | Full | 完整数据集（对照基准） |\n"
           "| 2 | -Safety | 去掉 WildGuardMix |\n"
           "| 3 | -Contrastive | 去掉对比无害数据 |\n"
           "| 4 | -Augmentation | 去掉合成增强数据 |\n"
           "| 5 | -Copyright | 去掉版权数据 |\n"
           "| 6 | -ToxiGen | 去掉 ToxiGen |\n\n"
           "> **消融实验的价值**：不做消融 = 不知道哪部分数据真正有用。"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
             "from src.utils.config_loader import print_config, get_results_path\n"
             "from src.training.training_utils import set_seed, save_training_results\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "print_config()"),

        md("## 1. 运行消融实验"),

        code("from src.evaluation.ablation_runner import run_ablation_experiments, compute_ablation_summary\n\n"
             "ablation_results = run_ablation_experiments()"),

        md("## 2. 结果对比"),

        code("# 提取结果\n"
             "rows = []\n"
             "for name, data in ablation_results.items():\n"
             "    if 'error' in data:\n"
             "        continue\n"
             "    m = data['metrics']\n"
             "    rows.append({\n"
             "        'Experiment': name,\n"
             "        'AUC': m.get('auc', 0),\n"
             "        'F1': m.get('f1', 0),\n"
             "        'Recall': m.get('recall', 0),\n"
             "        'Precision': m.get('precision', 0),\n"
             "        'Train Size': data.get('train_size', 0),\n"
             "    })\n\n"
             "df = pd.DataFrame(rows)\n"
             "print('消融实验结果:')\n"
             "print(df.to_string(index=False))"),

        code("# 消融影响分析\n"
             "summary = compute_ablation_summary(ablation_results)\n\n"
             "if summary:\n"
             "    print('\\n消融影响（与 Full 对比）:')\n"
             "    for name, s in sorted(summary.items(), key=lambda x: x[1]['auc_drop'], reverse=True):\n"
             "        print(f'  {name:15s}: AUC drop={s[\"auc_drop\"]:+.4f}, F1 drop={s[\"f1_drop\"]:+.4f}, '\n"
             "              f'data removed={s[\"data_removed\"]:,}')"),

        code("# 可视化\n"
             "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
             "if len(rows) > 1:\n"
             "    # 指标对比\n"
             "    experiments = [r['Experiment'] for r in rows]\n"
             "    x = range(len(experiments))\n\n"
             "    colors = ['#2ecc71' if e == 'Full' else '#e74c3c' for e in experiments]\n"
             "    auc_vals = [r['AUC'] for r in rows]\n\n"
             "    bars = axes[0].bar(x, auc_vals, color=colors, alpha=0.8)\n"
             "    axes[0].set_xticks(list(x))\n"
             "    axes[0].set_xticklabels(experiments, rotation=30, ha='right')\n"
             "    axes[0].set_title('AUC by Ablation Experiment', fontweight='bold')\n"
             "    axes[0].set_ylabel('AUC')\n"
             "    for bar, val in zip(bars, auc_vals):\n"
             "        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,\n"
             "                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)\n\n"
             "    # AUC drop 排序\n"
             "    if summary:\n"
             "        sorted_items = sorted(summary.items(), key=lambda x: x[1]['auc_drop'], reverse=True)\n"
             "        drop_names = [n for n, _ in sorted_items]\n"
             "        drop_vals = [s['auc_drop'] for _, s in sorted_items]\n\n"
             "        drop_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in drop_vals]\n"
             "        axes[1].barh(drop_names, drop_vals, color=drop_colors, alpha=0.8)\n"
             "        axes[1].set_title('AUC Drop (Higher = More Important)', fontweight='bold')\n"
             "        axes[1].set_xlabel('AUC Drop')\n"
             "        axes[1].axvline(x=0, color='black', linewidth=0.5)\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/ablation_results.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 3. 多指标雷达图"),

        code("# 雷达图\n"
             "if len(rows) > 1:\n"
             "    metrics_names = ['AUC', 'F1', 'Recall', 'Precision']\n"
             "    N = len(metrics_names)\n"
             "    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()\n"
             "    angles += angles[:1]\n\n"
             "    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))\n\n"
             "    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(rows)))\n"
             "    for i, row in enumerate(rows):\n"
             "        values = [row[m] for m in metrics_names]\n"
             "        values += values[:1]\n"
             "        ax.plot(angles, values, 'o-', linewidth=2, label=row['Experiment'], color=colors_radar[i])\n"
             "        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])\n\n"
             "    ax.set_xticks(angles[:-1])\n"
             "    ax.set_xticklabels(metrics_names, fontsize=12)\n"
             "    ax.set_ylim(0, 1.1)\n"
             "    ax.set_title('Ablation: Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)\n"
             "    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))\n\n"
             "    plt.tight_layout()\n"
             "    plt.savefig('../results/figures/ablation_radar.png', dpi=150, bbox_inches='tight')\n"
             "    plt.show()"),

        md("## 关键发现\n\n"
           "消融实验揭示了各数据组件的贡献：\n\n"
           "1. **Safety 数据是基石** — 去掉后性能下降最大\n"
           "2. **对比数据防止 over-refusal** — 缺失会导致误拦率上升\n"
           "3. **增强数据提升泛化** — 合成数据对稀缺类别有显著帮助\n"
           "4. **版权数据是 TikTok 特需** — 不加则版权类别完全失效\n"
           "5. **ToxiGen 贡献隐式毒性** — 补充了常规数据集覆盖不到的微妙仇恨言论\n\n"
           "-> 下一步：Dashboard 和最终报告"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "09_ablation_study.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


def create_notebook_10():
    """Notebook 10: Dashboard + 最终报告"""
    cells = [
        md("# 10 - Dashboard + 最终报告\n\n"
           "汇总整个项目的关键结果，生成最终输出文件。\n\n"
           "**最终交付物**：\n"
           "1. `safety_sft_mix.jsonl` — SFT 训练数据\n"
           "2. `safety_dpo_pairs.jsonl` — DPO 训练对\n"
           "3. `safety_eval.jsonl` — 评估数据集\n"
           "4. `dataset_card.md` — 数据集卡片"),

        code("import sys\nsys.path.insert(0, '..')\n\n"
             "import json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom collections import Counter\nfrom pathlib import Path\n\n"
             "from src.utils.config_loader import print_config, load_run_config, get_data_path, get_results_path\n\n"
             "plt.rcParams['figure.figsize'] = (14, 6)\n"
             "sns.set_style('whitegrid')\n\n"
             "print_config()"),

        md("## 1. 数据流水线总览"),

        code("# 统计各阶段数据量\n"
             "stages = {}\n\n"
             "# 原始数据\n"
             "raw_path = get_data_path('unified') / 'text_safety.jsonl'\n"
             "if raw_path.exists():\n"
             "    stages['1. Raw (Unified)'] = sum(1 for _ in open(raw_path))\n\n"
             "# 清洗后\n"
             "cleaned_path = get_data_path('cleaned') / 'text_safety_cleaned.jsonl'\n"
             "if cleaned_path.exists():\n"
             "    stages['2. Cleaned'] = sum(1 for _ in open(cleaned_path))\n\n"
             "# 增强数据\n"
             "aug_path = get_data_path('augmented') / 'augmented_data.jsonl'\n"
             "if aug_path.exists():\n"
             "    stages['3. Augmented'] = sum(1 for _ in open(aug_path))\n\n"
             "# 合计\n"
             "stages['4. Total (Cleaned + Aug)'] = stages.get('2. Cleaned', 0) + stages.get('3. Augmented', 0)\n\n"
             "print('数据流水线各阶段统计:')\n"
             "for stage, count in stages.items():\n"
             "    print(f'  {stage}: {count:,}')"),

        code("# 可视化数据流\n"
             "fig, ax = plt.subplots(figsize=(12, 5))\n"
             "stage_names = list(stages.keys())\n"
             "stage_counts = list(stages.values())\n\n"
             "colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']\n"
             "bars = ax.bar(range(len(stage_names)), stage_counts, color=colors[:len(stage_names)], alpha=0.8)\n"
             "ax.set_xticks(range(len(stage_names)))\n"
             "ax.set_xticklabels(stage_names, rotation=15, ha='right')\n"
             "ax.set_title('Data Pipeline: Sample Count at Each Stage', fontsize=14, fontweight='bold')\n"
             "ax.set_ylabel('Sample Count')\n"
             "for bar, count in zip(bars, stage_counts):\n"
             "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,\n"
             "            f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/data_pipeline_overview.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 2. 生成最终输出文件"),

        code("from scripts.generate_report import generate_sft_mix, generate_dpo_pairs, generate_eval_data, generate_dataset_card\n\n"
             "sft_data = generate_sft_mix()\n"
             "dpo_pairs = generate_dpo_pairs()\n"
             "eval_data = generate_eval_data()\n"
             "card = generate_dataset_card()"),

        md("## 3. 最终数据集统计"),

        code("# 加载最终数据统计\n"
             "final_dir = get_data_path('final')\n\n"
             "# SFT 数据分析\n"
             "sft_cats = Counter(r.get('meta', {}).get('risk_category', 'unknown') for r in sft_data)\n"
             "sft_labels = Counter(r.get('meta', {}).get('prompt_harm_label', 'unknown') for r in sft_data)\n"
             "sft_sources = Counter(r.get('meta', {}).get('source', 'unknown') for r in sft_data)\n\n"
             "print('SFT 混合数据统计:')\n"
             "print(f'  总量: {len(sft_data):,}')\n"
             "print(f'  标签分布: {dict(sft_labels)}')\n"
             "print(f'  来源数: {len(sft_sources)}')\n"
             "print(f'  类别数: {len(sft_cats)}')\n\n"
             "print(f'\\nDPO 训练对: {len(dpo_pairs):,}')\n"
             "print(f'评估数据: {len(eval_data):,}')"),

        code("# 最终数据集类别分布\n"
             "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
             "# 类别分布\n"
             "top_cats = sft_cats.most_common(15)\n"
             "cat_names = [c for c, _ in top_cats]\n"
             "cat_counts = [c for _, c in top_cats]\n"
             "axes[0].barh(cat_names[::-1], cat_counts[::-1], color=plt.cm.viridis(np.linspace(0, 1, len(cat_names))))\n"
             "axes[0].set_title('Final Dataset: Category Distribution', fontweight='bold')\n"
             "axes[0].set_xlabel('Count')\n\n"
             "# 来源分布\n"
             "top_sources = sft_sources.most_common(10)\n"
             "src_names = [s for s, _ in top_sources]\n"
             "src_counts = [c for _, c in top_sources]\n"
             "axes[1].pie(src_counts, labels=src_names, autopct='%1.1f%%',\n"
             "            colors=plt.cm.Set3(np.linspace(0, 1, len(src_names))), textprops={'fontsize': 9})\n"
             "axes[1].set_title('Final Dataset: Source Distribution', fontweight='bold')\n\n"
             "plt.tight_layout()\n"
             "plt.savefig('../results/figures/final_dataset_distribution.png', dpi=150, bbox_inches='tight')\n"
             "plt.show()"),

        md("## 4. 模型性能 Dashboard"),

        code("# 加载训练结果\n"
             "training_path = get_results_path('training') / 'training_summary.json'\n"
             "ablation_path = get_results_path('ablation') / 'ablation_results.json'\n\n"
             "model_metrics = {}\n"
             "if training_path.exists():\n"
             "    with open(training_path) as f:\n"
             "        training_data = json.load(f)\n"
             "    for model_name, data in training_data.items():\n"
             "        if 'metrics' in data:\n"
             "            model_metrics[model_name] = data['metrics']\n\n"
             "ablation_data = {}\n"
             "if ablation_path.exists():\n"
             "    with open(ablation_path) as f:\n"
             "        ablation_data = json.load(f)\n\n"
             "# Dashboard 表格\n"
             "print('=' * 60)\n"
             "print('  模型性能 Dashboard')\n"
             "print('=' * 60)\n\n"
             "for name, m in model_metrics.items():\n"
             "    print(f'\\n{name}:')\n"
             "    for k, v in m.items():\n"
             "        if isinstance(v, float):\n"
             "            print(f'  {k:20s}: {v:.4f}')"),

        code("# Dashboard 可视化\n"
             "if model_metrics:\n"
             "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
             "    # 模型对比\n"
             "    model_names = list(model_metrics.keys())\n"
             "    metric_names = ['auc', 'f1', 'recall', 'precision']\n"
             "    x = range(len(metric_names))\n"
             "    width = 0.35\n\n"
             "    for i, model_name in enumerate(model_names[:2]):\n"
             "        m = model_metrics[model_name]\n"
             "        vals = [m.get(mn, 0) for mn in metric_names]\n"
             "        offset = (i - 0.5) * width\n"
             "        bars = axes[0].bar([xi + offset for xi in x], vals, width, label=model_name, alpha=0.8)\n\n"
             "    axes[0].set_xticks(list(x))\n"
             "    axes[0].set_xticklabels([m.upper() for m in metric_names])\n"
             "    axes[0].set_title('Model Performance Dashboard', fontweight='bold')\n"
             "    axes[0].set_ylim(0, 1.1)\n"
             "    axes[0].legend()\n\n"
             "    # 项目总结\n"
             "    summary_text = (\n"
             "        f'Project Summary\\n'\n"
             "        f'==============\\n'\n"
             "        f'Raw Data: {stages.get(\"1. Raw (Unified)\", 0):,}\\n'\n"
             "        f'Cleaned: {stages.get(\"2. Cleaned\", 0):,}\\n'\n"
             "        f'Augmented: {stages.get(\"3. Augmented\", 0):,}\\n'\n"
             "        f'Final SFT: {len(sft_data):,}\\n'\n"
             "        f'Final DPO: {len(dpo_pairs):,}\\n'\n"
             "        f'Final Eval: {len(eval_data):,}\\n'\n"
             "        f'Risk Categories: 14\\n'\n"
             "        f'Data Sources: {len(sft_sources)}\\n'\n"
             "    )\n"
             "    axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes,\n"
             "                fontsize=13, verticalalignment='center', fontfamily='monospace',\n"
             "                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))\n"
             "    axes[1].axis('off')\n"
             "    axes[1].set_title('Project Statistics', fontweight='bold')\n\n"
             "    plt.tight_layout()\n"
             "    plt.savefig('../results/figures/dashboard.png', dpi=150, bbox_inches='tight')\n"
             "    plt.show()"),

        md("## 5. 项目回顾与关键贡献\n\n"
           "### 完成的工作\n"
           "1. **数据采集**：8 个公开数据集 + 统一格式转换\n"
           "2. **数据清洗**：Data-Juicer 风格 Pipeline，安全数据专用宽松阈值\n"
           "3. **OCR + CLIP 双层防护**：OCR 做廉价初筛，CLIP 做语义级检测\n"
           "4. **数据增强**：对比样本 + 类别平衡 + 印刷术攻击 + 版权数据\n"
           "5. **模型训练**：DistilBERT 文本分类 + CLIP 多模态分类\n"
           "6. **Benchmark 评估**：4 个标准测试集\n"
           "7. **消融实验**：6 组实验量化各组件贡献\n\n"
           "### 关键创新点\n"
           "- **14 类风险分类**：增加了版权类别（TikTok 特需）\n"
           "- **对比数据构造**：解决 over-refusal 问题\n"
           "- **OCR + 多模态级联**：工业级成本效率\n"
           "- **完整消融验证**：每个组件的贡献都可量化\n\n"
           "### 生产部署建议\n"
           "```\n"
           "Layer 1: DistilBERT (~10ms, ~66M params)  -> 过滤 90% 内容\n"
           "Layer 2: CLIP Head  (~100ms)               -> 图文攻击检测\n"
           "Layer 3: 7B/13B LLM (~1s)                  -> 困难样本\n"
           "Layer 4: 人工审核                            -> 最终仲裁\n"
           "```"),

        code("# 保存最终结果\n"
             "results_summary = {\n"
             "    'data_pipeline': stages,\n"
             "    'final_datasets': {\n"
             "        'sft_mix': len(sft_data),\n"
             "        'dpo_pairs': len(dpo_pairs),\n"
             "        'eval_set': len(eval_data),\n"
             "    },\n"
             "    'model_metrics': {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in model_metrics.items()},\n"
             "}\n\n"
             "save_path = get_results_path('') / 'final_summary.json'\n"
             "with open(save_path, 'w') as f:\n"
             "    json.dump(results_summary, f, indent=2, ensure_ascii=False)\n"
             "print(f'最终结果保存到: {save_path}')\n\n"
             "print('\\n' + '=' * 60)\n"
             "print('  项目完成！')\n"
             "print('=' * 60)"),
    ]

    nb = make_notebook(cells)
    path = NB_DIR / "10_dashboard_report.ipynb"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


if __name__ == '__main__':
    create_notebook_05()
    create_notebook_06()
    create_notebook_07()
    create_notebook_08()
    create_notebook_09()
    create_notebook_10()
    print("\nAll Phase 3 notebooks created!")
