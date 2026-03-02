"""
DistilBERT 文本安全分类器
TikTok 审核第一层：毫秒级推理，初筛 90% 内容

为什么用 DistilBERT 而不是大模型？
- ~66M 参数，推理 <10ms
- 大模型用在后面的层级（级联架构）
- 工业界标准做法：轻量模型初筛 + 重模型精筛
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_run_config, get_data_path, get_results_path


class SafetyDataset(Dataset):
    """文本安全数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_training_data(exclude_sources=None):
    """
    加载训练数据

    Args:
        exclude_sources: 要排除的数据源列表（用于消融实验）

    Returns:
        train_texts, train_labels, test_texts, test_labels, label_names
    """
    config = load_run_config()

    # 加载清洗后数据
    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"
    augmented_dir = get_data_path("augmented")

    records = []
    with open(cleaned_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    # 加载增强数据（如果有）
    aug_file = augmented_dir / "augmented_data.jsonl"
    if aug_file.exists():
        with open(aug_file, "r") as f:
            for line in f:
                records.append(json.loads(line))

    # 排除指定数据源（消融实验）
    if exclude_sources:
        records = [r for r in records
                   if r.get("meta", {}).get("source", "") not in exclude_sources]

    # 提取文本和标签（二分类：harmful / unharmful）
    texts = []
    labels = []
    for r in records:
        text = r.get("text", "")
        harm_label = r.get("meta", {}).get("prompt_harm_label", "unknown")
        if harm_label in ("harmful", "unharmful"):
            texts.append(text)
            labels.append(1 if harm_label == "harmful" else 0)

    # 分割训练/测试（80/20）
    random.seed(42)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_texts = [texts[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    test_texts = [texts[i] for i in indices[split:]]
    test_labels = [labels[i] for i in indices[split:]]

    label_names = ["unharmful", "harmful"]

    return train_texts, train_labels, test_texts, test_labels, label_names


def train_classifier(exclude_sources=None, custom_tag=""):
    """
    训练 DistilBERT 文本安全分类器

    Args:
        exclude_sources: 消融实验排除的数据源
        custom_tag: 模型保存的自定义标签

    Returns:
        dict: 训练结果（metrics, history, model_path）
    """
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    config = load_run_config()
    epochs = config["classifier_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    device_name = config.get("device", "cpu")

    # 设备选择
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载训练数据...")
    train_texts, train_labels, test_texts, test_labels, label_names = \
        load_training_data(exclude_sources=exclude_sources)
    print(f"  训练集: {len(train_texts):,} | 测试集: {len(test_texts):,}")
    print(f"  训练集标签分布: {Counter(train_labels)}")

    # 加载模型和分词器
    model_name = config.get("model_name", "distilbert-base-uncased")
    print(f"加载模型: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    model = model.to(device)

    # 创建数据集
    train_dataset = SafetyDataset(train_texts, train_labels, tokenizer)
    test_dataset = SafetyDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练历史
    history = {"train_loss": [], "eval_auc": [], "eval_f1": []}

    # 训练循环
    print(f"\n开始训练: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print("=" * 60)

    step_count = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_steps += 1
            step_count += 1

            if step_count % 100 == 0:
                history["train_loss"].append(total_loss / epoch_steps)

        avg_loss = total_loss / epoch_steps

        # 评估
        eval_metrics = evaluate_model(model, test_loader, device, label_names)
        history["eval_auc"].append(eval_metrics["auc"])
        history["eval_f1"].append(eval_metrics["f1"])

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"AUC={eval_metrics['auc']:.4f}, F1={eval_metrics['f1']:.4f}, "
              f"Precision={eval_metrics['precision']:.4f}, Recall={eval_metrics['recall']:.4f}")

    # 最终评估
    final_metrics = evaluate_model(model, test_loader, device, label_names, detailed=True)

    # 保存模型
    tag = f"_{custom_tag}" if custom_tag else ""
    save_dir = get_results_path(f"models/text_classifier{tag}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n模型保存到: {save_dir}")

    results = {
        "metrics": final_metrics,
        "history": history,
        "model_path": str(save_dir),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "epochs": epochs,
        "exclude_sources": exclude_sources,
    }

    return results


def evaluate_model(model, data_loader, device, label_names, detailed=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs[:, 1].tolist())  # P(harmful)

    # 计算指标
    metrics = {
        "auc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "accuracy": sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
    }

    if detailed:
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
        metrics["classification_report"] = classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True
        )
        metrics["all_preds"] = all_preds
        metrics["all_labels"] = all_labels
        metrics["all_probs"] = all_probs

    return metrics


if __name__ == "__main__":
    results = train_classifier()
    print(f"\n最终指标:")
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
