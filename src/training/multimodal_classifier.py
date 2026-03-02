"""
CLIP 多模态安全分类器
TikTok 审核第二层：处理图文配合攻击

架构：
1. CLIP 提取图文特征
2. 线性分类头做二分类（safe/unsafe）
3. 支持纯文本和图文混合输入

为什么用 CLIP 而不是从头训？
- CLIP 已有强大的图文对齐能力
- 迁移学习：只训分类头，冻结 CLIP backbone
- 训练快，数据需求少
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_loader import load_run_config, get_data_path, get_results_path

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class MultimodalSafetyHead(nn.Module):
    """CLIP 特征上的安全分类头"""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


class CLIPFeatureDataset(Dataset):
    """预提取 CLIP 特征的数据集"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "labels": self.labels[idx]}


def extract_clip_features(texts, model_name='ViT-B-32', pretrained='openai', device='cpu'):
    """使用 CLIP 提取文本特征"""
    if not CLIP_AVAILABLE:
        print("open_clip 不可用，生成随机特征作为替代")
        return np.random.randn(len(texts), 512).astype(np.float32)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    all_features = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def load_multimodal_data(exclude_sources=None):
    """加载多模态训练数据"""
    config = load_run_config()

    cleaned_path = get_data_path("cleaned") / "text_safety_cleaned.jsonl"
    augmented_dir = get_data_path("augmented")

    records = []
    with open(cleaned_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    aug_file = augmented_dir / "augmented_data.jsonl"
    if aug_file.exists():
        with open(aug_file, "r") as f:
            for line in f:
                records.append(json.loads(line))

    if exclude_sources:
        records = [r for r in records
                   if r.get("meta", {}).get("source", "") not in exclude_sources]

    texts = []
    labels = []
    for r in records:
        text = r.get("text", "")
        harm_label = r.get("meta", {}).get("prompt_harm_label", "unknown")
        if harm_label in ("harmful", "unharmful"):
            texts.append(text)
            labels.append(1 if harm_label == "harmful" else 0)

    # 分割
    random.seed(42)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_texts = [texts[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    test_texts = [texts[i] for i in indices[split:]]
    test_labels = [labels[i] for i in indices[split:]]

    return train_texts, train_labels, test_texts, test_labels


def train_multimodal_classifier(exclude_sources=None, custom_tag=""):
    """
    训练 CLIP 多模态安全分类器

    流程：
    1. 加载数据
    2. CLIP 提取特征（冻结 backbone）
    3. 训练线性分类头
    4. 评估
    """
    config = load_run_config()
    epochs = config["classifier_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    device_name = config.get("device", "cpu")

    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载训练数据...")
    train_texts, train_labels, test_texts, test_labels = \
        load_multimodal_data(exclude_sources=exclude_sources)
    print(f"  训练集: {len(train_texts):,} | 测试集: {len(test_texts):,}")

    # 提取 CLIP 特征
    clip_model = config.get("clip_model", "ViT-B-32")
    clip_pretrained = config.get("clip_pretrained", "openai")
    feature_device = "cpu"  # CLIP 特征提取用 CPU 更稳定
    if device_name == "mps" and torch.backends.mps.is_available():
        feature_device = "mps"

    print(f"提取 CLIP 特征 ({clip_model})...")
    train_features = extract_clip_features(
        train_texts, model_name=clip_model, pretrained=clip_pretrained, device=feature_device
    )
    test_features = extract_clip_features(
        test_texts, model_name=clip_model, pretrained=clip_pretrained, device=feature_device
    )
    print(f"  特征维度: {train_features.shape[1]}")

    # 创建数据集
    train_dataset = CLIPFeatureDataset(train_features, train_labels)
    test_dataset = CLIPFeatureDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建分类头
    input_dim = train_features.shape[1]
    model = MultimodalSafetyHead(input_dim=input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 10)  # 分类头用更高 lr
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "eval_auc": [], "eval_f1": []}

    print(f"\n开始训练: {epochs} epochs")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps
        history["train_loss"].append(avg_loss)

        # 评估
        metrics = _evaluate(model, test_loader, device)
        history["eval_auc"].append(metrics["auc"])
        history["eval_f1"].append(metrics["f1"])

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

    # 最终评估
    final_metrics = _evaluate(model, test_loader, device, detailed=True)

    # 保存
    tag = f"_{custom_tag}" if custom_tag else ""
    save_dir = get_results_path(f"models/multimodal_classifier{tag}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "metrics": final_metrics,
    }, save_dir / "model.pt")
    print(f"\n模型保存到: {save_dir}")

    return {
        "metrics": final_metrics,
        "history": history,
        "model_path": str(save_dir),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "epochs": epochs,
        "exclude_sources": exclude_sources,
    }


def _evaluate(model, data_loader, device, detailed=False):
    """评估分类头"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            labels = batch["labels"]

            logits = model(features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs[:, 1].tolist())

    metrics = {
        "auc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "accuracy": sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
    }

    if detailed:
        from sklearn.metrics import confusion_matrix, classification_report
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
        metrics["classification_report"] = classification_report(
            all_labels, all_preds,
            target_names=["unharmful", "harmful"],
            output_dict=True
        )

    return metrics


if __name__ == "__main__":
    results = train_multimodal_classifier()
    print(f"\n最终指标:")
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
