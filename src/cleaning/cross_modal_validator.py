"""
CLIP 跨模态验证模块
用 CLIP 计算图文对的语义相似度，分析多模态安全攻击的特征

CLIP 在安全场景的局限：
CLIP 是在正常图文对上训练的，对"有害配对"的判断可能不准确。
一张暴力图片配描述暴力的文本，CLIP 分数很高但内容不安全。
所以 CLIP 用于辅助特征提取，不能单独做安全判断。
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("警告: open_clip 未安装，CLIP 功能不可用")


class CLIPValidator:
    """CLIP 跨模态验证器"""

    def __init__(self, model_name='ViT-B-32', pretrained='openai', device=None):
        if not CLIP_AVAILABLE:
            raise RuntimeError("open_clip 未安装")

        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        print(f"加载 CLIP 模型: {model_name} (pretrained={pretrained}, device={device})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("  CLIP 模型加载完成")

    @torch.no_grad()
    def encode_text(self, texts):
        """编码文本为 CLIP embedding"""
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_image(self, images):
        """编码图片为 CLIP embedding"""
        if not isinstance(images, list):
            images = [images]

        processed = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            processed.append(self.preprocess(img))

        batch = torch.stack(processed).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def compute_similarity(self, texts, images):
        """
        计算文本和图片的 CLIP 余弦相似度

        Args:
            texts: 文本列表
            images: 图片列表（路径或 PIL.Image）

        Returns:
            np.ndarray: 相似度矩阵 (len(texts) × len(images))
        """
        text_features = self.encode_text(texts)
        image_features = self.encode_image(images)
        similarity = text_features @ image_features.T
        return similarity

    def compute_pairwise_similarity(self, text_image_pairs):
        """
        计算图文对的逐对相似度

        Args:
            text_image_pairs: [(text, image), ...] 列表

        Returns:
            list[float]: 每对的相似度
        """
        similarities = []
        for text, image in text_image_pairs:
            sim = self.compute_similarity([text], [image])
            similarities.append(float(sim[0, 0]))
        return similarities

    def build_ip_embedding_library(self, ip_images, ip_names):
        """
        构建版权 IP embedding 库

        Args:
            ip_images: IP 参考图片列表
            ip_names: 对应的 IP 名称列表

        Returns:
            dict: {ip_name: embedding_vector}
        """
        library = {}
        for name, img in zip(ip_names, ip_images):
            embedding = self.encode_image([img])
            library[name] = embedding[0]
        return library

    def check_copyright(self, image, ip_library, threshold=0.85):
        """
        版权检测：检查图片是否与已知 IP 相似

        Args:
            image: 待检测图片
            ip_library: IP embedding 库
            threshold: 相似度阈值

        Returns:
            list[dict]: 匹配结果 [{ip_name, similarity, is_match}, ...]
        """
        query_embedding = self.encode_image([image])[0]
        results = []
        for ip_name, ip_embedding in ip_library.items():
            similarity = float(np.dot(query_embedding, ip_embedding))
            results.append({
                'ip_name': ip_name,
                'similarity': similarity,
                'is_match': similarity >= threshold
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results


def analyze_attack_types_with_clip(validator, attack_data):
    """
    用 CLIP 分析三种多模态攻击类型的图文相似度特征

    预期结果：
    - Typographic 攻击: CLIP 相似度最低（图片只是渲染文字，语义对齐差）
    - Query-Relevant 攻击: CLIP 相似度最高（语义相关图片）
    - Vanilla 攻击: CLIP 相似度中等

    Args:
        validator: CLIPValidator 实例
        attack_data: 按攻击类型分组的数据

    Returns:
        dict: {attack_type: {'mean_sim': float, 'std_sim': float, 'samples': int}}
    """
    results = {}
    for attack_type, pairs in attack_data.items():
        if not pairs:
            continue
        similarities = validator.compute_pairwise_similarity(pairs)
        results[attack_type] = {
            'mean_sim': np.mean(similarities),
            'std_sim': np.std(similarities),
            'min_sim': np.min(similarities),
            'max_sim': np.max(similarities),
            'samples': len(similarities),
        }
    return results


if __name__ == '__main__':
    if CLIP_AVAILABLE:
        validator = CLIPValidator()
        # 简单测试
        text_emb = validator.encode_text("a photo of a cat")
        print(f"文本 embedding shape: {text_emb.shape}")
        print(f"embedding 范数: {np.linalg.norm(text_emb):.4f}")
    else:
        print("CLIP 不可用，跳过测试")
