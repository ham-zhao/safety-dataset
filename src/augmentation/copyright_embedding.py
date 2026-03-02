"""
版权 IP embedding 增强模块
基于 CLIP 的 IP embedding 库生成版权相关训练样本

在 TikTok 场景：
- 用户上传的视频中可能包含未授权的卡通角色、品牌 logo
- 需要 CLIP embedding 级别的检测，而不仅仅是文本关键词
"""

import json
import random
import pickle
import numpy as np
from pathlib import Path

# 20 个知名 IP 的文本描述（用于生成 CLIP text embedding 作为近似）
IP_DATABASE = {
    "Mickey Mouse": "Mickey Mouse Disney cartoon character with round ears",
    "Hello Kitty": "Hello Kitty Sanrio white cat character with bow",
    "Pikachu": "Pikachu Pokemon yellow electric mouse character",
    "SpongeBob": "SpongeBob SquarePants yellow sponge cartoon character",
    "Mario": "Super Mario Nintendo red hat plumber game character",
    "Elsa": "Elsa Frozen Disney ice queen character",
    "Naruto": "Naruto anime character orange ninja",
    "Batman": "Batman DC Comics dark knight superhero",
    "Spider-Man": "Spider-Man Marvel red blue web superhero",
    "Doraemon": "Doraemon blue robot cat Japanese anime character",
    "Peppa Pig": "Peppa Pig pink cartoon pig character",
    "Minions": "Minions yellow small creatures goggles",
    "Winnie the Pooh": "Winnie the Pooh Disney bear honey",
    "Tom and Jerry": "Tom and Jerry cartoon cat mouse chase",
    "Sonic": "Sonic the Hedgehog blue fast game character",
    "Totoro": "Totoro Studio Ghibli large forest spirit",
    "Buzz Lightyear": "Buzz Lightyear Toy Story space ranger",
    "Iron Man": "Iron Man Marvel red gold armor superhero",
    "Sailor Moon": "Sailor Moon anime magical girl character",
    "Shrek": "Shrek green ogre DreamWorks character",
}


def generate_copyright_training_data(ip_embedding_path=None, n_samples=50, seed=42):
    """
    生成版权相关的训练样本

    两类样本：
    1. 含有 IP 内容的有害样本（未授权使用）
    2. 不含 IP 内容的安全样本（原创或合规内容）

    Args:
        ip_embedding_path: IP embedding 库路径（如果已有）
        n_samples: 每类样本数
        seed: 随机种子

    Returns:
        list[dict]: 训练样本
    """
    random.seed(seed)
    records = []

    # 有害样本：涉及版权 IP 的未授权使用
    ip_names = list(IP_DATABASE.keys())
    copyright_harmful_templates = [
        "Create an image of {ip} for my commercial product",
        "Generate {ip} fan art to sell as merchandise",
        "Make a {ip} themed poster for my business",
        "Design a {ip} logo for my company brand",
        "Draw {ip} for my YouTube channel banner",
        "Create {ip} stickers to sell online",
        "Make a {ip} t-shirt design for my store",
        "Generate a {ip} animation for my advertisement",
    ]

    for i in range(n_samples):
        ip = random.choice(ip_names)
        template = random.choice(copyright_harmful_templates)
        text = template.format(ip=ip)

        records.append({
            "text": f"User: {text}",
            "images": [],
            "meta": {
                "source": "copyright_augmented",
                "risk_category": "14_copyright",
                "severity": "medium",
                "attack_type": "vanilla",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unknown",
                "response_type": "unknown",
                "is_contrastive_benign": False,
                "synthetic": True,
                "ip_name": ip,
            }
        })

    # 安全样本：合规的创作内容
    safe_templates = [
        "Create an original character design for my story",
        "Generate a unique mascot for my school project",
        "Design an abstract logo using geometric shapes",
        "Draw a landscape painting inspired by nature",
        "Create a portrait illustration in watercolor style",
        "Make an original comic strip about daily life",
        "Design a poster using royalty-free graphics",
        "Generate abstract art with vibrant colors",
    ]

    for i in range(n_samples):
        text = random.choice(safe_templates)
        records.append({
            "text": f"User: {text}",
            "images": [],
            "meta": {
                "source": "copyright_augmented",
                "risk_category": "safe",
                "severity": "none",
                "attack_type": "vanilla",
                "prompt_harm_label": "unharmful",
                "response_harm_label": "unknown",
                "response_type": "compliance",
                "is_contrastive_benign": True,
                "synthetic": True,
            }
        })

    random.shuffle(records)
    return records


def build_ip_embedding_library(output_path, device=None):
    """
    构建 IP text embedding 库（使用 CLIP text encoder 作为代理）

    Args:
        output_path: 保存路径
        device: 计算设备

    Returns:
        dict: {ip_name: embedding_vector}
    """
    try:
        from src.cleaning.cross_modal_validator import CLIPValidator
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.cleaning.cross_modal_validator import CLIPValidator

    validator = CLIPValidator(device=device)

    ip_library = {}
    for ip_name, description in IP_DATABASE.items():
        embedding = validator.encode_text(description)
        ip_library[ip_name] = embedding[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(ip_library, f)

    print(f"IP embedding 库已保存: {output_path} ({len(ip_library)} IPs)")
    return ip_library
