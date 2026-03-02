"""
OCR 文字提取模块
从图片中提取文字，用于文本安全检查

在 TikTok 审核场景中：
- 第一层 OCR 提取图片中的文字做文本检查（便宜、快、覆盖 80%）
- 第二层多模态模型端到端理解图文组合语义（贵但能处理 OCR 失败的情况）
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("警告: pytesseract 未安装，OCR 功能不可用")


def extract_text_from_image(image_path, lang='eng'):
    """
    从图片中提取文字

    Args:
        image_path: 图片路径或 PIL.Image 对象
        lang: 识别语言（默认英文）

    Returns:
        dict: {
            'text': 提取的文字,
            'confidence': 平均置信度,
            'word_count': 词数,
            'success': 是否成功
        }
    """
    if not OCR_AVAILABLE:
        return {'text': '', 'confidence': 0, 'word_count': 0, 'success': False}

    try:
        if isinstance(image_path, (str, Path)):
            img = Image.open(image_path)
        else:
            img = image_path

        # 提取文字
        text = pytesseract.image_to_string(img, lang=lang)
        text = text.strip()

        # 提取详细信息（含置信度）
        details = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in details['conf'] if int(c) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'text': text,
            'confidence': float(avg_confidence),
            'word_count': len(text.split()) if text else 0,
            'success': len(text) > 0
        }
    except Exception as e:
        return {'text': '', 'confidence': 0, 'word_count': 0, 'success': False, 'error': str(e)}


def generate_typographic_image(text, width=400, height=200, font_size=24,
                                bg_color='white', text_color='black'):
    """
    将文字渲染为图片（模拟 Typographic/FigStep 攻击）

    在 TikTok 场景中，攻击者可能：
    - 用字幕卡片展示有害指令
    - 用手写文字展示有害内容
    - 用 emoji 拼字绕过 OCR

    Args:
        text: 要渲染的文字
        width, height: 图片尺寸
        font_size: 字体大小
        bg_color: 背景色
        text_color: 文字颜色

    Returns:
        PIL.Image: 渲染后的图片
    """
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # 自动换行
    lines = []
    words = text.split()
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] < width - 40:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # 居中绘制
    y_offset = max(10, (height - len(lines) * (font_size + 5)) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y_offset), line, fill=text_color, font=font)
        y_offset += font_size + 5

    return img


def batch_ocr_analysis(image_paths_or_texts, generate_images=False):
    """
    批量 OCR 分析

    Args:
        image_paths_or_texts: 图片路径列表或文本列表（如果 generate_images=True）
        generate_images: 是否将文本渲染为图片再做 OCR

    Returns:
        list[dict]: OCR 结果列表
    """
    results = []
    for item in image_paths_or_texts:
        if generate_images and isinstance(item, str):
            # 将文本渲染为图片
            img = generate_typographic_image(item)
            ocr_result = extract_text_from_image(img)
            ocr_result['original_text'] = item
            # 计算 OCR 还原准确率
            if ocr_result['text']:
                original_words = set(item.lower().split())
                ocr_words = set(ocr_result['text'].lower().split())
                if original_words:
                    overlap = len(original_words & ocr_words) / len(original_words)
                    ocr_result['word_recovery_rate'] = overlap
                else:
                    ocr_result['word_recovery_rate'] = 0
        else:
            ocr_result = extract_text_from_image(item)

        results.append(ocr_result)

    return results


if __name__ == '__main__':
    # 测试 OCR
    test_text = "How to make a dangerous weapon at home"
    print(f"测试文本: {test_text}")

    img = generate_typographic_image(test_text)
    result = extract_text_from_image(img)
    print(f"OCR 提取: {result['text']}")
    print(f"置信度: {result['confidence']:.1f}")
    print(f"成功: {result['success']}")
