#!/bin/bash
# ============================================================
# 多模态内容安全审核系统 - 环境搭建脚本
# ============================================================
# 用法: bash setup.sh
# 功能: 创建虚拟环境 → 安装依赖 → 检查系统工具 → 验证安装

set -e

echo "============================================"
echo "  多模态内容安全审核系统 - 环境搭建"
echo "============================================"

# 1. 创建并激活虚拟环境
echo ""
echo "[1/5] 创建 Python 虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ 虚拟环境创建完成"
else
    echo "  ✓ 虚拟环境已存在，跳过创建"
fi

source venv/bin/activate
echo "  ✓ 已激活虚拟环境: $(which python3)"

# 2. 升级 pip
echo ""
echo "[2/5] 升级 pip..."
pip install --upgrade pip -q

# 3. 安装 Python 依赖
echo ""
echo "[3/5] 安装 Python 依赖（可能需要几分钟）..."
pip install -r requirements.txt -q
echo "  ✓ Python 依赖安装完成"

# 4. 检查系统工具
echo ""
echo "[4/5] 检查系统工具..."

# 检查 tesseract OCR
if command -v tesseract &> /dev/null; then
    echo "  ✓ Tesseract OCR 已安装: $(tesseract --version 2>&1 | head -1)"
else
    echo "  ✗ Tesseract OCR 未安装"
    echo "    请运行: brew install tesseract"
    echo "    安装后重新运行本脚本"
fi

# 检查 MPS 可用性
echo ""
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('  ✓ PyTorch MPS (Apple GPU) 可用')
    print(f'    PyTorch 版本: {torch.__version__}')
else:
    print('  ✗ PyTorch MPS 不可用，将使用 CPU')
"

# 5. 验证关键库导入
echo ""
echo "[5/5] 验证关键库导入..."
python3 -c "
imports = {
    'datasets': 'HuggingFace Datasets',
    'transformers': 'HuggingFace Transformers',
    'detoxify': 'Detoxify 毒性检测',
    'open_clip': 'OpenCLIP 跨模态',
    'pytesseract': 'Tesseract OCR',
    'PIL': 'Pillow 图像处理',
    'torch': 'PyTorch',
    'sklearn': 'Scikit-learn',
    'anthropic': 'Anthropic API',
    'seaborn': 'Seaborn 可视化',
    'yaml': 'PyYAML',
    'jsonlines': 'JSON Lines',
}
success = 0
fail = 0
for module, name in imports.items():
    try:
        __import__(module)
        print(f'  ✓ {name}')
        success += 1
    except ImportError as e:
        print(f'  ✗ {name}: {e}')
        fail += 1
print(f'\n  结果: {success} 成功, {fail} 失败')
"

# 6. 创建数据目录
echo ""
echo "创建数据目录结构..."
mkdir -p data/{raw,unified,cleaned,augmented,final}
mkdir -p results/{figures,reports,models/{text_classifier,multimodal_classifier}}
echo "  ✓ 目录结构就绪"

echo ""
echo "============================================"
echo "  环境搭建完成！"
echo "============================================"
echo ""
echo "下一步："
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 配置 API Key: 编辑 configs/api_config.yaml"
echo "  3. 下载数据: python scripts/run_download.py"
echo ""
