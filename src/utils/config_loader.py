"""
配置加载器 - 统一读取 run_config.yaml，根据 run_mode 返回当前配置
所有 Pipeline 和 Notebook 通过此模块获取运行参数
"""

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录（向上查找包含 configs/ 的目录）"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir():
            return parent
    # 回退：假设从项目根目录运行
    return Path.cwd()


def load_yaml(config_name: str) -> dict:
    """加载 configs/ 目录下的 YAML 配置文件"""
    root = get_project_root()
    config_path = root / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_run_config() -> dict:
    """
    加载运行配置，根据 run_mode 返回对应的参数

    Returns:
        dict: 包含当前模式的所有参数 + common 配置
        示例: {
            'run_mode': 'smoke_test',
            'text_sample_size': 2000,
            'image_text_sample_size': 500,
            'synthesis_count': 50,
            'classifier_epochs': 1,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'seed': 42,
            'device': 'mps',
            ...
        }
    """
    config = load_yaml("run_config.yaml")
    run_mode = config["run_mode"]

    if run_mode not in ("smoke_test", "full_run"):
        raise ValueError(f"无效的 run_mode: {run_mode}，必须是 smoke_test 或 full_run")

    # 合并当前模式配置 + 通用配置
    mode_config = config[run_mode].copy()
    mode_config["run_mode"] = run_mode
    if "common" in config:
        mode_config.update(config["common"])

    return mode_config


def load_api_config() -> dict:
    """加载 API 配置（Anthropic API Key 等）"""
    config = load_yaml("api_config.yaml")
    # 环境变量优先级高于配置文件
    if os.environ.get("ANTHROPIC_API_KEY"):
        config["anthropic"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]
    return config


def load_eval_config() -> dict:
    """加载评估配置（Benchmark + 消融实验参数）"""
    return load_yaml("eval_config.yaml")


def get_data_path(subdir: str = "") -> Path:
    """获取数据目录路径"""
    root = get_project_root()
    path = root / "data" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path(subdir: str = "") -> Path:
    """获取结果目录路径"""
    root = get_project_root()
    path = root / "results" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


# 便捷打印当前配置
def print_config():
    """打印当前运行配置摘要"""
    config = load_run_config()
    mode = config["run_mode"]
    print(f"{'='*50}")
    print(f"  当前运行模式: {mode.upper()}")
    print(f"{'='*50}")
    print(f"  文本样本数:     {config['text_sample_size']:,}")
    print(f"  图文样本数:     {config['image_text_sample_size']:,}")
    print(f"  合成增强数:     {config['synthesis_count']:,}")
    print(f"  分类器 Epochs:  {config['classifier_epochs']}")
    print(f"  设备:           {config.get('device', 'cpu')}")
    print(f"  随机种子:       {config.get('seed', 42)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print_config()
