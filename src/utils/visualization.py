"""
可视化工具模块 - 通用的图表生成函数
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def setup_plot_style():
    """设置全局绘图风格"""
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    sns.set_style('whitegrid')
    sns.set_palette('husl')


def save_figure(fig, name, results_dir=None):
    """保存图表到 results/figures/"""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / f"{name}.png", dpi=150, bbox_inches='tight')


def plot_category_distribution(counts, title="Risk Category Distribution", save_name=None):
    """绘制风险类别分布柱状图"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(16, 6))

    colors = ['#2ecc71' if 'safe' in str(c) else
              ('#e74c3c' if v < 100 else '#3498db')
              for c, v in counts.items()]

    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(c).replace('_', '\n') for c in counts.index],
                       rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Sample Count')

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_name=None):
    """绘制混淆矩阵热力图"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve", save_name=None):
    """绘制 ROC 曲线"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_ablation_comparison(results, metric_name="AUC", save_name=None):
    """绘制消融实验对比柱状图"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    values = list(results.values())

    colors = ['#2ecc71' if n == 'Full' else '#e74c3c' for n in names]
    bars = ax.bar(names, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_title(f'Ablation Study: {metric_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name)
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig
