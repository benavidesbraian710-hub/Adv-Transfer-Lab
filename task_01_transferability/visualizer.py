import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_results(csv_path, out_dir, is_grid=True, filename_suffix=""):
    """
    可视化实验结果：支持均值+标准差折线图，以及包含标准差的增强版热力图。
    filename_suffix 用于区分不同的实验阶段图片名。
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # 构造唯一文件名
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""

    # 1. 对比折线图 (Source vs Target)
    plt.figure(figsize=(10, 6))
    for eps in df['eps'].unique():
        sub = df[df['eps'] == eps].sort_values('step')
        label_pfx = f"Eps {round(eps * 255)}/255"

        plt.errorbar(
            sub['step'], sub['tgt_mean'], yerr=sub['tgt_std'],
            fmt='-s', capsize=5, elinewidth=1.5, markeredgewidth=1.5,
            label=f'{label_pfx} Target (VGG16)'
        )

        plt.errorbar(
            sub['step'], sub['src_mean'], yerr=sub['src_std'],
            fmt='--o', alpha=0.4, capsize=3,
            label=f'{label_pfx} Source (ResNet18)'
        )

    plt.title("Adversarial Transferability: ASR vs Steps (Mean ± Std Dev)")
    plt.xlabel("Attack Steps")
    plt.ylabel("Success Rate (ASR %)")
    plt.legend(loc='lower right')
    plt.grid(True, ls='--', alpha=0.7)
    # 修改点：文件名动态化
    plt.savefig(os.path.join(out_dir, f"asr_comparison{suffix_str}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 增强版热力图
    if is_grid:
        pivot_mean = df.pivot(index="eps", columns="step", values="tgt_mean")
        pivot_std = df.pivot(index="eps", columns="step", values="tgt_std")
        new_labels = [f"{round(float(x) * 255)}/255" for x in pivot_mean.index]

        annot_matrix = []
        for m_row, s_row in zip(pivot_mean.values, pivot_std.values):
            annot_matrix.append([f"{m:.1f}\n±{s:.2f}" for m, s in zip(m_row, s_row)])

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_mean,
            annot=np.array(annot_matrix),
            fmt="",
            cmap="YlGnBu",
            yticklabels=new_labels,
            cbar_kws={'label': 'Transfer ASR (%)'}
        )

        plt.title("Transfer ASR Heatmap (Target Model: Mean ± Std Dev)")
        plt.xlabel("Attack Steps")
        plt.ylabel("Epsilon ($\epsilon$)")
        # 修改点：文件名动态化
        plt.savefig(os.path.join(out_dir, f"asr_heatmap{suffix_str}.png"), dpi=300, bbox_inches='tight')
        plt.close()