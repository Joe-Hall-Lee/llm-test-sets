import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_output_dir(base_dir='./output_multi2pl'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def visualize_subdataset_subplots(file_path, data_list, start, output_dir):
    """
    根据提供的小数据集名称及起始索引，将数据划分成多个子数据集，
    并保存两组图表：难度分布和区分度分布。
    """
    data = load_data(file_path)

    disc_values = [np.mean(d) for d in data.get("disc", [])]
    diff_values = [np.mean(d) for d in data.get("diff", [])]
    ability = data.get("ability", [])

    if not disc_values or not diff_values or not ability:
        print("数据不完整，无法进行可视化。")
        return

    n_items = len(disc_values)
    start0 = [s - 1 for s in start]
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        e = start0[i+1] if i < len(data_list) - 1 else n_items
        subsets.append((data_list[i], s, e))

    colors = ['skyblue', 'salmon', 'lightgreen', 'orchid']

    # 难度分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_diff = diff_values[s:e]
        pct75_diff = np.percentile(subset_diff, 75) if subset_diff else 0
        x_vals = list(range(1, len(subset_diff) + 1))
        ax.plot(x_vals, subset_diff, 'o-', color=color)
        ax.set_title(f"{name} Difficulty (75th pct: {pct75_diff:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Difficulty")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    difficulty_plot_path = os.path.join(
        output_dir, 'difficulty_distribution.png')
    plt.savefig(difficulty_plot_path)
    plt.close()
    print(f"已保存难度分布图: {difficulty_plot_path}")

    # 区分度分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_disc = disc_values[s:e]
        pct75_disc = np.percentile(subset_disc, 75) if subset_disc else 0
        x_vals = list(range(1, len(subset_disc) + 1))
        ax.plot(x_vals, subset_disc, 'o-', color=color)
        ax.set_title(f"{name} Discrimination (75th pct: {pct75_disc:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Discrimination")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    discrimination_plot_path = os.path.join(
        output_dir, 'discrimination_distribution.png')
    plt.savefig(discrimination_plot_path)
    plt.close()
    print(f"已保存区分度分布图: {discrimination_plot_path}")


if __name__ == "__main__":
    # 文件路径
    file_path = 'irt/strong-multi2pl-2000/best_parameters.json'

    data_list = ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']
    # 每个小数据集的起始索引
    start = [1, 14043, 20554, 20752]

    # 创建输出目录
    output_dir = ensure_output_dir()
    print(f"输出目录: {output_dir}")

    # 获取最大能力值
    theta_value = np.max(load_data(file_path).get("ability", []))

    # 生成所有可视化结果
    visualize_subdataset_subplots(file_path, data_list, start, output_dir)
