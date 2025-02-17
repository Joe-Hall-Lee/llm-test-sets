#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np


def visualize_subdataset_subplots(file_path, data_list, start):
    """
    从整个大数据集的 JSON 文件中读取数据，
    根据提供的小数据集名称及起始索引，将数据划分成多个小数据集，
    并绘制两张图：一张显示各子数据集的难度分布，另一张显示各子数据集的区分度分布。
    每个图中有 4 个子图，每个子图横坐标为数据点的相对编号，纵坐标分别为难度或区分度。
    同时计算并打印每个子数据集的 75th percentile 值。

    参数：
      file_path: str, JSON 文件路径，包含整个大数据集。
      data_list: list of str, 小数据集名称列表，例如 ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']。
      start: list of int, 每个小数据集的起始索引（基于 1 的索引），例如 [1, 14043, 20554, 20752]。
    """
    # 读取整个数据集
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取需要的字段：难度和区分度
    difficulty = data.get("diff", [])
    discrimination = data.get("disc", [])
    if not difficulty or not discrimination:
        print("数据不完整，无法进行可视化。")
        return

    n_items = len(difficulty)
    # 将 1 基索引转换为 0 基索引
    start0 = [s - 1 for s in start]
    # 根据 start 构建每个小数据集的区间
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        e = start0[i+1] if i < len(data_list) - 1 else n_items
        subsets.append((data_list[i], s, e))

    # 定义颜色列表（数量应与 data_list 相匹配）
    colors = ['skyblue', 'salmon', 'lightgreen', 'orchid']

    # -------------------------------
    # 绘制难度分布的子图（散点图）并计算 75th percentile
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    print("各子数据集难度 75th percentile：")
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_diff = difficulty[s:e]
        # 计算 75th percentile
        pct75_diff = np.percentile(subset_diff, 75) if subset_diff else 0
        print(f"{name}: {pct75_diff:.2f}")
        # 横坐标为相对题号
        x_vals = list(range(1, len(subset_diff) + 1))
        ax.plot(x_vals, subset_diff, 'o-', color=color)
        ax.set_title(f"{name} Difficulty (75th pct: {pct75_diff:.2f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Difficulty")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 绘制区分度分布的子图（散点图）并计算 75th percentile
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    print("\n各子数据集区分度 75th percentile：")
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_disc = discrimination[s:e]
        pct75_disc = np.percentile(subset_disc, 75) if subset_disc else 0
        print(f"{name}: {pct75_disc:.2f}")
        x_vals = list(range(1, len(subset_disc) + 1))
        ax.plot(x_vals, subset_disc, 'o-', color=color)
        ax.set_title(f"{name} Discrimination (75th pct: {pct75_disc:.2f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Discrimination")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 整个大数据集的 JSON 文件路径
    file_path = 'irt/2pl-2000/parameters.json'
    # 小数据集名称列表
    data_list = ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']
    # 每个小数据集的起始索引（基于 1 的索引）
    start = [1, 14043, 20554, 20752]

    visualize_subdataset_subplots(file_path, data_list, start)
