#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

def visualize_subdataset_distribution(file_path, data_list, start):
    """
    从整个大数据集的 JSON 文件中读取数据，
    根据提供的小数据集名称及起始索引，将数据划分成多个小数据集，
    并在同一图中以不同颜色显示各小数据集的难度和区分度分布。

    参数：
        file_path: str, JSON 文件路径，包含整个大数据集。
        data_list: list of str, 小数据集名称列表，例如 ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']。
        start: list of int, 每个小数据集起始索引（基于 1 的索引），例如 [1, 14043, 20554, 20752]。
    """
    # 读取整个数据集
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取需要的字段：难度和区分度
    difficulty = data.get("diff", [])
    discrimination = data.get("disc", [])
    
    # 检查数据完整性
    if not difficulty or not discrimination:
        print("数据不完整，无法进行可视化。")
        return

    n_items = len(difficulty)
    
    # 将 1 基索引转换为 0 基索引
    start0 = [s - 1 for s in start]
    
    # 根据 start 列表构建每个子数据集的区间
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        # 如果不是最后一个数据集，结束索引为下一个数据集的起始索引（不包含该索引）
        e = start0[i+1] if i < len(data_list) - 1 else n_items
        subsets.append((data_list[i], s, e))
    
    # 定义颜色列表（数量应与 data_list 相匹配）
    colors = ['skyblue', 'salmon', 'lightgreen', 'orchid']  # 可根据需要调整颜色

    # -------------------------------
    # 可视化难度分布
    # -------------------------------
    plt.figure(figsize=(12, 6))
    bins = 20  # 可根据数据调整
    for (name, s, e), color in zip(subsets, colors):
        subset_diff = difficulty[s:e]
        plt.hist(subset_diff, bins=bins, alpha=0.5, color=color, label=name, edgecolor='black')
    plt.xlabel('Difficulty')
    plt.ylabel('Frequency')
    plt.title('Difficulty Distribution by Subdataset')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 可视化区分度分布
    # -------------------------------
    plt.figure(figsize=(12, 6))
    for (name, s, e), color in zip(subsets, colors):
        subset_disc = discrimination[s:e]
        plt.hist(subset_disc, bins=bins, alpha=0.5, color=color, label=name, edgecolor='black')
    plt.xlabel('Discrimination')
    plt.ylabel('Frequency')
    plt.title('Discrimination Distribution by Subdataset')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = 'data/combine.jsonlines'
    # 小数据集名称列表
    data_list = ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']
    # 每个小数据集的起始索引（基于 1 的索引）
    start = [1, 14043, 20554, 20752]
    
    visualize_subdataset_distribution(file_path, data_list, start)
