import json
import numpy as np
import matplotlib.pyplot as plt


def icc_3pl(theta, a, b, c):
    """
    计算 3PL 模型的 ICC（项目特征曲线）
    
    参数：
        theta: 潜在能力
        a: 区分度
        b: 难度
        c: 猜测参数
    返回：
        题目正确率
    """
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def load_data(file_path):
    """加载数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fisher_information_3pl(theta, a, b, c):
    """
    计算 3PL 模型的 Fisher 信息
    使用公式：I(theta) = a^2 * (p(theta) - c)^2 * (1 - p(theta)) / (p(theta) * (1 - c)^2)
    """
    p = icc_3pl(theta, a, b, c)
    return (a**2 * (p - c)**2 * (1 - p)) / (p * (1 - c)**2)


def leh_score_3pl(theta_max, a, b, c):
    """
    计算 LEH 分数：在 theta_max 处，使用数值方法计算 ICC 的导数
    """
    epsilon = 1e-5
    p_theta = icc_3pl(theta_max, a, b, c)
    p_theta_plus = icc_3pl(theta_max + epsilon, a, b, c)
    return (p_theta_plus - p_theta) / epsilon


def visualize_3pl_subdataset_subplots(file_path, data_list, start):
    """
    从整个大数据集的 JSON 文件中读取数据，
    根据提供的小数据集名称及起始索引，将数据划分成多个子数据集，
    并绘制三组子图：
      1. 难度分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为难度。
      2. 区分度分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为区分度。
      3. 猜测参数分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为猜测参数。
    同时计算并在每个子图标题中显示该子数据集 75th percentile 值。
    """
    # 读取整个数据集
    data = load_data(file_path)

    # 提取需要的字段：区分度（disc）、难度（diff）和猜测参数（lambdas）
    disc_values = data.get("disc", [])  # 区分度参数 a
    diff_values = data.get("diff", [])  # 难度参数 b
    lambdas_values = data.get("lambdas", [])  # 猜测参数 c
    ability = data.get("ability", [])  # 能力参数

    if not disc_values or not diff_values or not lambdas_values or not ability:
        print("数据不完整，无法进行可视化。")
        return

    n_items = len(disc_values)
    # 将 1 基索引转换为 0 基索引
    start0 = [s - 1 for s in start]
    # 构建每个子数据集的区间
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
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        # 当前子数据集的难度数据（对应于 diff_values）
        subset_diff = diff_values[s:e]
        pct75_diff = np.percentile(subset_diff, 75) if subset_diff else 0
        # 横坐标为相对题号
        x_vals = list(range(1, len(subset_diff) + 1))
        ax.plot(x_vals, subset_diff, 'o-', color=color)
        ax.set_title(f"{name} Difficulty (75th pct: {pct75_diff:.4f})")
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
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        # 当前子数据集的区分度数据（对应于 disc_values）
        subset_disc = disc_values[s:e]
        pct75_disc = np.percentile(subset_disc, 75) if subset_disc else 0
        x_vals = list(range(1, len(subset_disc) + 1))
        ax.plot(x_vals, subset_disc, 'o-', color=color)
        ax.set_title(f"{name} Discrimination (75th pct: {pct75_disc:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Discrimination")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 绘制猜测参数分布的子图（散点图）并计算 75th percentile
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        # 当前子数据集的猜测参数数据（对应于 lambdas_values）
        subset_lambdas = lambdas_values[s:e]
        pct75_lambdas = np.percentile(subset_lambdas, 75) if subset_lambdas else 0
        x_vals = list(range(1, len(subset_lambdas) + 1))
        ax.plot(x_vals, subset_lambdas, 'o-', color=color)
        ax.set_title(
            f"{name} Guessing Parameter (75th pct: {pct75_lambdas:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Guessing Parameter")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def analyze_3pl_leh_fisher_by_subset(file_path, data_list, start, theta_value):
    """
    根据给定的小数据集划分，对每个子数据集计算 LEH 分数和 Fisher 信息，
    并计算每个子数据集的 75th percentile 值，然后打印结果。

    参数：
      file_path: JSON 文件路径。
      data_list: 小数据集名称列表，例如 ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']。
      start: 每个小数据集的起始索引（基于 1 的索引）。
      theta_value: 用于计算 LEH 和 Fisher 信息的 theta 值。
    """
    data = load_data(file_path)

    # 提取参数，使用 "disc" 为区分度参数 a，"diff" 为难度参数 b，"lambdas" 为猜测参数 c
    disc_values = data.get("disc", [])  # 区分度参数 a
    diff_values = data.get("diff", [])  # 难度参数 b
    lambdas_values = data.get("lambdas", [])  # 猜测参数 c
    ability = data.get("ability", [])  # 能力参数
    if not disc_values or not diff_values or not lambdas_values or not ability:
        print("数据不完整，无法计算 LEH 和 Fisher 信息。")
        return

    n_items = len(disc_values)
    start0 = [s - 1 for s in start]
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        e = start0[i+1] if i < len(data_list) - 1 else n_items
        subsets.append((data_list[i], s, e))

    print("\n各子数据集 LEH 和 Fisher 信息的 75th percentile：")
    for (name, s, e) in subsets:
        subset_a = disc_values[s:e]
        subset_b = diff_values[s:e]
        subset_c = lambdas_values[s:e]
        leh_scores = []
        fisher_infos = []
        for a, b, c in zip(subset_a, subset_b, subset_c):
            leh_scores.append(leh_score_3pl(theta_value, a, b, c))
            fisher_infos.append(fisher_information_3pl(theta_value, a, b, c))
        pct75_leh = np.percentile(leh_scores, 75) if leh_scores else 0
        pct75_fisher = np.percentile(fisher_infos, 75) if fisher_infos else 0
        print(
            f"{name}: LEH 75th pct = {pct75_leh:.4f}, Fisher 75th pct = {pct75_fisher:.4f}")


if __name__ == "__main__":
    # 整个大数据集的 JSON 文件路径
    file_path = 'irt/strong-3pl-2000/best_parameters.json'
    # 小数据集名称列表
    data_list = ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']
    # 每个小数据集的起始索引（基于 1 的索引）
    start = [1, 14043, 20554, 20752]
    # 用于计算 LEH 和 Fisher 信息的 theta 值
    theta_value = np.max(load_data(file_path).get("ability", []))  # 动态获取最大能力值

    visualize_3pl_subdataset_subplots(file_path, data_list, start)
    analyze_3pl_leh_fisher_by_subset(file_path, data_list, start, theta_value)

    data = load_data(file_path)
    a_item = data.get("disc", [])[4]
    b_item = data.get("diff", [])[4]
    c_item = data.get("lambdas", [])[4]
    theta_min = -50
    theta_max = 50
    theta_range = np.linspace(theta_min, theta_max, 1000)
    icc_values = icc_3pl(theta_range, a_item, b_item, c_item)

    plt.figure(figsize=(8, 6))
    plt.plot(theta_range, icc_values,
             label=f"Item (a={a_item}, b={b_item}, c={c_item})", color='blue')
    plt.title("ICC Curve for a 3PL Model Item")
    plt.ticklabel_format(style='plain', axis='y')  # 禁用科学计数法
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.9f'))  # 设置小数点后 4 位
    plt.xlabel("Theta (Latent Ability)")
    plt.ylabel("P(X=1|Theta)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
