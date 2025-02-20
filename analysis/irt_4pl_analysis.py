import json
import numpy as np
import matplotlib.pyplot as plt


def icc_4pl(theta, a, b, c, d):
    """
    计算 4PL 模型的 ICC（项目特征曲线）
    
    参数：
        theta: 潜在能力
        a: 区分度
        b: 难度
        c: 猜测参数
        d: 上渐近线参数
    返回：
        题目正确率
    """
    return c + (d - c) / (1 + np.exp(-a * (theta - b)))


def load_data(file_path):
    """加载数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def fisher_information_4pl(theta, a, b, c, d):
    """
    计算 4PL 模型的 Fisher 信息
    公式推导自：I(theta) = a²(d−c)²e^{a(θ−b)} / [1 + e^{a(θ−b)}]⁴ * P(θ)(1−P(θ))
    """
    exponent = np.exp(-a * (theta - b))
    p = icc_4pl(theta, a, b, c, d)
    return (a**2 * (d - c)**2 * exponent) / ((1 + exponent)**4 * p * (1 - p))


def leh_score_4pl(theta_max, a, b, c, d):
    """
    计算 LEH 分数：在 theta_max 处，使用数值方法计算 ICC 的导数
    """
    epsilon = 1e-5
    p_theta = icc_4pl(theta_max, a, b, c, d)
    p_theta_plus = icc_4pl(theta_max + epsilon, a, b, c, d)
    return (p_theta_plus - p_theta) / epsilon


def visualize_4pl_subdataset_subplots(file_path, data_list, start):
    """
    从整个大数据集的 JSON 文件中读取数据，
    根据提供的小数据集名称及起始索引，将数据划分成多个子数据集，
    并绘制四组子图：
      1. 难度分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为难度。
      2. 区分度分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为区分度。
      3. 猜测参数分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为猜测参数。
      4. 上渐近线分布：每个子图横坐标为子数据集内的题目相对编号，纵坐标为上渐近线参数。
    同时计算并在每个子图标题中显示该子数据集 75th percentile 值。
    """
    # 读取整个数据集
    data = load_data(file_path)

    # 提取需要的字段：区分度（disc）、难度（diff）、猜测参数（lambdas）、上渐近线（lambdas）
    disc_values = data.get("disc", [])  # 区分度参数 a
    diff_values = data.get("diff", [])  # 难度参数 b
    lambdas_values = data.get("lambdas", [])  # 猜测参数 c
    lambdas_values = data.get("lambdas", [])  # 上渐近线参数 d
    ability = data.get("ability", [])  # 能力参数

    required_fields = [disc_values, diff_values,
                       lambdas_values, lambdas_values, ability]
    if any(not field for field in required_fields):
        print("数据不完整，无法进行可视化。")
        return

    n_items = len(disc_values)
    # 将 1 基索引转换为 0 基索引
    start0 = [s - 1 for s in start]
    # 构建每个子数据集的区间
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        e = start0[i+1] if i < len(data_list)-1 else n_items
        subsets.append((data_list[i], s, e))

    # 定义颜色列表（数量应与 data_list 相匹配）
    colors = ['skyblue', 'salmon', 'lightgreen', 'orchid']

    # -------------------------------
    # 绘制难度分布的子图（散点图）
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_diff = diff_values[s:e]
        pct75 = np.percentile(subset_diff, 75) if subset_diff else 0
        x_vals = list(range(1, len(subset_diff)+1))
        ax.plot(x_vals, subset_diff, 'o-', color=color)
        ax.set_title(f"{name} Difficulty (75th pct: {pct75:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Difficulty")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 绘制区分度分布的子图（散点图）
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_disc = disc_values[s:e]
        pct75 = np.percentile(subset_disc, 75) if subset_disc else 0
        x_vals = list(range(1, len(subset_disc)+1))
        ax.plot(x_vals, subset_disc, 'o-', color=color)
        ax.set_title(f"{name} Discrimination (75th pct: {pct75:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Discrimination")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 绘制猜测参数分布的子图（散点图）
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_c = lambdas_values[s:e]
        pct75 = np.percentile(subset_c, 75) if subset_c else 0
        x_vals = list(range(1, len(subset_c)+1))
        ax.plot(x_vals, subset_c, 'o-', color=color)
        ax.set_title(f"{name} Guessing Param (75th pct: {pct75:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Guessing Parameter")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 绘制上渐近线分布的子图（散点图）
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for (name, s, e), color, ax in zip(subsets, colors, axes):
        subset_d = lambdas_values[s:e]
        pct75 = np.percentile(subset_d, 75) if subset_d else 0
        x_vals = list(range(1, len(subset_d)+1))
        ax.plot(x_vals, subset_d, 'o-', color=color)
        ax.set_title(f"{name} Upper Asymptote (75th pct: {pct75:.4f})")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Upper Asymptote")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def analyze_4pl_leh_fisher_by_subset(file_path, data_list, start, theta_value):
    """
    根据给定的小数据集划分，对每个子数据集计算 LEH 分数和 Fisher 信息，
    并计算每个子数据集的 75th percentile 值，然后打印结果。
    """
    data = load_data(file_path)

    # 提取参数
    disc_values = data.get("disc", [])  # a
    diff_values = data.get("diff", [])  # b
    lambdas_values = data.get("lambdas", [])  # c
    lambdas_values = data.get("lambdas", [])  # d
    ability = data.get("ability", [])

    required_fields = [disc_values, diff_values,
                       lambdas_values, lambdas_values, ability]
    if any(not field for field in required_fields):
        print("数据不完整，无法计算 LEH 和 Fisher 信息。")
        return

    n_items = len(disc_values)
    start0 = [s - 1 for s in start]
    subsets = []
    for i in range(len(data_list)):
        s = start0[i]
        e = start0[i+1] if i < len(data_list)-1 else n_items
        subsets.append((data_list[i], s, e))

    print("\n各子数据集 LEH 和 Fisher 信息的 75th percentile：")
    for (name, s, e) in subsets:
        subset_a = disc_values[s:e]
        subset_b = diff_values[s:e]
        subset_c = lambdas_values[s:e]
        subset_d = lambdas_values[s:e]

        leh_scores = []
        fisher_infos = []
        for a, b, c, d in zip(subset_a, subset_b, subset_c, subset_d):
            leh_scores.append(leh_score_4pl(theta_value, a, b, c, d))
            fisher_infos.append(
                fisher_information_4pl(theta_value, a, b, c, d))

        pct75_leh = np.percentile(leh_scores, 75) if leh_scores else 0
        pct75_fisher = np.percentile(fisher_infos, 75) if fisher_infos else 0
        print(
            f"{name}: LEH 75th pct = {pct75_leh:.4f}, Fisher 75th pct = {pct75_fisher:.4f}")


if __name__ == "__main__":
    # 配置参数
    file_path = 'irt/strong-4pl-2000/best_parameters.json'
    data_list = ['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA']
    start = [1, 14043, 20554, 20752]
    theta_value = np.max(load_data(file_path).get("ability", []))  # 取最大能力值

    # 执行分析
    visualize_4pl_subdataset_subplots(file_path, data_list, start)
    analyze_4pl_leh_fisher_by_subset(file_path, data_list, start, theta_value)

    # 绘制示例 ICC 曲线
    data = load_data(file_path)
    a = data["disc"][0]
    b = data["diff"][0]
    c = data["lambdas"][0]
    d = data["lambdas"][0]

    theta_range = np.linspace(-50, 50, 1000)
    icc_values = icc_4pl(theta_range, a, b, c, d)

    plt.figure(figsize=(8, 6))
    plt.plot(theta_range, icc_values,
             label=f"Item (a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f})",
             color='blue')
    plt.title("ICC Curve for a 4PL Model Item")
    plt.xlabel("Theta (Latent Ability)")
    plt.ylabel("P(X=1|Theta)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
