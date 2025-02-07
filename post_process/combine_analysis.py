import json
import numpy as np
import matplotlib.pyplot as plt

# 3PL 模型的 ICC


def icc(theta, a, b, c):
    """
    计算 3PL 模型的 ICC
    theta: 潜在能力
    a: 区分度
    b: 难度
    c: 猜测
    """
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def fisher_information(theta, a, b, c):
    """
    计算 3PL 模型的 Fisher 信息
    """
    p = icc(theta, a, b, c)  # 计算 ICC 值
    numerator = (1 - p) / p
    denominator = ((p - c) / (1 - c)) ** 2
    return a ** 2 * numerator * denominator  # 计算 Fisher 信息并返回


def leh_score(theta_max, a, b, c):
    """
    计算 LEH 分数：在最大潜在能力处，ICC 的导数
    """
    # 使用数值方法来计算 ICC 在 theta_max 处的导数
    epsilon = 1e-5  # 微小增量
    p_theta_max = icc(theta_max, a, b, c)
    p_theta_max_plus_epsilon = icc(theta_max + epsilon, a, b, c)
    # LEH 通过导数计算
    return (p_theta_max_plus_epsilon - p_theta_max) / epsilon


# 读取 JSON 数据
with open(r'F:\CESI\llm-test-sets\irt\combined-3pl-10000\best_parameters.json', 'r') as file:
    data = json.load(file)

# 提取 diff 和 disc 字段
diff = data['diff']
disc = data['disc']
subject_ids = data['subject_ids']

a_values = data['diff']  # 区分度参数 a
b_values = data['disc']  # 难度参数 b
c_values = data['lambdas']  # 猜测参数 c

# 提取前 14016 条和后面的数据
first_diff = diff[:14016]
first_disc = disc[:14016]
first_a = a_values[:14016]
first_b = b_values[:14016]
first_c = c_values[:14016]

second_diff = diff[14016:]
second_disc = disc[14016:]
second_a = a_values[14016:]
second_b = b_values[14016:]
second_c = c_values[14016:]

# 计算第七十五分位数（75th percentile）难度和区分度
percentile_75_diff_first = np.percentile(first_diff, 75) if first_diff else 0
percentile_75_disc_first = np.percentile(first_disc, 75) if first_disc else 0

percentile_75_diff_second = np.percentile(
    second_diff, 75) if second_diff else 0
percentile_75_disc_second = np.percentile(
    second_disc, 75) if second_disc else 0

# 打印结果
print(f"第一数据集的第七十五分位数难度: {percentile_75_diff_first}")
print(f"第一数据集的第七十五分位数区分度: {percentile_75_disc_first}")
print(f"第二数据集的第七十五分位数难度: {percentile_75_diff_second}")
print(f"第二数据集的第七十五分位数区分度: {percentile_75_disc_second}")

theta_max = -31.4534969329834

# 计算 LEH 和 Fisher 信息
leh_scores_first = []
fisher_infos_first = []
leh_scores_second = []
fisher_infos_second = []

# 遍历前 14016 条数据，计算 LEH 和 Fisher 信息
for i in range(14016):
    # 从数据中读取 a, b, c 值
    a = first_a[i]
    b = first_b[i]
    c = first_c[i]

    leh_scores_first.append(leh_score(theta_max, a, b, c))
    fisher_infos_first.append(fisher_information(theta_max, a, b, c))

# 遍历后面的数据，计算 LEH 和 Fisher 信息
for i in range(14016, len(diff)):
    a = second_a[i - 14016]  # 处理后半部分的索引
    b = second_b[i - 14016]
    c = second_c[i - 14016]

    leh_scores_second.append(leh_score(theta_max, a, b, c))
    fisher_infos_second.append(fisher_information(theta_max, a, b, c))

# 计算并打印 LEH 和 Fisher 信息的第七十五分位数（75th percentile）
percentile_75_leh_first = np.percentile(
    leh_scores_first, 75) if leh_scores_first else 0
percentile_75_fisher_first = np.percentile(
    fisher_infos_first, 75) if fisher_infos_first else 0

percentile_75_leh_second = np.percentile(
    leh_scores_second, 75) if leh_scores_second else 0
percentile_75_fisher_second = np.percentile(
    fisher_infos_second, 75) if fisher_infos_second else 0

print(f"第一数据集的第七十五分位数 LEH: {percentile_75_leh_first}")
print(f"第一数据集的第七十五分位数 Fisher 信息: {percentile_75_fisher_first}")
print(f"第二数据集的第七十五分位数 LEH: {percentile_75_leh_second}")
print(f"第二数据集的第七十五分位数 Fisher 信息: {percentile_75_fisher_second}")

first_a = a_values[5]
first_b = b_values[5]
first_c = c_values[5]

theta_min = -50
theta_max = 50

# 设置 theta 的值范围
theta_range = np.linspace(theta_min, theta_max, 1000)

# 计算第一个题目的 ICC 值
icc_values = icc(theta_range, first_a, first_b, first_c)

# 绘制第一个题目的 ICC 曲线
plt.figure(figsize=(8, 6))
plt.plot(theta_range, icc_values,
         label=f"Item (a={first_a}, b={first_b}, c={first_c})")
plt.title("ICC Curve")
plt.xlabel("Theta (Latent Ability)")
plt.ylabel("P(Xui = 1 | Theta)")
plt.grid(True)

plt.show()
