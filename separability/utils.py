import pandas as pd
import numpy as np
import random


# 固定随机种子
seed = 42
np.random.seed(seed)
random.seed(seed)


def calculate_bootstrap_scores(data, n_bootstrap_samples=1000):
    """
    计算 Bootstrap 分数
    """
    bootstrap_scores = []
    for _ in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_scores.append(np.mean(bootstrap_sample))  # 计算均值
    return np.array(bootstrap_scores)


def calculate_bootstrap_confidence_interval(bootstrap_scores, model_name=None, confidence=0.95):
    """
    使用 Bootstrap 样本的分位数计算置信区间
    """
    if len(bootstrap_scores) <= 1:
        return None, None

    # 计算置信区间的上下界
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    # 使用 np.percentile 计算分位数
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    if model_name:
        print(
            f"Model: {model_name} | Bootstrap 置信区间 ({confidence*100}%): [{lower_bound}, {upper_bound}]")
    else:
        print(
            f"Bootstrap 置信区间 ({confidence*100}%): [{lower_bound}, {upper_bound}]")

    return lower_bound, upper_bound


def calculate_separability(df, n_bootstrap_samples=1000):
    """
    计算给定 DataFrame 的模型可分离性。
    """
    model_names = [col for col in df.columns if col not in [
        'Index', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject']]

    # 获取每对模型的置信区间，存储字典中
    model_confidence_intervals = {}
    for model in model_names:
        model_data = df[model].dropna()
        # 尝试将数据转换为数值类型
        try:
            model_data = pd.to_numeric(model_data, errors='raise')
        except ValueError as e:
            raise Exception(f"Error converting data for model {model}: {e}")

        bootstrap_scores = calculate_bootstrap_scores(
            model_data, n_bootstrap_samples)
        lower_bound, upper_bound = calculate_bootstrap_confidence_interval(
            bootstrap_scores, model_name=model)
        model_confidence_intervals[model] = (lower_bound, upper_bound)

    total_pairs = 0
    separable_pairs = 0

    # 遍历每对模型
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]

            ci1_lower, ci1_upper = model_confidence_intervals[model1]
            ci2_lower, ci2_upper = model_confidence_intervals[model2]

            total_pairs += 1
            # 判断两个置信区间是否重叠
            if ci1_upper < ci2_lower or ci2_upper < ci1_lower:
                separable_pairs += 1

    if total_pairs == 0:
        return 0.0
    else:
        return (separable_pairs / total_pairs) * 100


def calculate_item_difficulty(df):
    """
    计算每个项目的难度：每道题目的正确率。
    """
    item_difficulty = {}

    # 每行是一个问题，每列是一个模型的回答，计算每道题目的正确率（平均得分）
    for index, row in df.iterrows():
        # 去掉非模型列（Index, Question, Answer, Subject）
        total_answers = row.dropna()[8:].values  # 只取模型列进行计算

        # 将 total_answers 转换为数值类型
        total_answers = pd.to_numeric(total_answers, errors='coerce')

        difficulty = 1 - sum(total_answers) / len(total_answers)
        # 以 (Index, Subject) 作为 key，存储难度
        item_difficulty[(row['Index'], row['Subject'])] = difficulty
        # print(f"题目 ({row['Index']}, {row['Subject']}) 的难度：{difficulty:.2f}")

    return item_difficulty


def calculate_item_discrimination(df, high_group_percentage=0.27, low_group_percentage=0.27):
    """
    计算每个项目的区分度：通过高低分组法计算区分度。
    """
    # 计算每个模型的总得分：每列是每个模型的评分，按列求和
    model_scores = df.iloc[:, 8:].apply(
        pd.to_numeric, errors='coerce').sum(axis=0)  # 每列为模型，计算每个模型的总得分

    # 按照模型总得分进行排序
    sorted_model_scores = model_scores.sort_values(ascending=False)

    # 计算高分组和低分组的大小
    high_group_size = int(len(sorted_model_scores) * high_group_percentage)
    low_group_size = int(len(sorted_model_scores) * low_group_percentage)

    # 高分组和低分组模型
    high_group = sorted_model_scores.head(high_group_size).index
    low_group = sorted_model_scores.tail(low_group_size).index
    print(f"高分组模型：{high_group}")
    print(f"低分组模型：{low_group}")

    item_discrimination = {}

    # 计算每道题目的区分度：高分组和低分组在该题目上的得分差
    for index, row in df.iterrows():
        # 使用 (Index, Subject) 作为唯一标识符
        high_group_scores = row[high_group].dropna().astype(
            float).values  # 获取高分组在该题目的得分
        low_group_scores = row[low_group].dropna().astype(
            float).values   # 获取低分组在该题目的得分

        if high_group_scores.size > 0 and low_group_scores.size > 0:
            # 计算高分组和低分组的得分率（通过率）
            PH = high_group_scores.mean()  # 高分组的得分率
            PL = low_group_scores.mean()   # 低分组的得分率

            # 计算区分度：高分组和低分组的得分率差
            discrimination = PH - PL
            item_discrimination[(row['Index'], row['Subject'])
                                ] = discrimination
            # print(f"题目 ({row['Index']}, {row['Subject']}) 的区分度：{discrimination:.2f}")

    return item_discrimination


def calculate_model_accuracy_variance(df):
    """
    计算每个模型的准确率及其方差
    """
    # 提取模型列（假设从第9列开始是模型列）
    model_columns = df.columns[8:]

    # 计算每个模型的准确率
    accuracies = []
    for model in model_columns:
        # 提取模型的回答列
        model_answers = df[model].dropna().astype(float)
        # 计算准确率（假设正确答案为1，错误答案为0）
        accuracy = model_answers.mean()
        accuracies.append(accuracy)

    # 计算准确率的方差
    variance = np.var(accuracies)

    return variance
