import pandas as pd
import random
from utils import calculate_separability, calculate_item_difficulty, calculate_item_discrimination, calculate_model_accuracy_variance


def select_top_items(item_dict, top_n=100):
    """
    选择区分度或难度最大的前 N 个项目
    """
    # 根据值进行排序并选出 top_n 个
    sorted_items = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = [item[0] for item in sorted_items[:top_n]]
    return top_items


def main(csv_file, n_bootstrap_samples=1000, top_n=1000):
    """
    主函数：读取 CSV 文件，计算可分离性，并打印结果。
    """
    # 读取 CSV 数据
    df = pd.read_csv(csv_file, encoding='latin-1')

    # 打印数据类型和前几行数据，检查读取情况
    print("Data types before conversion:")
    print(df.dtypes)
    print(df.head())

    # 计算项目区分度和难度
    item_discrimination = calculate_item_discrimination(df)
    item_difficulty = calculate_item_difficulty(df)

    # 选择区分度、难度最高的前 top_n 个项目
    top_discrimination_items = select_top_items(item_discrimination, top_n)
    top_difficulty_items = select_top_items(item_difficulty, top_n)

    # 随机抽取 top_n 个项目
    all_items = [(row['Index'], row['Subject']) for _, row in df.iterrows()]
    random_items = random.sample(all_items, top_n)
    # print(f"随机抽取的 {top_n} 个项目：{random_items}")

    # 选择区分度最高的前 top_n 个项目
    df_top_discrimination_items = df[df[['Index', 'Subject']].apply(
        tuple, axis=1).isin(top_discrimination_items)]

    # 选择难度最高的前 top_n 个项目
    df_top_difficulty_items = df[df[['Index', 'Subject']].apply(
        tuple, axis=1).isin(top_difficulty_items)]

    # 选择随机抽取的 top_n 个项目
    df_random_items = df[df[['Index', 'Subject']].apply(
        tuple, axis=1).isin(random_items)]

    # -----------------  统计 Subject 分布  -----------------
    print("\nSubject 分布 - 区分度最高的前 {} 个项目:".format(top_n))
    print(df_top_discrimination_items['Subject'].value_counts())

    print("\nSubject 分布 - 难度最高的前 {} 个项目:".format(top_n))
    print(df_top_difficulty_items['Subject'].value_counts())

    print("\nSubject 分布 - 随机抽取的 {} 个项目:".format(top_n))
    print(df_random_items['Subject'].value_counts())
    # --------------------------------------------------------

    # 计算子集的可分离性
    print(f"\n计算区分度最高的前 {top_n} 个项目的可分离性：")
    separability_discrimination = calculate_separability(
        df_top_discrimination_items, n_bootstrap_samples)

    print(f"\n计算难度最高的前 {top_n} 个项目的可分离性：")
    separability_difficulty = calculate_separability(
        df_top_difficulty_items, n_bootstrap_samples)

    print(f"\n计算随机抽取的 {top_n} 个项目的可分离性：")
    separability_random = calculate_separability(
        df_random_items, n_bootstrap_samples)

    # 输出可分离性
    print(f"\n区分度最高的 {top_n} 个项目的可分离性为：{separability_discrimination:.2f}%")
    print(f"难度最高的 {top_n} 个项目的可分离性为：{separability_difficulty:.2f}%")
    print(f"随机抽取的 {top_n} 个项目的可分离性为：{separability_random:.2f}%")

    # 计算每个子集的模型准确率方差
    variance_discrimination = calculate_model_accuracy_variance(
        df_top_discrimination_items)
    print(f"区分度最高的 {top_n} 个项目的模型准确率方差为：{variance_discrimination:.6f}")

    variance_difficulty = calculate_model_accuracy_variance(
        df_top_difficulty_items)
    print(f"难度最高的 {top_n} 个项目的模型准确率方差为：{variance_difficulty:.6f}")

    variance_random = calculate_model_accuracy_variance(df_random_items)
    print(f"随机抽取的 {top_n} 个项目的模型准确率方差为：{variance_random:.6f}")


if __name__ == "__main__":
    # 设置文件路径和 bootstrap 样本数
    csv_file_path = "result/mmlu/mmlu_results.csv"
    n_bootstrap_samples = 100  # 设置 bootstrap 样本数
    top_n = 1000  # 选择前 N 个项目
    main(csv_file_path, n_bootstrap_samples, top_n)
