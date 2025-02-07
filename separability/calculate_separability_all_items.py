import pandas as pd
from utils import calculate_separability, calculate_model_accuracy_variance

def main(csv_file, n_bootstrap_samples=1000):
    """
    主函数：读取 CSV 文件，计算可分离性，并打印结果。
    """
    df = pd.read_csv(csv_file, encoding='latin-1')
    # 检查数据类型
    print("Data types before conversion:")
    print(df.dtypes)
    print(df.head())

    separability = calculate_separability(df, n_bootstrap_samples)
    print(f"模型对的可分离性为：{separability:.2f}%")
    variance = calculate_model_accuracy_variance(
        df)
    print(f"模型准确率方差为：{variance:.6f}")


if __name__ == "__main__":
    csv_file_path = "result/mmlu/mmlu_results.csv"
    n_bootstrap_samples = 100
    main(csv_file_path, n_bootstrap_samples)
