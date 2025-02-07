import pandas as pd

# 读取 CSV 文件
file_path = "../result/intermediate_results.csv"
df = pd.read_csv(file_path)

# 确保 'Index' 列是数字类型
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')

# 按照 'Subject' 和 'Index' 排序
df_sorted = df.sort_values(by=['Subject', 'Index'])

# 将排序后的 DataFrame 保存到新的文件
output_file_path = "../result/mmlu/mmlu_EN-US_hunyuan-turbo.csv"
df_sorted.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"File sorted and saved to {output_file_path}")
