import pandas as pd

# 读取两个 CSV 文件
df1 = pd.read_csv('result/mmlu/mmlu_results.csv', encoding='latin1')
df2 = pd.read_csv('eval/data/mmlu.csv', encoding='latin1')

# 创建一个字典来存储结果
result = {}

# 遍历第一个 DataFrame 的每一行
for index, row in df1.iterrows():
    subject = row['Subject']

    # 在第二个 DataFrame 中查找匹配的 Subject
    matching_rows = df2[df2['Subject'] == subject]

    # 如果找到匹配的 Subject，比较数据数量
    if not matching_rows.empty:
        count1 = len(df1[df1['Subject'] == subject])
        count2 = len(matching_rows)
        if count1 != count2:
            result[subject] = (count1, count2)

# 打印每个 Subject 和 Subject 的数量
print("每个 Subject 和 Subject 的数量：")
for Subject in df1['Subject'].unique():
    count1 = len(df1[df1['Subject'] == Subject])
    matching_rows = df2[df2['Subject'] == Subject]
    if not matching_rows.empty:
        count2 = len(matching_rows)
    else:
        count2 = 0
    print(f"{Subject}: {count1} (df1), {count2} (df2)")

# 打印总数据数
total_data_count_df1 = len(df1)
total_data_count_df2 = len(df2)
print(f"\n总数据数（df1）：{total_data_count_df1}")
print(f"总数据数（df2）：{total_data_count_df2}")

# 打印结果
print("\n数据数量不相等的 Subject/Subject 和差值：")
for Subject, counts in result.items():
    print(f"{Subject}: df1={counts[0]}, df2={counts[1]}, 差值={counts[0] - counts[1]}")