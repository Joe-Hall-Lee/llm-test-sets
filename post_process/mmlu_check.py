import pandas as pd

file1_path = "F:/CESI/ChallengeBench/result/mmlu/mmlu_EN-US_ernie-4.0-turbo.csv"
file2_path = "F:/CESI/ChallengeBench/result/mmlu/mmlu_EN-US_moonshot-v1.csv"

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 确保 'Index' 列是数字类型
df1['Index'] = pd.to_numeric(df1['Index'], errors='coerce')
df2['Index'] = pd.to_numeric(df2['Index'], errors='coerce')

# 按照 'Subject' 和 'Index' 排序
df1_sorted = df1.sort_values(by=['Subject', 'Index'])
df2_sorted = df2.sort_values(by=['Subject', 'Index'])

# 找到不匹配的行
mismatch = []

for i, (row1, row2) in enumerate(zip(df1_sorted[['Subject', 'Index']].values, df2_sorted[['Subject', 'Index']].values)):
    if not all(row1 == row2):
        mismatch.append((i, row1, row2))

# 输出不匹配的行
if mismatch:
    print(f"Found {len(mismatch)} mismatched rows:")
    for i, row1, row2 in mismatch:
        print(f"Row {i}:")
        print(f"  File 1 -> Subject: {row1[0]}, Index: {row1[1]}")
        print(f"  File 2 -> Subject: {row2[0]}, Index: {row2[1]}")
else:
    print("No mismatches found.")
