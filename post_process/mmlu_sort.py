import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm

# 读取两个文件
df1 = pd.read_csv('F:/CESI/code/mmlu/combined_results.csv')
df2 = pd.read_csv('F:/CESI/ChallengeBench/result/mmlu/mmlu_EN-US_doubao-pro.csv')

# 创建一个字典来快速查找 df1 中的信息
df1_dict = {(row['question'], row['question_id']): row for index, row in df1.iterrows()}

# 创建新的 DataFrame 来存储结果
new_df = pd.DataFrame(columns=df1.columns)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


for index, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Processing"):
    question = row['Question']
    subject = row['Subject']

    # 先尝试完全匹配
    match = df1_dict.get((question, subject))

    # 如果没有完全匹配，尝试找到最高相似度的匹配
    if match is None:
        best_match = None
        best_ratio = 0
        for key in df1_dict:
            if subject == key[1]:  # 只在相同的 subject 下进行模糊匹配
                ratio = similar(question, key[0])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = df1_dict[key]

        if best_match is not None:
            match = best_match

    if match is not None:
        new_df = pd.concat([new_df, pd.DataFrame([match])], ignore_index=True)
    else:
        print(f"Warning: No acceptable match found for question '{question}' in subject '{subject}'.")

# 保存 new_df 到 CSV 文件
new_df.to_csv('F:/CESI/code/mmlu/matched_results.csv', index=False)  # index=False 不保存索引