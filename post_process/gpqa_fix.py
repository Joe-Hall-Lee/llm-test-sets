import pandas as pd
from difflib import SequenceMatcher

# 读取两个 CSV 文件
file1 = pd.read_csv('../eval/data/gpqa_diamond.csv', encoding='latin1')
file2 = pd.read_csv('../result/gpqa_moonshot-v1.csv', encoding='latin1')


# 定义一个函数来计算两个字符串的相似度
def calculate_similarity(a, b):
    # 将 a 和 b 转换为字符串
    a = str(a)
    b = str(b)

    # 如果 a 或 b 为空字符串，返回 0
    if not a or not b:
        return 0

    return SequenceMatcher(None, a, b).ratio()


# 初始化一个列表来存储结果
result = []

# 遍历第一个文件的每一行
for index, row in file1.iterrows():
    # 初始化最大相似度和最相似的行
    max_similarity = 0
    most_similar_row = None

    # 遍历第二个文件的每一行
    for index2, row2 in file2.iterrows():
        # 计算当前行与第二个文件的当前行的相似度
        similarity = calculate_similarity(row['Pre-Revision Question'], row2['Pre-Revision Question'])

        # 如果相似度更高，更新最大相似度和最相似的行
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_row = row2

    # 将最相似的行添加到结果中
    result.append(most_similar_row)

# 将结果转换为 DataFrame
result_df = pd.DataFrame(result)

# 保存结果到新 CSV 文件
result_df.to_csv('result.csv', index=False)
