import os
import pandas as pd
import re
from tqdm import tqdm


# 读取数据文件
def read_data(file):
    data = pd.read_csv(file, encoding='latin1')
    return data


# 读取 llama-3.1-70b 文件夹中的 CSV 文件
def read_llama_data(llama_folder):
    llama_data = {}
    for filename in os.listdir(llama_folder):
        if filename.endswith(".csv"):
            subject = filename.replace('.csv', '')
            file_path = os.path.join(llama_folder, filename)
            llama_data[subject] = pd.read_csv(file_path, encoding='latin1')
            print()

    print(len(llama_data))

    return llama_data


# 清理问题文本（去掉多余的空格、换行等）
def clean_question_text(question):
    # 去掉选项部分，留下问题本身
    cleaned_question = re.sub(r'\n|A\)|B\)|C\)|D\)', '', question)
    cleaned_question = cleaned_question.strip()
    return cleaned_question


def merge_data(data, llama_data):
    merged_data = []

    # 使用 tqdm 显示进度条
    for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="合并数据进度"):
        question = row['Question']
        answer = row['Answer']
        subject = row['Subject']

        # 尝试转换索引为整数，如果失败则使用默认值
        try:
            index = int(row["Index"])
        except ValueError:
            index = -1  # 如果无法转换，设置为无效索引

        # 查找对应的 llama 数据
        if subject in llama_data:
            llama_df = llama_data[subject]

            # 通过索引直接匹配
            if index >= 0 and index < len(llama_df):
                matching_row = llama_df.iloc[index]
                correct = matching_row['correct']
            else:
                correct = 'TRUE'

            if correct == 'TRUE' or correct == True:
                score = 1
            else:
                score = 0

            # 创建合并后的新行
            merged_row = row.to_dict()
            merged_row['Correct'] = correct
            merged_row['Score'] = score
            merged_data.append(merged_row)
        else:
            raise Exception(f"No data found for subject: {subject}")

    return pd.DataFrame(merged_data)


# 保存合并后的数据到新的 CSV 文件
def save_merged_data(merged_data, output_file):
    merged_data.to_csv(output_file, index=False, encoding="utf-8-sig")


# 主函数
def main(file, llama_folder, output_file):
    data = read_data(file)
    llama_data = read_llama_data(llama_folder)
    merged_data = merge_data(data, llama_data)
    save_merged_data(merged_data, output_file)
    print("数据已成功合并并保存到 CSV 文件中！")


# 调用主函数
file = '../eval/data/mmlu.csv'
llama_folder = 'D:/微信/WeChat Files/wxid_h3s8ybj8g8r822/FileStorage/File/2024-12/outer_eval/mmlu/output/QwQ-32B-Preview_results'
output_file = 'F:/CESI/ChallengeBench/result/mmlu/mmlu_EN-US_qwq-32b.csv'  # 输出合并后的 CSV 文件路径

main(file, llama_folder, output_file)
