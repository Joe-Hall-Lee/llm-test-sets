import os
import json
import csv
import glob
import pandas as pd


def merge_json_to_csv(input_dir, output_file, csv_sort_file):
    # 存储所有模型列表
    models = [
        'ernie-4.0-turbo'
    ]

    # 存储问题数据的字典
    questions_data = {}

    # 遍历目录中的所有 JSONL 文件
    for json_file in glob.glob(os.path.join(input_dir, '*.jsonl')):
        # 从文件名中提取模型名称
        model = os.path.basename(json_file).split('.jsonl')[0]

        if model not in models:
            raise Exception(f"Invalid model name: {model}")

        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                # 使用问题 ID 作为唯一标识
                question_id = data['id']

                # 初始化问题数据
                if question_id not in questions_data:
                    questions_data[question_id] = {
                        'question_id': question_id,
                        'question': data['question'],
                        'answer': data['answer'],
                        'model_results': {m: 1 for m in models}
                    }

                score = data.get('score', None)  # 默认值设置为 None

                if score == 'A':
                    result = 1
                else:
                    result = 0

                # 更新模型结果
                questions_data[question_id]['model_results'][model] = result

    # 读取 CSV 文件中 question_id 的值
    csv_data = pd.read_csv(csv_sort_file)
    question_ids = csv_data['question_id'].tolist()

    # 写入 CSV 文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 添加 BOM
        csvfile.write(u'\ufeff')

        # 定义 CSV 表头
        fieldnames = ['question_id', 'question', 'answer'] + models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # 写入表头
        writer.writeheader()

        # 写入每行数据
        for question_id in question_ids:
            if question_id in questions_data:
                row = {
                    'question_id': question_id,
                    'question': questions_data[question_id]['question'],
                    'answer': questions_data[question_id]['answer']
                }
                row.update(questions_data[question_id]['model_results'])
                writer.writerow(row)

    print(f"CSV 文件已生成: {output_file}")


input_directory = 'result'
output_csv_file = 'result/chinese_simpleqa/chinese_simpleqa_results1.csv'
csv_sort_file = 'result/chinese_simpleqa/chinese_simpleqa_results.csv'

merge_json_to_csv(input_directory, output_csv_file, csv_sort_file)
