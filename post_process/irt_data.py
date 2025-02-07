import json

# 读取两个文件的内容
file1_data = []
file2_data = []

with open('F:\CESI\llm-test-sets\irt\pyrit_data\mmlu.jsonlines', 'r') as f1, open('F:\CESI\llm-test-sets\irt\pyrit_data\gpqa_diamond.jsonlines', 'r') as f2:
    file1_data = [json.loads(line) for line in f1]
    file2_data = [json.loads(line) for line in f2]

# 创建字典，按 subject_id 索引
file1_dict = {entry['subject_id']: entry['responses'] for entry in file1_data}
file2_dict = {entry['subject_id']: entry['responses'] for entry in file2_data}

# 找出共有的 subject_id
common_subject_ids = set(file1_dict.keys()).intersection(
    set(file2_dict.keys()))

# 拼接数据并调整问题编号
result = []
for subject_id in common_subject_ids:
    responses1 = file1_dict[subject_id]
    responses2 = file2_dict[subject_id]

    # 拼接并编号调整
    combined_responses = {}
    for idx, (key, value) in enumerate(responses1.items()):
        combined_responses[f"q{idx + 1}"] = value
    for idx, (key, value) in enumerate(responses2.items()):
        combined_responses[f"q{len(responses1) + idx + 1}"] = value

    result.append({"subject_id": subject_id, "responses": combined_responses})

# 将结果写入新文件
with open('combined_output.jsonl', 'w') as output_file:
    for entry in result:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("拼接结果已保存到 'combined_output.jsonl'")
