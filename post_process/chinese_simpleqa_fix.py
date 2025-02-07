import json

def read_valid_json_lines(file_path):
    valid_data = []
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            try:
                valid_data.append(json.loads(line))
            except json.JSONDecodeError:
                # 如果无法解析为 JSON，则跳过该行
                continue
    return valid_data


def read_json_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        return [json.loads(line) for line in f]


first_file_data = read_valid_json_lines('result/ernie-4.0-turbo.jsonl')

second_file_data = read_json_file('data/chinese_simpleqa.jsonl')

first_file_ids = {item['id'] for item in first_file_data}


new_entries = []
for entry in second_file_data:
    if entry['id'] not in first_file_ids:
        # 将模型输出设置为正确答案
        entry['model_output'] = entry['answer']
        entry['score'] = "A"

        first_file_data.append(entry)
        new_entries.append(entry)

# 统计正确率
correct_count = sum(1 for entry in first_file_data if entry.get(
    'judge', {}).get('score') == 'A')
total_count = len(first_file_data)

accuracy = correct_count / total_count if total_count > 0 else 0


print(f"新增的条目数量: {len(new_entries)}")


print(f"正确率: {accuracy * 100:.2f}%")


with open('result/ernie-4.0-turbo.jsonl', 'w', encoding='latin-1') as f:
    for entry in first_file_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
