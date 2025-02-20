import json
import numpy as np
from collections import defaultdict


def calculate_ctt_metrics(file_path, data_list, start_indices):
    # 读取数据
    subjects = []
    all_questions = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            subjects.append(entry)
            all_questions.update(entry['responses'].keys())

    # 生成题目元数据
    questions = sorted(all_questions, key=lambda x: int(x[1:]))
    question_numbers = [int(q[1:]) for q in questions]

    # 计算总分并排序
    for subj in subjects:
        subj['total_score'] = sum(subj['responses'].values())
    sorted_subjects = sorted(
        subjects, key=lambda x: x['total_score'], reverse=True)

    # 确定高低分组（27%）
    n_total = len(sorted_subjects)
    n_group = max(1, int(round(n_total * 0.27)))
    high_group = sorted_subjects[:n_group]
    low_group = sorted_subjects[-n_group:]

    # 初始化结果存储
    question_metrics = []
    for q in questions:
        # 计算难度（错误率）
        total = len(subjects)
        correct = sum(s['responses'][q] for s in subjects)
        difficulty = 1 - (correct / total) if total > 0 else 0

        # 计算区分度
        ph = sum(s['responses'][q]
                 for s in high_group) / n_group if n_group > 0 else 0
        pl = sum(s['responses'][q]
                 for s in low_group) / n_group if n_group > 0 else 0
        discrimination = ph - pl

        question_metrics.append({
            'number': int(q[1:]),
            'difficulty': difficulty,
            'discrimination': discrimination
        })

    # 映射题目到数据集
    dataset_ranges = []
    for i in range(len(start_indices)):
        start = start_indices[i]
        end = start_indices[i+1] - \
            1 if i < len(start_indices)-1 else max(question_numbers)
        dataset_ranges.append((start, end))

    dataset_results = defaultdict(
        lambda: {'difficulties': [], 'discriminations': []})
    for qm in question_metrics:
        q_num = qm['number']
        for idx, (s, e) in enumerate(dataset_ranges):
            if s <= q_num <= e:
                dataset = data_list[idx]
                dataset_results[dataset]['difficulties'].append(
                    qm['difficulty'])
                dataset_results[dataset]['discriminations'].append(
                    qm['discrimination'])
                break

    # 计算平均值
    final_results = {}
    for dataset in data_list:
        diffs = dataset_results[dataset]['difficulties']
        disc = dataset_results[dataset]['discriminations']

        mean_diff = np.mean(diffs) if diffs else None
        mean_disc = np.mean(disc) if disc else None

        final_results[dataset] = {
            'difficulty_mean': float(mean_diff) if mean_diff is not None else None,
            'discrimination_mean': float(mean_disc) if mean_disc is not None else None
        }

    return final_results


# 使用示例
if __name__ == "__main__":
    results = calculate_ctt_metrics(
        file_path="data/combine.jsonlines",
        data_list=['mmlu', 'BBH', 'gpqa_diamond', 'TheoremQA'],
        start_indices=[1, 14043, 20554, 20752]
    )

    print("Dataset\tDifficulty Mean\tDiscrimination Mean")
    for dataset, metrics in results.items():
        print(
            f"{dataset}\t{metrics['difficulty_mean']:.3f}\t\t{metrics['discrimination_mean']:.3f}")
