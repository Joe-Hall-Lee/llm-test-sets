import json
from collections import Counter
import matplotlib.pyplot as plt


def process_data(file_path):
    """
    从文件中读取数据，统计每个题目被多少模型答对，并绘制图表。
    """
    # 读取文件并解析数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    # 初始化计数器
    question_correct_count = Counter()  # 记录每个题目被答对的次数
    model_correct_count = Counter()  # 记录每个模型答对的题目数
    total_questions = len(data[0]["responses"])  # 总题目数，根据第一个模型的回答判断

    # 遍历每个模型的响应数据
    for entry in data:
        responses = entry.get("responses", {})
        correct_answers = sum(responses.values())  # 该模型答对的题目数
        model_correct_count[entry["subject_id"]] = correct_answers  # 更新模型的答对数量

        # 统计每个题目被答对的次数
        for question, result in responses.items():
            if result == 1:  # 统计答对的题目
                question_correct_count[question] += 1

    # 统计每个答对模型数对应的题目数
    model_correct_stats = Counter(question_correct_count.values())

    # 绘制题目答对的模型数分布图
    x = sorted(model_correct_stats.keys())  # 横坐标：答对的模型数
    y = [model_correct_stats[correct] for correct in x]  # 纵坐标：题目数

    plt.figure(figsize=(8, 6))
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Models Answered Correctly')
    plt.ylabel('Number of Questions')
    plt.title('Distribution of Questions by Number of Correct Models')
    plt.xticks(range(min(x), max(x) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 计算模型准确率并绘制准确率分布图
    model_accuracy = {model_id: correct_count / total_questions for model_id,
                      correct_count in model_correct_count.items()}

    # 按准确率从大到小排序
    sorted_accuracy = sorted(model_accuracy.items(),
                             key=lambda x: x[1])

    # 分离排序后的模型 ID 和准确率
    sorted_model_ids = [item[0] for item in sorted_accuracy]
    sorted_accuracies = [item[1] for item in sorted_accuracy]

    # 绘制模型准确率条形图
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_model_ids, sorted_accuracies,
             color='lightgreen', edgecolor='black')
    plt.xlabel('Accuracy')
    # 隐藏纵坐标名称
    plt.yticks([])  # 隐藏纵坐标标签
    plt.ylabel('Model')
    plt.title('Model Accuracy')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


file_path = 'F:/CESI/llm-test-sets/irt/pyrit_data/mmlu.jsonlines'
process_data(file_path)
