import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 CSV 文件
file_path = '../result/gpqa_diamond/gpqa_diamond_results.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# 模型列名
model_columns = [
    'gemini-1.5', 'glm-4-plus', 'doubao-pro', 'ernie-4.0-turbo', 'qwen-plus', '360gpt2-pro', 'yi-lightning',
    'moonshot-v1',

]

# 计算每个模型的正确率
model_accuracy = {}
for model in model_columns:
    accuracy = data[model].mean()
    model_accuracy[model] = accuracy

# 打印数据预览
print("数据预览：")
print(data.head())

# 打印每个模型的正确率
print("\n每个模型的正确率：")
for model, accuracy in model_accuracy.items():
    print(f"{model}: {accuracy:.2%}")

# 计算每道题目的正确率
data['correct'] = data[model_columns].apply(lambda row: row.apply(lambda x: x == 1).sum(), axis=1)
data['total'] = len(model_columns)
data['question_accuracy'] = data['correct'] / data['total']

# 计算整体正确率
overall_accuracy = data['question_accuracy'].mean()
print(f"\n整体正确率: {overall_accuracy:.2%}")

# 计算每道题正确率的方差
question_accuracy_variance = data['question_accuracy'].var()
print(f"\n每道题正确率的方差: {question_accuracy_variance:.4f}")

# 统计模型错误数量分布
error_counts = data['correct'].value_counts().sort_index()
total_questions = len(data)

print("\n模型错误数量分布：")
for correct_count, num_questions in error_counts.items():
    error_count = len(model_columns) - correct_count
    percentage = (num_questions / total_questions) * 100
    print(f"有 {error_count} 个模型做错的题目百分比: {percentage:.2f}%")

# --- 可视化 ---

# 1. 每个模型的正确率柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_accuracy.keys()), y=list(model_accuracy.values()), width=0.5, color="#6495ED")
plt.title('Accuracy of Each Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(model_accuracy.values()):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
plt.xticks(rotation=20)
sns.despine()  # 去除边框
plt.show()

# 2. 多少个模型答对的题目数量分布柱状图
plt.figure(figsize=(10, 6))
error_counts.plot(kind='bar', width=0.5, color="#6495ED")
plt.title('Distribution of Questions by Number of Models Answering Correctly')
plt.xlabel('Number of Models that Answered Correctly')
plt.ylabel('Number of Questions')
total_questions = len(data)
for i, v in enumerate(error_counts):
    percentage = (v / total_questions) * 100
    plt.text(i, v + 0.02, f"{v}\n({percentage:.2f}%)", ha='center')
plt.xticks(rotation=0)
sns.despine()  # 去除边框
plt.show()
