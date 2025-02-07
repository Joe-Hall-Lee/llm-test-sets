import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr


def visualize_and_correlate(file_path, model_columns):
    """
    从 JSON 文件中读取数据，可视化难度和区分度的分布，同时根据 ability 排序模型，
    计算模型排序的多种相关系数（Spearman、Kendall、Pearson）。
    """
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取相关字段
    ability = data.get("ability", [])
    subject_ids = data.get("subject_ids", {})
    difficulty = data.get("diff", [])
    discrimination = data.get("disc", [])

    # 检查数据完整性
    if not subject_ids or not ability:
        print("数据不完整，请检查 JSON 文件是否包含 'ability' 和 'subject_ids' 字段。")
        return

    if not difficulty or not discrimination:
        print("数据不完整，无法进行难度和区分度的可视化。")
    else:
        # 找到难度和区分度最高的前十道题的序号
        difficulty_top10 = sorted(
            range(len(difficulty)), key=lambda x: difficulty[x], reverse=True)[:10]
        discrimination_top10 = sorted(
            range(len(discrimination)), key=lambda x: discrimination[x], reverse=True)[:10]

        print("\n难度最高的前十道题的序号（从1开始）:", [i + 1 for i in difficulty_top10])
        print("区分度最高的前十道题的序号（从1开始）:", [i + 1 for i in discrimination_top10])

        # 可视化难度和区分度的分布
        plt.figure(figsize=(12, 6))

        # 难度分布
        plt.subplot(1, 2, 1)
        plt.hist(difficulty, bins=20, color='skyblue',
                 edgecolor='black', alpha=0.7)
        plt.xlabel('Difficulty')
        plt.ylabel('Frequency')
        plt.title('Distribution of Question Difficulty')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 区分度分布
        plt.subplot(1, 2, 2)
        plt.hist(discrimination, bins=20, color='salmon',
                 edgecolor='black', alpha=0.7)
        plt.xlabel('Discrimination')
        plt.ylabel('Frequency')
        plt.title('Distribution of Question Discrimination')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    # 根据 ability 排序模型
    sorted_indices = sorted(range(len(ability)),
                            key=lambda x: ability[x], reverse=True)
    sorted_models = [subject_ids[str(i)] for i in sorted_indices]
    sorted_ability = [ability[i] for i in sorted_indices]

    # 打印 ability 排序结果
    print("\n根据 ability 排序的模型：")
    for rank, (model, score) in enumerate(zip(sorted_models, sorted_ability), 1):
        print(f"{rank}. {model} - {score:.2f}")

    # 计算相关系数
    ranked_indices = [sorted_models.index(
        model) for model in model_columns if model in sorted_models]
    reference_ranks = [model_columns.index(
        model) for model in model_columns if model in sorted_models]

    if len(ranked_indices) == len(reference_ranks) > 1:
        spearman_corr, spearman_p = spearmanr(ranked_indices, reference_ranks)
        kendall_corr, kendall_p = kendalltau(ranked_indices, reference_ranks)
        pearson_corr, pearson_p = pearsonr(ranked_indices, reference_ranks)

        print("\n相关系数计算结果：")
        print(f"Spearman相关系数: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
        print(
            f"Kendall Tau相关系数: {kendall_corr:.4f} (p-value: {kendall_p:.4f})")
        print(f"Pearson相关系数: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    else:
        print("\n无法计算相关系数，因匹配的模型数不足。")

    # 可视化 ability 排序分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_ability)), sorted_ability,
            tick_label=sorted_models, color='skyblue', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Ability')
    plt.title('Ability of Models (Sorted)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


file_path = 'F:\CESI\llm-test-sets\irt\gpqa_diamond-3pl-15000/best_parameters.json'
model_columns = [
    'gpt-4o',
    'doubao-pro',
    'qwen-2.5-72b',
    '360gpt2-pro',
    'llama-3.1-70b',
    'hunyuan-turbo',
    'yi-lightning',
    'qwen-plus',
    'glm-4-plus',
    'qwq-32b',
    'gemini-1.5',
    'ernie-4.0-turbo',
    'moonshot-v1',
    'spark4.0-ultra'
]

# model_columns = [
#     'deepseek-v3'
#     'gemini-1.5',
#     'glm-4-plus',
#     'doubao-pro',
#     'ernie-4.0-turbo',
#     'qwen-plus',
#     '360gpt2-pro',
#     'yi-lightning',
#     'hunyuan-turbo',
#     'moonshot-v1',
#     'spark4.0-ultra'

# ]

visualize_and_correlate(file_path, model_columns)
