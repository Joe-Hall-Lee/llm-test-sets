import json

def calculate_accuracy(file_path, threshold=0.5):
    correct = 0
    total = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析 JSON 行
            data = json.loads(line.strip())
            
            # 提取真实标签和预测概率
            true_label = int(data['response'])
            pred_prob = float(data['prediction'])
            
            # 将概率转换为预测标签
            pred_label = 1 if pred_prob >= threshold else 0
            
            # 统计正确预测
            if pred_label == true_label:
                correct += 1
            total += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


if __name__ == "__main__":
    file_path = "/H1/zhouhongli/llm-test-sets/irt/3pl-5000/model_predictions.jsonlines"
    acc, correct, total = calculate_accuracy(file_path)
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")