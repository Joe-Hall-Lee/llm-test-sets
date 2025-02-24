# flake8: noqa: E501
from .common_answers import (common_false_list, common_start_false_dict,
                             common_start_true_dict, common_true_list)


def get_gt_label(item):
    return item['gt_answer']


def get_pred_label(model_response, item, prompt_style, type):
    model_response = model_response.strip().lower()
    low_index = len(model_response)

    start_str1_dict = common_start_true_dict
    start_str2_dict = common_start_false_dict

    start_option1_list, start_option2_list = [], []
    # some of the model will give response containing the question,
    # we usually preprocess the response to remove the question part,
    # but sometimes due to the model's response format, some of the
    # question part is not removed, so here we are checking the
    # response with the question part as well.
    for key in start_str1_dict.keys():
        for str1 in start_str1_dict[key]:
            for i in range(key, len(str1) + 1):
                start_option1_list.append(str1[-i:])
    for key in start_str2_dict.keys():
        for str2 in start_str2_dict[key]:
            for i in range(key, len(str2) + 1):
                start_option2_list.append(str2[-i:])

    inner_option1_list = [
        'there is a causal relationship', '存在因果关系', '有因果关系',
        'answer (yes or no?): yes', 'answer is yes', "\"yes\"", 'answer: yes',
        'answer is: yes', 'answer is:\n\nyes', 'answer is:\nyes',
        'there is a causal relationship', '存在因果关系', '存在', '有因果关系', '答案是:是',
        '答案是:\n\n是', '答案是:\n是', '答案:是', '答案是是', '答案为是', "\"是\"", '是的',
        '存在明确的因果关系'
    ] + common_true_list
    inner_option2_list = [
        'there is no causal relationship', '不存在因果关系', '没有因果关系', '没有明显的因果关系',
        '不存在', 'answer (yes or no?): no', 'answer is no', "\"no\"",
        'answer: no', 'answer is: no', 'answer is:\n\nno', 'answer is:\nno',
        'there is no causal relationship', '不存在因果关系', '没有因果关系', '没有明显的因果关系',
        '不存在', '答案是:否', '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', '答案为否',
        "\"否\"", '回答是:否', '没有直接的因果关系'
    ] + common_false_list

    if model_response.startswith(tuple(start_option1_list)):
        return 1
    elif model_response.startswith(tuple(start_option2_list)):
        return 0
    elif any(
            model_response.find(option) > -1 and
        (low_index := min(low_index, model_response.find(option))) > -1
            for option in inner_option1_list):
        label = 1
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option2_list):
            label = 0
        return label
    elif any(response in model_response for response in inner_option2_list):
        return 0
    else:
        return -1
