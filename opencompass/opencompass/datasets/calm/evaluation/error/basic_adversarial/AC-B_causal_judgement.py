# flake8: noqa: E501
import re


def check_standalization(model_response, prompt_style, type):
    if model_response.startswith(
        ('no', '否', 'yes', '是', '- yes', '- 是', '- no', '- 否')):
        return 0
    else:
        return 1


def check_empty(model_response):
    if model_response == '':
        return 1
    else:
        return 0


def check_repetition(model_response):
    if any(response in model_response for response in [
            'answer (yes or no ?)',
            'question: how would a typical person answer each of the following questions about causation?',
            '答案（是或否？）', '问题：对于以下关于因果关系的问题，一个普通人会怎么回答？'
    ]):
        return 1
    else:
        return 0


def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    result = 1 if chinese_pattern.search(text) is not None else 0
    return result


def contains_english(text):
    english_pattern = re.compile(r'[A-Za-z]{2,}')
    result = 1 if english_pattern.search(text) is not None else 0

    return result


def check_abnormality(preds):
    abnormalities = 'All Yes' if all(pred == 1 for pred in preds) else \
                    'All No' if all(pred == 0 for pred in preds) else 0
    return abnormalities
