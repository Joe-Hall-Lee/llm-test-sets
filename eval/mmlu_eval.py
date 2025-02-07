"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import common
from common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from type import Eval, EvalResult, SamplerBase, SingleEvalResult
from call_model import call_model

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(self, num_examples: int | None = None, language: str = "EN-US", max_workers: int = 10):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        filename = "mmlu.csv"

        local_path = os.path.join(data_dir, filename)
        df = pd.read_csv(local_path, encoding='latin1')
        self.original_data = df.copy()  # 保存原始数据
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.results = []  # 用于存储评估结果
        self.max_workers = max_workers  # 设置最大线程数

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict, row_index: int):
            try:
                prompt_messages = [
                    dict(content=format_multichoice_question(row), role="user"),
                ]
                response_text = normalize_response(call_model(format_multichoice_question(row), sampler))
                if response_text is None:
                    print(f"Skipping question due to API error: {row['Question']}")
                    return row_index, SingleEvalResult(html="API Error", score=0.0, metrics={}, convo=[])

                extracted_answer = None
                for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                    match = re.search(regex, response_text)
                    if match:
                        extracted_answer = normalize_extracted_answer(match.group(1))
                        break
                score = 1.0 if extracted_answer == row["Answer"] else 0.0
                html = common.jinja_env.from_string(HTML_JINJA).render(
                    prompt_messages=prompt_messages,
                    next_message=dict(content=response_text, role="assistant"),
                    score=score,
                    correct_answer=row["Answer"],
                    extracted_answer=extracted_answer,
                )
                convo = prompt_messages + [dict(content=response_text, role="assistant")]
                category = subject2category.get(row["Subject"], "other")
                result = SingleEvalResult(
                    html=html, score=score, metrics={category: score}, convo=convo
                )

                # 保存评估结果
                result_data = {
                    "Index": row["Index"],
                    "Question": row["Question"],
                    "Subject": row["Subject"],
                    f"Correct_{sampler}": extracted_answer == row["Answer"],
                    f"Extracted_{sampler}": extracted_answer,
                    f"Score_{sampler}": score,
                    f"Invalid_{sampler}": not extracted_answer
                }

                # 在每个问题评估后直接保存结果
                self.results.append(result_data)
                self.save_intermediate_results()  # 保存每个问题的结果到文件

                return row_index, result
            except Exception as e:
                print(f"Error processing question: {row['Question']}. Error: {e}")
                return row_index, SingleEvalResult(html=f"Error: {e}", score=0.0, metrics={}, convo=[])

        # 使用 ThreadPoolExecutor 进行并发处理，并返回带有索引的结果
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_example, row, idx) for idx, row in enumerate(self.examples)]
            results = []
            for future in tqdm(as_completed(futures), total=len(self.examples), desc="Processing Examples"):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error in worker thread: {e}")

        # 根据 row_index 对结果进行排序，确保顺序与原始数据一致
        results.sort(key=lambda x: x[0])  # 按索引排序
        sorted_results = [result for _, result in results]  # 取出排序后的结果

        return common.aggregate_results(sorted_results)

    def save_intermediate_results(self):
        # 保存中间结果到文件
        if self.results:
            results_df = pd.DataFrame(self.results)
            file_path = f"../result/intermediate_results.csv"
            results_df.to_csv(file_path, index=False, encoding="utf-8")
            print(f"保存中间结果到 {file_path}")

    def get_results_df(self):
        return pd.DataFrame(self.results)
