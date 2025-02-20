from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassBenchCheklistDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'checklist'],
    output_column='judge',
)

subjective_all_sets = {
    'en': [
        'language/compass_bench_language_en_val',
        'instruct/compass_bench_instruct_en_val',
        'reasoning/compass_bench_reasoning_en_val',
        'coding/compass_bench_coding_en_val',
    ],
    'cn': [
        'language/compass_bench_language_cn_val',
        'instruct/compass_bench_instruct_cn_val',
        'reasoning/compass_bench_reasoning_cn_val',
        'coding/compass_bench_coding_cn_val',
    ],
}
data_path = './data/compassbench_v1_3/'

pair_prompt_en = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of the \
responses generated by two AI models.
We will provide you with the user query and a pair of AI-generated \
responses (Response A and Response B).
You should first read the user query and the conversation history \
carefully for analyzing the task, and then evaluate the quality of the \
responses based on and rules provided below.

# Conversation between User and AI

## User Query
<|begin_of_query|>

{question}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{prediction}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{prediction2}

<|end_of_response_B|>

# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your \
assessment to the checklist.

## Rules

You should compare the above two responses based on your analysis of the \
user queries and the conversation history.
You should first write down your analysis and the checklist that you used \
for the evaluation, and then provide your assessment according to the \
checklist.
There are five choices to give your final assessment: ["A++", "A+", \
"A=B", "B+", "B++"], which correspond to the following meanings:

- `A++`: Response A is much better than Response B.
- `A+`: Response A is only slightly better than Response B.
- `A=B`: Response A and B are of the same quality. Please use this \
choice sparingly.
- `B+`: Response B is only slightly better than Response A.
- `B++`: Response B is much better than Response A.

## Output Format
First, please output your analysis for each model response, and \
then summarize your assessment to three aspects: "reason A=B", \
"reason A>B", and "reason B>A", and finally make your choice for \
the final assessment.

Please provide your evaluation results in the following json \
format by filling in the placeholders in []:
```
{
    "analysis of A": "[analysis of Response A]",
    "analysis of B": "[analysis of Response B]",
    "reason of A=B": "[where Response A and B perform equally well]",
    "reason of A>B": "[where Response A is better than Response B]",
    "reason of B>A": "[where Response B is better than Response A]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```
"""


pair_prompt_cn = """# 指令

您是一位专业评估专家。您的任务是评估两个AI模型生成回答的质量。
我们将为您提供用户问题及一对AI生成的回答（回答A和回答B）。
您应当首先仔细阅读用户问题，然后根据以下提供的规则评估回答的质量。

# 用户与AI之间的对话

## 用户问题
<|begin_of_query|>

{question}

<|end_of_query|>

## 回答A
<|begin_of_response_A|>

{prediction}

<|end_of_response_A|>

## 回答B
<|begin_of_response_B|>

{prediction2}

<|end_of_response_B|>

# 评估

## 检查清单

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

请参考此检查清单来评估回答的质量，但不要局限于此检查清单。

## 规则

您应当基于用户查询，分析比较上述两种回答。
您应当基于检查清单写下您的分析，然后提供您的评价。
有五个选项供您做出最终评估：["A++", "A+", "A=B", "B+", "B++"]，它们对应如下含义：

- `A++`：回答A远胜于回答B。
- `A+`：回答A略优于回答B。
- `A=B`：回答A和回答B质量相同。请谨慎使用此选项。
- `B+`：回答B略优于回答A。
- `B++`：回答B远胜于回答A。

## 输出格式
首先，请输出您对每个模型回答的分析，
然后总结您的评估到三个方面："A=B的理由"，"A优于B的理由"，和 "B优于A的理由"，
最后做出您对最终评估的选择。

请按照以下json格式提供您的评估结果，通过填充[]中的占位符：
```
{
    "回答A的分析": "[回答A的分析]",
    "回答B的分析": "[回答B的分析]",
    "A=B的理由": "[A和B回答差不多的理由]",
    "A优于B的理由": "[回答A优于B的理由]",
    "B优于A的理由": "[回答B优于A的理由]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```
"""

checklist_datasets = []
gpt4 = [
    dict(
        abbr='gpt4-1106',
    )
]
for lan, data_name_list in subjective_all_sets.items():
    if lan == 'en':
        pair_prompt = pair_prompt_en
    elif lan == 'cn':
        pair_prompt = pair_prompt_cn
    for _name in data_name_list:
        subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{question}'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

        subjective_eval_cfg = dict(
            evaluator=dict(
                type=LMEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt=pair_prompt),
                        ]
                    ),
                ),
            ),
            pred_role='BOT',
        )

        checklist_datasets.append(
            dict(
                abbr=f'{_name}',
                type=CompassBenchCheklistDataset,
                path=data_path,
                name=_name,
                reader_cfg=subjective_reader_cfg,
                infer_cfg=subjective_infer_cfg,
                eval_cfg=subjective_eval_cfg,
                mode='m2n',
                infer_order='random',
                base_models=gpt4,
            )
        )
