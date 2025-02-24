from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

ocnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'], output_column='label')

# TODO: two prompt templates for ocnli
ocnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'contradiction':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='语句一：“{sentence1}”\n语句二：“{sentence2}”\n请问这两句话是什么关系？'
                ),
                dict(role='BOT', prompt='矛盾')
            ]),
            'entailment':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='语句一：“{sentence1}”\n语句二：“{sentence2}”\n请问这两句话是什么关系？'
                ),
                dict(role='BOT', prompt='蕴含')
            ]),
            'neutral':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='语句一：“{sentence1}”\n语句二：“{sentence2}”\n请问这两句话是什么关系？'
                ),
                dict(role='BOT', prompt='无关')
            ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

ocnli_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

ocnli_datasets = [
    dict(
        type=HFDataset,
        abbr='ocnli',
        path='json',
        split='train',
        data_files='./data/CLUE/OCNLI/dev.json',
        reader_cfg=ocnli_reader_cfg,
        infer_cfg=ocnli_infer_cfg,
        eval_cfg=ocnli_eval_cfg)
]
