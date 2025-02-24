from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDataset

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='answer',
    test_split='train')

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: 'Passage:{passage}。\nQuestion:{question}。\nAnswer: No.',
            1: 'Passage:{passage}。\nQuestion:{question}。\nAnswer: Yes.',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

BoolQ_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

BoolQ_datasets = [
    dict(
        type=BoolQDataset,
        abbr='BoolQ',
        path='json',
        data_files='opencompass/boolq',
        split='train',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg)
]
