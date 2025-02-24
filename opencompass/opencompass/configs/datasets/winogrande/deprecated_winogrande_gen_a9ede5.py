from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
)

winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                'Which of the following is a good sentence:\nA. {opt1}\nB. {opt2}\nAnswer:'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDatasetV2,
        path='./data/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg,
    )
]
