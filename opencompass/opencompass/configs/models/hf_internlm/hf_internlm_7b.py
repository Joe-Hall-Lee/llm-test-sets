from opencompass.models import HuggingFaceBaseModel


models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm-7b-hf',
        path='internlm/internlm-7b',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
