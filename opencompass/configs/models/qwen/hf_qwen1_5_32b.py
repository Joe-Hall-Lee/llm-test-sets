from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen1.5-32b-hf',
        path='Qwen/Qwen1.5-32B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
