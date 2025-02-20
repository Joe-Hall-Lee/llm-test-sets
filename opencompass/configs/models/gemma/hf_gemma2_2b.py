from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma2-2b-hf',
        path='google/gemma-2-2b',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        ),
    )
]
