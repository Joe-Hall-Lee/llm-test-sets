from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-v2-chat-hf',
        path='deepseek-ai/DeepSeek-V2-Chat',
        max_out_len=1024,
        batch_size=2,
        model_kwargs=dict(
            device_map='sequential',
            torch_dtype='torch.bfloat16',
            max_memory={i: '75GB' for i in range(8)},
            attn_implementation='eager'
        ),
        run_cfg=dict(num_gpus=8),
    )
]
