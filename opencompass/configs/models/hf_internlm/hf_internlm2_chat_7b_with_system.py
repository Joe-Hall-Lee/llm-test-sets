from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-7b-hf',
        path='internlm/internlm2-chat-7b',
        tokenizer_path='internlm/internlm2-chat-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
        generation_kwargs = {'eos_token_id': [2, 92542]},
        batch_padding=True,
    )
]
