from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='USER: ', end=' '),
        dict(role='BOT', begin='ASSISTANT: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='wizardlm-70b-v1.0-vllm',
        path='WizardLM/WizardLM-70B-V1.0',
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
