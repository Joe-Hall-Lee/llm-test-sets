from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2-base-7b-turbomind',
        path='internlm/internlm2-base-7b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
