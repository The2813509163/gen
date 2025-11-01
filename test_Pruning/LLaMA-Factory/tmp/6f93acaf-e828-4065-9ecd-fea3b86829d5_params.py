datasets = [
    [
        dict(
            abbr='gsm8k',
            eval_cfg=dict(
                dataset_postprocessor=dict(
                    type='opencompass.datasets.gsm8k_dataset_postprocess'),
                evaluator=dict(
                    type='opencompass.datasets.MATHEvaluator', version='v2'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.math_postprocess_v2')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            '{question}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='opencompass/gsm8k',
            reader_cfg=dict(
                input_columns=[
                    'question',
                ], output_column='answer'),
            type='opencompass.datasets.GSM8KDataset'),
    ],
]
eval = dict(runner=dict(task=dict()))
models = [
    dict(
        abbr='Llama-3.2-1B_hf',
        batch_size=8,
        generation_kwargs=dict(),
        max_out_len=512,
        max_seq_len=None,
        model_kwargs=dict(),
        pad_token_id=None,
        path='/deltadisk/shared_data/models/Llama-3.2-1B',
        peft_kwargs=dict(),
        peft_path=
        '/deltadisk/guestnju/qianxuzhen/Pruning-LLMs/LLaMA-Factorysaves/meta-llama__Llama-3.2-1B/lora/sft/max_samples_10000',
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel'
    ),
]
work_dir = 'outputs/default/20250604_231441'
