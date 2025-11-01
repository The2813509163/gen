datasets = [
    [
        dict(
            abbr='game24',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.Game24Evaluator'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.game24_postprocess')),
            infer_cfg=dict(
                inferencer=dict(
                    generation_kwargs=dict(do_sample=False, temperature=0.7),
                    method_evaluate='value',
                    method_generate='propose',
                    method_select='greedy',
                    n_evaluate_sample=3,
                    n_select_sample=5,
                    prompt_wrapper=dict(
                        type='opencompass.datasets.Game24PromptWrapper'),
                    type='opencompass.openicl.icl_inferencer.ToTInferencer'),
                prompt_template=dict(
                    template='{input}',
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/game24/game24.csv',
            reader_cfg=dict(input_columns=[
                'input',
            ], output_column='output'),
            type='opencompass.datasets.Game24Dataset'),
    ],
]
models = [
    dict(
        abbr='Llama-3.2-1B-Instruct_hf',
        batch_size=8,
        generation_kwargs=dict(),
        max_out_len=512,
        max_seq_len=None,
        model_kwargs=dict(),
        pad_token_id=None,
        path='/deltadisk/shared_data/models/Llama-3.2-1B-Instruct',
        peft_kwargs=dict(),
        peft_path=
        '/deltadisk/guestnju/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves/meta-llama__Llama-3.2-1B-Instruct/lora/sft/max_samples_500000',
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type=
        'opencompass.models.huggingface_above_v4_33.HuggingFacewithChatTemplate'
    ),
]
work_dir = 'outputs/default/20250610_225850'
