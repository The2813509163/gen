datasets = [
    [
        dict(
            abbr='gsm8k_4',
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
                ],
                output_column='answer',
                test_range='[660:825]'),
            type='opencompass.datasets.GSM8KDataset'),
        dict(
            abbr='math_4',
            eval_cfg=dict(
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
                            '{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='opencompass/math',
            reader_cfg=dict(
                input_columns=[
                    'problem',
                ],
                output_column='solution',
                test_range='[2500:3125]'),
            type='opencompass.datasets.MATHDataset'),
        dict(
            abbr='svamp_4',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.Gsm8kEvaluator'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.gsm8k_postprocess')),
            infer_cfg=dict(
                inferencer=dict(
                    max_out_len=512,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Question: There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            'To find the size of each group of bananas, we divide the total number of bananas (290) by the number of groups (2): 290 / 2 = 145. Therefore, each group of bananas contains 145 bananas. The answer is 145.\n',
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            "To find Marco's strawberries' weight, we subtract his dad's strawberries' weight (11 pounds) from the total weight of their strawberries (30 pounds): 30 - 11 = 19. Therefore, Marco's strawberries weighed 19 pounds. The answer is 19.\n",
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            'To find the cost of each book, we subtract the initial amount of money Edward had ($6) from the current amount of money he has ($12) and divide it by the number of books (2): (12 - 6) / 2 = 6 / 2 = 3 Therefore, each book cost $3. The answer is 3.\n',
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            'To find the number of pages in each chapter, we divide the total number of pages in the book (594) by the number of chapters (3): 594 / 3 = 198. Therefore, each chapter has 198 pages. The answer is 198.\n',
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: {question}\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/svamp/test.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'question',
                ],
                output_column='answer',
                test_range='[500:625]'),
            type='opencompass.datasets.SVAMPDataset'),
    ],
]
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
        '/deltadisk/guestnju/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves/meta-llama__Llama-3.2-1B/lora/sft/max_samples_10000',
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel'
    ),
]
work_dir = 'outputs/default/20250601_225830'
