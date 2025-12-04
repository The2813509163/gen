#!/bin/bash

# 设置参数
HF_TYPE="base"
HF_PATH="/data/kris/shared_data/models/Llama-3.2-3B"
PEFT_PATH="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory/saves/meta-llama__Llama-3.2-3B-tap0.9-learnable-interm/my_awesome_experiment_result/my_awesome_experiment_peft_path"
EVAL_OUTPUT_PATH="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory/saves/meta-llama__Llama-3.2-3B-tap0.9-learnable-interm/my_awesome_experiment_result/my_awesome_experiment_eval"
DATASETS="gsm8k_gen_1d7fe4 math_4shot_base_gen_db136b svamp_gen_fb25e4 piqa_gen_1194eb siqa_gen_18632c squad20_gen_1710bc ARC_c_gen_1e0de5 ARC_e_gen_1e0de5 lambada_gen_217e11"
# DATASETS="lambada_gen_217e11"
BATCH_SIZE=16
MAX_OUT_LEN=512
MAX_NUM_WORKERS=8

export update_step=78111
export tap_args='{"tap_enabled": true,"tap_stop_at_steps": 4500,"tap_remain_ratio": 0.9}'
export learnable_mask=true
export HIO_r=512

# 运行 OpenCompass
opencompass \
    --hf-type $HF_TYPE \
    --hf-path $HF_PATH \
    --peft-path $PEFT_PATH \
    --datasets $DATASETS \
    --batch-size $BATCH_SIZE \
    --max-out-len $MAX_OUT_LEN \
    --max-num-workers $MAX_NUM_WORKERS \
    --work-dir $EVAL_OUTPUT_PATH \
