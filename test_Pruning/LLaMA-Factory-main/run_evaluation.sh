#!/bin/bash

# 设置参数
HF_TYPE="base"
# HF_PATH="/data/kris/shared_data/models/Llama-3.2-1B-Instruct"
HF_PATH="/data/kris/shared_data/models/Llama-3.2-3B"
PEFT_PATH="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves/meta-llama__Llama-3.2-3B-base/max_samples_zipf_dummy_1M_alpha_0.005_T_4"
# PEFT_PATH="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves/meta-llama__Llama-3.2-3B/"

# DATASETS="gsm8k_0shot_v2_gen_17d799 math_0shot_gen_11c4b5 svamp_gen_fb25e4 piqa_gen_1194eb siqa_gen_18632c squad20_gen_1710bc ARC_c_gen_1e0de5 ARC_e_gen_1e0de5"
# DATASETS="gsm8k_gen_17d0dc"
# DATASETS="gsm8k_gen_1d7fe4 math_4shot_base_gen_db136b"
DATASETS="gsm8k_gen_1d7fe4 math_4shot_base_gen_db136b svamp_gen_fb25e4 piqa_gen_1194eb siqa_gen_18632c squad20_gen_1710bc ARC_c_gen_1e0de5 ARC_e_gen_1e0de5 lambada_gen_217e11"
# DATASETS="ARC_c_gen_1e0de5"
# DATASETS="lambada_gen_217e11"
#DATASETS="svamp_gen_fb25e4"
#DATASETS="siqa_gen_18632c"
#DATASETS="piqa_gen_1194eb squad20_gen_1710bc ARC_c_gen_1e0de5 ARC_e_gen_1e0de5 mmlu_gen_23a9a9"
BATCH_SIZE=16
MAX_OUT_LEN=512
MAX_NUM_WORKERS=8
CUDA_VISIBLE_DEVICES="4,5,6,7"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# export update_step=78111
# export tap_args='{"tap_enabled": true,"tap_stop_at_steps": 9000,"tap_remain_ratio": 0.9}'
# export learnable_mask=true
# export HIO_r=512

# 运行 OpenCompass
opencompass \
    --hf-type $HF_TYPE \
    --hf-path $HF_PATH \
    --peft-path $PEFT_PATH \
    --datasets $DATASETS \
    --batch-size $BATCH_SIZE \
    --max-out-len $MAX_OUT_LEN \
    --max-num-workers $MAX_NUM_WORKERS \