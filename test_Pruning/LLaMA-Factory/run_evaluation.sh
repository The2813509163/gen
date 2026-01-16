#!/bin/bash

# 设置参数
HF_TYPE="base"
HF_PATH="meta-llama/Llama-3.1-8B"
PEFT_PATH="/lustre/fsw/portfolios/edgeai/users/yilzhao/CODES/gen/tmp/outputs/main/0107-1"
DATASETS="gsm8k_gen_1d7fe4 math_4shot_base_gen_db136b svamp_gen_fb25e4 piqa_gen_1194eb siqa_gen_18632c squad20_gen_1710bc ARC_c_gen_1e0de5 ARC_e_gen_1e0de5 lambada_gen_217e11_mmlu_ppl"
BATCH_SIZE=16
MAX_OUT_LEN=512
MAX_NUM_WORKERS=8
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export update_step=78111
export tap_args='{"tap_enabled": true,"tap_stop_at_steps": 16000,"tap_remain_ratio": 0.9}'
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
