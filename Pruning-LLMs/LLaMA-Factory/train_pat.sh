#!/bin/bash
# YAML_P = "examples/train_lora/llama3.2-1b-instruct-tap0.9-learnable_lora_sft.yaml"
# 设置环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7
#export tap_args='{"tap_enabled": true,"tap_stop_at_steps": 8000,"tap_remain_ratio": 0.9}'

# 执行训练命令
llamafactory-cli train examples/train_lora/llama3.2-3b-base_lora_sft.yaml