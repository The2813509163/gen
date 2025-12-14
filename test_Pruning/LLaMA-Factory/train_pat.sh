#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAINER="Super2Trainer"
export WANDB_PROJECT="Llama3-Experiment" 
export WANDB_RUN_NAME="1214-test-pruning"
llamafactory-cli train examples/train_lora/llama3.1-8b-base_lora_sft.yaml

