#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAINER="Super2Trainer"
llamafactory-cli train examples/train_lora/llama3.1-8b-base_lora_sft.yaml

