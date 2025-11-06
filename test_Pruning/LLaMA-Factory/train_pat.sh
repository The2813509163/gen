#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export MODELPATH="/data/home/duyuanProfV2/workspace/qianxuzhen/data/kris/shared_data/models/Llama-3.2-3B"
llamafactory-cli train examples/train_lora/test.yaml


