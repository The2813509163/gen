# 环境配置

## test-pat

```
cd ./test_Pruning
conda create -n test-pat python=3.10 -y
conda activate test-pat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "flash-attn==2.5.5" --no-build-isolation
conda install -c conda-forge pyarrow sentencepiece 
pip install -e thirdparty/transformers-4.51.1
pip install -e thirdparty/peft-0.15.1
pip install tensorboard
pip install wandb
pip install -e LLaMA-Factory
```



## opencompass-pat

```
cd ./test_Pruning
conda create -n opencompass-pat python=3.10 -y
conda activate opencompass-pat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "flash-attn==2.5.5" --no-build-isolation
conda install -c conda-forge pyarrow sentencepiece 
pip install opencompass
pip install -e thirdparty/transformers-4.51.1
pip install -e thirdparty/peft-0.15.1
```



# 训练步骤

## 训练及评估数据集路径修改

1.test_Pruning/LLaMA-Factory/data/dataset_info.json文件中的修改nvidia__OpenMathInstruct中的hf_hub_url

####  还有Pruning-LLMs/LLaMA-Factory/data/dataset_info.json文件中的修改nvidia__OpenMathInstruct中的hf_hub_url

<img width="849" height="297" alt="屏幕截图 2025-11-01 212559" src="https://github.com/user-attachments/assets/d360db7e-9b07-4645-bee5-ba5bc5f7dbcb" />


2.miniconda3/envs/opencompass-/lib/python3.10/site-packages/opencompass/utils/datasets.py中DEFAULT_DATA_FOLDER,直接设置成本地opencompass数据集路径

<img width="1020" height="111" alt="image" src="https://github.com/user-attachments/assets/96123121-f638-4290-9e7c-76443136257a" />



## 实验设置
实验设置在yaml文件里完成，yaml文件需放在./test_Pruning/LLaMA-Factory/example/train_lora下

yaml文件中的主要实验设置如下，

| 参数名                      | 内容                                                 |
| --------------------------- | ---------------------------------------------------- |
| model_name_or_path          | 基础模型路径                                         |
| output_dir                  | 结果保存路径                                         |
| tap_stop_at_steps           | 剪枝停止步数                                         |
| tap_remain_ratio            | 剪枝保留率                                           |
| max_samples                 | 训练样本条数                                         |
| num_train_epochs            | 训练轮数                                             |
| dataset                     | sft训练数据集                                        |
| save_steps                  | 保存步数间隔                                         |
| report_to                   | wandb                                                |
| per_device_train_batch_size | sft数据批大小                                        |
| resume_from_checkpoint      | 恢复训练开关                                         |
| dummy_batch_size            | dummy数据批大小                                      |
| teacher_model_path          | 教师模型路径（一般就是基础模型）                     |
| dummy_dataset_path          | dummy数据集路径                                      |
| alpha_schedule              | 列表，每一行两个数分别是step_multiplier和alpha_value |

选择Trainer通过设置环境变量TRAINER来实现，可设置成Trainer或者Super2Trainer

训练脚本可参考./test_Pruning/LLaMA-Factory/train_pat.sh

```
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAINER="Super2Trainer"
export WANDB_PROJECT="Llama3-Experiment"
export WANDB_RUN_NAME="1214-test-pruning"
llamafactory-cli train examples/train_lora/llama3.1-8b-base_lora_sft.yaml
```






