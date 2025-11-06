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
pip install -e LLaMA-Factory
```

## factory-pat（更新）

```
cd ./Pruning-LLMs
conda create -n factory-pat python=3.10 -y
conda activate factory-pat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "flash-attn==2.5.5" --no-build-isolation
conda install -c conda-forge pyarrow sentencepiece 
pip install -e thirdparty/transformers-4.51.1
pip install -e thirdparty/peft-0.15.1
pip install tensorboard
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

## 路径修改

1.test_Pruning/LLaMA-Factory/data/dataset_info.json文件中的修改nvidia__OpenMathInstruct中的hf_hub_url

####  还有Pruning-LLMs/LLaMA-Factory/data/dataset_info.json文件中的修改nvidia__OpenMathInstruct中的hf_hub_url

<img width="849" height="297" alt="屏幕截图 2025-11-01 212559" src="https://github.com/user-attachments/assets/d360db7e-9b07-4645-bee5-ba5bc5f7dbcb" />


2.test_Pruning/thirdparty/transformers-4.51.1/src/transformers/trainer.py中的路径（红色下划线），改dummy数据集就行了，基础模型路径已更新成modelpath参数传参

<img width="1312" height="896" alt="屏幕截图 2025-11-01 212040" src="https://github.com/user-attachments/assets/9ac65444-94bf-4349-ab3c-485cae36f886" />



#### 3.miniconda3/envs/opencompass-/lib/python3.10/site-packages/opencompass/utils/datasets.py中DEFAULT_DATA_FOLDER,直接设置成本地opencompass数据集路径

<img width="1020" height="111" alt="image" src="https://github.com/user-attachments/assets/96123121-f638-4290-9e7c-76443136257a" />


4../noderun/run_single_experiment.sh设置基础模型路径和实验结果保存路径(该目录下将会存放peft路径和eva路径)


<img width="1047" height="115" alt="屏幕截图 2025-11-06 232109" src="https://github.com/user-attachments/assets/5fc37111-e2a0-4bd7-bcc8-24f6b9b0f350" />


5../noderun/run_all_experiments.sh设置两个绝对路径，第一个设置成Pruning-LLMs/LLaMA-Factory，另一个设置成test_Pruning/LLaMA-Factory

<img width="1270" height="105" alt="屏幕截图 2025-11-06 232439" src="https://github.com/user-attachments/assets/f30c85a3-08e9-44f1-9c2c-a9e6d73bd44b" />



## 实验设置

1.打开./noderun/experiments.conf进行实验设置，目前的Trainer类有三种，分别是“Trainer”、“Super2Trainer”、“CustomTrainer”,其中"Trainer"是最原始的方法

<img width="1168" height="385" alt="屏幕截图 2025-11-06 233645" src="https://github.com/user-attachments/assets/1e0b6dcf-2a18-4a20-b6bb-ebbda341c296" />


2.在./noderun/run_single_experiment.sh中进行gpu、基础模型的设置等

<img width="1047" height="115" alt="屏幕截图 2025-11-06 232109" src="https://github.com/user-attachments/assets/3b82fd77-5c97-4592-af6c-a6599de85463" />


3.一键运行，即可完成训练+评估一套连招

```
cd ./noderun
./run_all_experiments.sh
```

