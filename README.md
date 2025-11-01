# 环境配置

## test-pat

```
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

## opencompass-pat

```
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

<img width="1312" height="896" alt="屏幕截图 2025-11-01 212040" src="https://github.com/user-attachments/assets/ca755d91-417b-4ed3-9ebc-199bd94499ca" />


2.test_Pruning/thirdparty/transformers-4.51.1/src/transformers/trainer.py中的两个路径（红色下划线），一个是dummy数据集，另一个就是基础模型路径

<img width="849" height="297" alt="屏幕截图 2025-11-01 212559" src="https://github.com/user-attachments/assets/308d63bc-f949-4309-9107-4a5268c94f16" />


3.test_Pruning/LLaMA-Factory/test.yaml文件中的model_name_or_path , output_dir

<img width="1671" height="119" alt="屏幕截图 2025-11-01 212727" src="https://github.com/user-attachments/assets/3a7d550f-7faa-4351-bf09-257900998d45" />



## 训练命令

```
cd test_Pruning/LLaMA-Factory
./train_pat.sh
```

