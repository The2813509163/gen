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



2.test_Pruning/thirdparty/transformers-4.51.1/src/transformers/trainer.py中的两个路径（红色下划线），一个是dummy数据集，另一个就是基础模型路径



3.test_Pruning/LLaMA-Factory/test.yaml文件中的model_name_or_path , output_dir




## 训练命令

```
cd test_Pruning/LLaMA-Factory
./train_pat.sh
```

