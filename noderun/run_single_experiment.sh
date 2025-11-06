#!/bin/bash

# 确保脚本在任何命令失败时立即退出
set -e
# source /home/kris/miniconda3/etc/profile.d/conda.sh
# eval "$(conda shell.bash hook)" 

################################################################################
# 第零步：请在此处配置您的实验参数
################################################################################

CUDA_DEVICES="7"

# 从命令行参数获取配置
if [ "$#" -ne 7 ]; then
    echo "错误: 需要提供7个参数！"
    echo "用法: bash $0 [实验名] [剪枝率] [停止步数] [数据量] [轮数] [批大小] [Trainer类]"
    exit 1
fi
# 实验命名
EXP_NAME="$1"
# 剪枝率 (例如 0.1)
PRUNING_RATE="$2"
# 剪枝过程停止的步数 (例如 1000)
PRUNING_STOP_STEP="$3"
# 数据量
MAX_SAMPLES="$4"
# 轮数
EPOCH="$5"
# batch_size
BATCH_SIZE="$6"
# 方法
TRAINER_CLASS="$7"


# 新增：设置所有实验结果的根目录 (请使用绝对路径)
# 例如：/data/my_llm_experiments
# 模型路径
MODEL_PATH="/data/kris/shared_data/models/Llama-3.2-3B"
BASE_SAVE_DIR="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory/saves"


# 当前脚本目录
CURRENT_DIR=$(pwd)
# 设置您的.yaml模版文件的绝对路径
TEMPLATE_YAML_PATH="${CURRENT_DIR}/examples/train_lora/test.yaml"
# 设置llama-factory项目中 'examples/train_lora' 目录的路径
LLAMA_FACTORY_TRAIN_DIR="${CURRENT_DIR}/examples/train_lora"

################################################################################
# 脚本核心逻辑 -- 通常无需修改以下内容
################################################################################


NEW_YAML_PATH="${LLAMA_FACTORY_TRAIN_DIR}/${NEW_YAML_NAME}"
echo ">>> 第一步：设置CUDA可见设备..."
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export TRAINER=$TRAINER_CLASS
export MODELPATH=$MODEL_PATH
echo "CUDA_VISIBLE_DEVICES 设置为: $CUDA_DEVICES"
echo

echo ">>> 第二步：设置实验名称..."
echo "实验名称为: $EXP_NAME"
echo

echo ">>> 第三步：设置实验保存路径..."
# 根据您设置的 BASE_SAVE_DIR 和实验名称构造本次实验的专属目录
SAVE_PATH="${BASE_SAVE_DIR}/${EXP_NAME}_result"
mkdir -p "$SAVE_PATH"
echo "所有结果将保存在: $SAVE_PATH"
echo

echo ">>> 第四步：根据主路径构造微调和评估结果路径..."
# 微调结果的输出路径 (LoRA模型等)
PEFT_RESULT_PATH="${SAVE_PATH}/${EXP_NAME}_peft_path"
# OpenCompass评估结果的输出路径
EVAL_RESULT_PATH="${SAVE_PATH}/${EXP_NAME}_eval"
mkdir -p "$PEFT_RESULT_PATH"
mkdir -p "$EVAL_RESULT_PATH"
echo "微调模型将保存至: $PEFT_RESULT_PATH"
echo "OpenCompass评估结果将保存至: $EVAL_RESULT_PATH"
echo

echo ">>> 第五步：复制并重命名YAML模版文件..."
if [ ! -f "$TEMPLATE_YAML_PATH" ]; then
    echo "错误: YAML模版文件不存在于 '$TEMPLATE_YAML_PATH'！请检查路径。"
    exit 1
fi
NEW_YAML_NAME="${EXP_NAME}.yaml"
NEW_YAML_PATH="${LLAMA_FACTORY_TRAIN_DIR}/${NEW_YAML_NAME}"
cp "$TEMPLATE_YAML_PATH" "$NEW_YAML_PATH"
echo "已从模版创建新的配置文件: $NEW_YAML_PATH"
echo

echo ">>> 第六步：修改新YAML文件中的参数..."
# 使用sed命令修改文件。使用'#'作为分隔符以避免与路径中的'/'冲突。
# 注意：这里的sed命令会直接修改文件（-i选项）

# 修改 model_path
sed -i "s#^model_name_or_path:.*#model_name_or_path: ${MODEL_PATH}#" "$NEW_YAML_PATH"
echo "  - model_name_or_path 已更新为: ${MODEL_PATH}"

# 修改 output_dir
sed -i "s#^output_dir:.*#output_dir: ${PEFT_RESULT_PATH}#" "$NEW_YAML_PATH"
echo "  - output_dir 已更新为: ${PEFT_RESULT_PATH}"

# 修改 tap_remain_ratio
sed -i "s#^tap_remain_ratio:.*#tap_remain_ratio: ${PRUNING_RATE}#" "$NEW_YAML_PATH"
echo "  - tap_remain_ratio 已更新为: ${PRUNING_RATE}"

# 修改 tap_stop_at_steps
sed -i "s#^tap_stop_at_steps:.*#tap_stop_at_steps: ${PRUNING_STOP_STEP}#" "$NEW_YAML_PATH"
echo "  - tap_stop_at_steps 已更新为: ${PRUNING_STOP_STEP}"

# 修改 max_samples
sed -i "s#^max_samples:.*#max_samples: ${MAX_SAMPLES}#" "$NEW_YAML_PATH"
echo "  - max_samples 已更新为: ${MAX_SAMPLES}"

# 修改 num_train_epochs
sed -i "s#^num_train_epochs:.*#num_train_epochs: ${EPOCH}#" "$NEW_YAML_PATH"
echo "  - num_train_epochs 已更新为: ${EPOCH}"

# 修改 per_device_train_batch_size
sed -i "s#^per_device_train_batch_size:.*#per_device_train_batch_size: ${BATCH_SIZE}#" "$NEW_YAML_PATH"
echo "  - per_device_train_batch_size 已更新为: ${BATCH_SIZE}"
echo

echo ">>> 第七步：开始运行llamafactory训练..."
echo "运行命令: llamafactory-cli train ${NEW_YAML_PATH}"
llamafactory-cli train "$NEW_YAML_PATH"
echo "训练完成！"
echo

EVAL_RESULT_PATH="${SAVE_PATH}/${EXP_NAME}_eval"
EVAL_SCRIPT_PATH="${CURRENT_DIR}/evaluation.sh"

if [ ! -f "$EVAL_SCRIPT_PATH" ]; then
    echo "错误: 评估脚本 'evaluation.sh' 在当前目录 '$CURRENT_DIR' 下未找到！"
    exit 1
fi

echo "正在修改评估脚本: ${EVAL_SCRIPT_PATH}"

sed -i "s#^HF_PATH=.*#HF_PATH=\"${MODEL_PATH}\"#" "$EVAL_SCRIPT_PATH"
echo "  - evaluation.sh 中的 HF_PATH 已更新。"

# 修改PEFT_PATH
# 假设 run_evaluation.sh 中的格式是 PEFT_PATH="..."
sed -i "s#^PEFT_PATH=.*#PEFT_PATH=\"${PEFT_RESULT_PATH}\"#" "$EVAL_SCRIPT_PATH"
echo "  - evaluation.sh 中的 PEFT_PATH 已更新。"

# 修改OpenCompass保存路径
# !!重要假设!!: 假设 run_evaluation.sh 中用于控制输出路径的变量名为 'EVAL_OUTPUT_PATH'
# 如果变量名不同，请修改下面的 'EVAL_OUTPUT_PATH'
sed -i "s#^EVAL_OUTPUT_PATH=.*#EVAL_OUTPUT_PATH=\"${EVAL_RESULT_PATH}\"#" "$EVAL_SCRIPT_PATH"
echo "  - evaluation.sh 中的 EVAL_OUTPUT_PATH 已更新。"
echo

echo "开始运行评估脚本..."
# conda activate opencompass-pat # <--- 新增
echo # <--- 新增
EVAL_CONDA_ENV="opencompass-pat"
# conda run -n "${EVAL_CONDA_ENV}" --no-capture-output --cwd . \
#         bash "$EVAL_SCRIPT_PATH"
# bash "$EVAL_SCRIPT_PATH"

echo ">>> 所有步骤已成功完成！ <<<"
