#!/bin/bash

# 主控脚本，用于在多台机器上并行运行多个实验

# 确保配置文件存在
CONFIG_FILE="experiments.conf"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 实验配置文件 '$CONFIG_FILE' 未找到！"
    exit 1
fi

################################################################################
# 第一步：定义实验配置
################################################################################



# --- 定义不同实验类型的基础路径 (逻辑保留) ---
PRUNING_REMOTE_WORKDIR="/home/kris/workspace/qianxuzhen/Pruning-LLMs/LLaMA-Factory"
DUMMY_REMOTE_WORKDIR="/home/kris/workspace/qianxuzhen/test/gen/test_Pruning/LLaMA-Factory"

# --- 变化点：统一的工作脚本 ---
WORKER_SCRIPT="run_single_experiment.sh"
# 共用评估脚本
EVAL_SCRIPT="evaluation.sh"

# 本地日志目录
LOG_DIR="logs_master"
mkdir -p "$LOG_DIR"

echo ">>> 开始分发并执行实验..."
echo ">>> 本地 '${LOG_DIR}' 目录将记录启动信息"
echo

# 读取配置文件并循环执行
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过注释和空行
    if [[ "$line" =~ ^\s*# ]] || [[ -z "$line" ]]; then
        continue
    fi

    # 解析配置
    read -r node exp_name pruning_rate stop_step max_samples epoch batch_size trainer_class <<< "$line"

    # 根据 trainer_class 选择配置和 Conda 环境
    remote_workdir=""
    CONDA_ENV_NAME=""

    if [[ "$trainer_class" == "Trainer" ]]; then
        echo "--------------------------------------------------"
        echo "识别到 PRUNING 实验 (Trainer: ${trainer_class}): [${exp_name}]"
        remote_workdir="$PRUNING_REMOTE_WORKDIR"
        CONDA_ENV_NAME="factory-pat"
    else
        echo "--------------------------------------------------"
        echo "识别到 DUMMY 实验 (Trainer: ${trainer_class}): [${exp_name}]"
        remote_workdir="$DUMMY_REMOTE_WORKDIR"
        CONDA_ENV_NAME="try"
    fi

    echo "准备在机器 [${node}] 上启动"
    echo "   -> 将使用的 Conda 环境: ${CONDA_ENV_NAME}"

    # 1. 检查本地脚本是否存在
    # --- 变化点：检查统一的 WORKER_SCRIPT ---
    if [ ! -f "$WORKER_SCRIPT" ] || [ ! -f "$EVAL_SCRIPT" ]; then
        echo "   -> 错误: 必需的脚本 ('$WORKER_SCRIPT' 或 '$EVAL_SCRIPT') 在本地不存在！"
        continue
    fi

    # 2. 远程创建目录
    ssh "$node" "mkdir -p ${remote_workdir}"

    # 3. 同步脚本
    # --- 变化点：同步统一的 WORKER_SCRIPT ---
    echo "   -> 同步脚本 ${WORKER_SCRIPT} 和 ${EVAL_SCRIPT} 到 ${node}:${remote_workdir}/"
    rsync -avz "$WORKER_SCRIPT" "$EVAL_SCRIPT" "${node}:${remote_workdir}/"

    # 4. 以后台模式执行远程脚本
    REMOTE_LOG_FILE="${remote_workdir}/${exp_name}.log"
    echo "   -> 启动远程脚本... 远程日志将保存在: ${node}:${REMOTE_LOG_FILE}"

    # ************************** 核心命令部分 **************************
    COMMAND_TO_RUN="
        source ~/.zshrc && \\
        # 1. 进入正确的工作目录
        cd ${remote_workdir} && \\

        # 2. 使用 conda run 在指定环境中，通过 nohup 启动工人脚本
        # --- 变化点：执行统一的 WORKER_SCRIPT ---
        conda run -n ${CONDA_ENV_NAME} --no-capture-output \\
            nohup bash ./${WORKER_SCRIPT} \\
                '${exp_name}' \\
                '${pruning_rate}' \\
                '${stop_step}' \\
                '${max_samples}' \\
                '${epoch}' \\
                '${batch_size}' \\
                '${trainer_class}' \\
                > ${REMOTE_LOG_FILE} 2>&1
    "

    # 发送远程执行命令
    ssh "$node" "zsh -l -c \"$COMMAND_TO_RUN\"" \
        > "${LOG_DIR}/${node}_${exp_name}_launch.log" 2>&1 &
    # ******************************************************************

    echo "   -> 启动命令已发送。"
    echo "--------------------------------------------------"
    echo

done < "$CONFIG_FILE"

echo ">>> 所有实验启动命令已发送。"
echo ">>> 等待所有后台任务启动..."

wait

echo
echo ">>> 所有启动进程已在后台运行！"
echo ">>> 请登录到目标机器检查实验状态和日志。"