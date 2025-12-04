#!/bin/bash
#SBATCH --nodes=1               # 请求1个节点
#SBATCH --ntasks-per-node=1     # 每个节点运行1个任务
#SBATCH --cpus-per-task=8       # 为这个任务请求8个CPU核心
#SBATCH --mem=32G               # 请求32GB内存
#SBATCH --partition=gpu         # (可选) 请求在GPU分区上运行
#SBATCH --gpus-per-task=1       # (可选) 请求1个GPU
#SBATCH --job-name=llm_pruning_exp # (推荐) 给作业一个名字

# 接收从主控脚本传来的参数
EXP_NAME=$1
PRUNING_RATE=$2
STOP_STEP=$3
MAX_SAMPLES=$4
EPOCH=$5
BATCH_SIZE=$6
TRAINER_CLASS=$7

# --- 定义脚本和工作目录 ---

# 这是您存放 master_submit.sh, run_sbatch_experiment.sh 等脚本的地方
# 请根据实际情况修改此路径
MY_CONTROLLER_SCRIPT_DIR="/home/kris/workspace/qianxuzhen/noderun"

# 根据 trainer_class 判断 LLaMA-Factory 的路径和 Conda 环境
if [[ "$TRAINER_CLASS" == "Trainer" ]]; then
    REMOTE_WORKDIR="/home/kris/workspace/qianxuzhen/Pruning-LLMs/LLaMA-Factory"
    CONDA_ENV_NAME="factory-pat"
else
    REMOTE_WORKDIR="/home/kris/workspace/qianxuzhen/test/gen/test_Pruning/LLaMA-Factory"
    CONDA_ENV_NAME="test-pat"
fi

# --- 关键步骤：复制控制脚本到工作目录 ---
# 根据您的要求，我们将需要的脚本复制到工作目录中，以确保 current_dir 的正确性
# 使用 -f 参数强制覆盖，确保每次运行的都是最新版的控制脚本
echo "正在将控制脚本复制到工作目录: ${REMOTE_WORKDIR}"
cp -f "${MY_CONTROLLER_SCRIPT_DIR}/run_single_experiment.sh" "${REMOTE_WORKDIR}/"
cp -f "${MY_CONTROLLER_SCRIPT_DIR}/evaluation.sh" "${REMOTE_WORKDIR}/"
echo "复制完成。"

# --- 激活环境并执行 ---

# 1. 进入 LLaMA-Factory 的工作目录
# 这是必须的，这样后续脚本中的 `pwd` 或 `./` 才能解析到正确的路径
cd ${REMOTE_WORKDIR}

# 2. 激活 Conda 环境
# 注意：根据集群配置，可能需要 'source /path/to/anaconda/etc/profile.d/conda.sh'
source ~/.zshrc 

echo "--------------------------------------------------------"
echo "作业ID: $SLURM_JOB_ID"
echo "在计算节点 $(hostname) 上运行实验: ${EXP_NAME}"
echo "当前工作目录 (pwd): $(pwd)"
echo "使用的 Conda 环境: ${CONDA_ENV_NAME}"
echo "--------------------------------------------------------"

# 3. 执行核心实验脚本
# 现在可以安全地使用 ./ 调用，因为它已经被复制到当前目录了
conda run -n ${CONDA_ENV_NAME} --no-capture-output \
    bash ./run_single_experiment.sh \
        "${EXP_NAME}" \
        "${PRUNING_RATE}" \
        "${STOP_STEP}" \
        "${MAX_SAMPLES}" \
        "${EPOCH}" \
        "${BATCH_SIZE}" \
        "${TRAINER_CLASS}"

echo "实验 ${EXP_NAME} 在节点 $(hostname) 上运行完成。"
