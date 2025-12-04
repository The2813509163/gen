#!/bin/bash
CONFIG_FILE="experiments.conf"
LOG_DIR="sbatch_logs"
mkdir -p "$LOG_DIR"

while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^\s*# ]] || [[ -z "$line" ]]; then
        continue
    fi

    # 注意：不再需要 node 列，调度器会自动选择
    read -r exp_name pruning_rate stop_step max_samples epoch batch_size trainer_class <<< "$line"

    echo "准备提交实验: [${exp_name}]"

    # 使用 sbatch 提交作业，并通过 --export 或作为参数传递变量
    sbatch \
      --job-name="${exp_name}" \
      --output="${LOG_DIR}/${exp_name}_%j.out" \
      run_sbatch_experiment.sh \
      "$exp_name" \
      "$pruning_rate" \
      "$stop_step" \
      "$max_samples" \
      "$epoch" \
      "$batch_size" \
      "$trainer_class"
done < "$CONFIG_FILE"

echo ">>> 所有实验已提交给作业调度系统。"
echo ">>> 使用 'squeue -u $USER' 命令查看状态。"