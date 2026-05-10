#!/bin/bash
# 一键新训练脚本：启动任务的全新训练

# ======================== 可配置参数 ========================
# 任务名称
TASK_NAME="Unitree-G1-29dof-AdaptiveVelocity"
# 训练迭代数
MAX_ITERATIONS=20000
# 指定GPU设备
DEVICE="cuda:0"
# ======================== 执行训练命令 ========================
echo -e "\n===== 开始执行新训练命令 ====="
# 拼接完整命令（移除续训相关参数）
TRAIN_CMD="python scripts/rsl_rl/train.py \
  --task ${TASK_NAME} \
  --max_iterations ${MAX_ITERATIONS} \
  --headless \
  --logger tensorboard \
  --device ${DEVICE}"

# 打印命令
echo "执行命令：${TRAIN_CMD}"
echo -e "========================\n"

# 执行训练命令
${TRAIN_CMD}

# ======================== 执行结果检查 ========================
if [ $? -eq 0 ]; then
    echo -e "\n===== 新训练脚本执行完成！ ====="
else
    echo -e "\n===== 新训练脚本执行失败！ ====="
    exit 1
fi
