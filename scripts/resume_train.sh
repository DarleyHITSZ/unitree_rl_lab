#!/bin/bash
# 一键续训脚本：从指定checkpoint继续训练Unitree G1 Walkflat任务

# ======================== 可配置参数 ========================
# 任务名称
TASK_NAME="Unitree-G1-29dof-Walkflat"
# 日志根目录
LOG_ROOT="/home/darley/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_walkflat"
# 续训的时间戳文件夹
RUN_FOLDER="2026-01-20_15-47-56"
# 要加载的checkpoint
CHECKPOINT_PATH="model_1299.pt"
# 训练迭代数
MAX_ITERATIONS=10
# 实验名称
EXPERIMENT_NAME="unitree_g1_29dof_walkflat"
# 指定GPU设备
DEVICE="cuda:0"
# ==========================================================================

# ======================== 路径校验 ========================
echo "===== 开始校验参数 ====="
# 检查checkpoint文件是否存在
if [ ! -f "${LOG_ROOT}/${RUN_FOLDER}/${CHECKPOINT_PATH}" ]; then
    echo "[错误] Checkpoint文件不存在：${CHECKPOINT_PATH}"
    echo "请检查路径是否正确，或确认文件是否存在！"
    exit 1
fi

echo "[成功] 所有路径校验通过！"
echo "任务名称：${TASK_NAME}"
echo "续训文件：${CHECKPOINT_PATH}"
echo "总迭代数：${MAX_ITERATIONS}"
echo "========================"

# ======================== 执行训练命令 ========================
echo -e "\n===== 开始执行续训命令 ====="
# 拼接完整命令
TRAIN_CMD="python scripts/rsl_rl/train.py \
  --task ${TASK_NAME} \
  --resume \
  --load_run ${RUN_FOLDER} \
  --checkpoint ${CHECKPOINT_PATH} \
  --max_iterations ${MAX_ITERATIONS} \
  --experiment_name ${EXPERIMENT_NAME} \
  --logger tensorboard \
  --headless \
  --device ${DEVICE}"

# 打印命令
echo "执行命令：${TRAIN_CMD}"
echo -e "========================\n"

# 执行训练命令
${TRAIN_CMD}

# ======================== 执行结果检查 ========================
if [ $? -eq 0 ]; then
    echo -e "\n===== 训练脚本执行完成！ ====="
else
    echo -e "\n===== 训练脚本执行失败！ ====="
    exit 1
fi
