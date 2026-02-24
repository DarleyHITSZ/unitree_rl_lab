#!/usr/bin/env python3
"""关键维度扫描脚本：18组S/E/B/H组合（固定clip=0.15, lr=3e-4）"""

import argparse
import os
import re
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py",
)
MAX_ITER = str(5000)
TASK = "Unitree-G1-29dof-Walkflat"
CONDA_ENV = "env_isaacsim"

EXPERIMENTS = [
    {"S": 24, "E": 5, "B": 4, "H": 0.01},
    {"S": 16, "E": 5, "B": 4, "H": 0.01},
    {"S": 32, "E": 5, "B": 4, "H": 0.01},
    {"S": 24, "E": 3, "B": 4, "H": 0.01},
    {"S": 24, "E": 8, "B": 4, "H": 0.01},
    {"S": 24, "E": 5, "B": 2, "H": 0.01},
    {"S": 24, "E": 5, "B": 8, "H": 0.01},
    {"S": 24, "E": 5, "B": 4, "H": 0.00},
    {"S": 24, "E": 5, "B": 4, "H": 0.02},
    {"S": 16, "E": 3, "B": 8, "H": 0.00},
    {"S": 16, "E": 3, "B": 2, "H": 0.02},
    {"S": 16, "E": 8, "B": 2, "H": 0.00},
    {"S": 16, "E": 8, "B": 8, "H": 0.02},
    {"S": 32, "E": 3, "B": 2, "H": 0.00},
    {"S": 32, "E": 3, "B": 8, "H": 0.02},
    {"S": 32, "E": 8, "B": 8, "H": 0.00},
    {"S": 32, "E": 8, "B": 2, "H": 0.02},
    {"S": 16, "E": 5, "B": 8, "H": 0.02},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="关键维度扫描脚本：18组S/E/B/H组合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="从指定实验编号开始运行（1-18），默认从第1个开始",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="运行指定实验编号。单个：--run 5，多个逗号分隔：--run 1,3,5",
    )
    return parser.parse_args()


def modify_config_file(num_steps, num_epochs, num_batches, entropy):
    with open(CONFIG_FILE, "r") as f:
        content = f.read()

    content = re.sub(r"(num_steps_per_env\s*=\s*)(\d+)", f"num_steps_per_env={num_steps}", content)
    content = re.sub(r"(num_learning_epochs\s*=\s*)(\d+)", f"num_learning_epochs={num_epochs}", content)
    content = re.sub(r"(num_mini_batches\s*=\s*)(\d+)", f"num_mini_batches={num_batches}", content)
    content = re.sub(r"(entropy_coef\s*=\s*)([\d.e+-]+)", f"entropy_coef={entropy}", content)
    content = re.sub(r"(clip_param\s*=\s*)([\d.e+-]+)", "clip_param=0.15", content)
    content = re.sub(r"(learning_rate\s*=\s*)([\d.e+-]+)", "learning_rate=3.0e-4", content)

    with open(CONFIG_FILE, "w") as f:
        f.write(content)


def run_training():
    cmd = [
        "conda",
        "run",
        "-n",
        CONDA_ENV,
        "--no-capture-output",
        "python",
        "scripts/rsl_rl/train.py",
        "--task",
        TASK,
        "--max_iterations",
        MAX_ITER,
        "--headless",
    ]
    print(f"[{datetime.now()}] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT)


def log_result(exp_num, params, status, log_file, message=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] Exp #{exp_num}: "
        f"S={params['S']} E={params['E']} B={params['B']} H={params['H']} | "
        f"Status: {status}"
    )
    if message:
        log_entry += f" | Message: {message}"
    log_entry += "\n"

    with open(log_file, "a") as f:
        f.write(log_entry)


if __name__ == "__main__":
    args = parse_args()

    experiment_indices = None

    if args.run is not None:
        try:
            exp_nums = [int(x.strip()) for x in args.run.split(",")]
            invalid_nums = [n for n in exp_nums if n < 1 or n > len(EXPERIMENTS)]
            if invalid_nums:
                print(f"[ERROR] 无效的实验编号: {invalid_nums}")
                print(f"[INFO] 有效范围: 1-{len(EXPERIMENTS)}")
                exit(1)
            experiment_indices = [n - 1 for n in exp_nums]
            print(f"[INFO] 将运行实验: {exp_nums}")
        except ValueError:
            print(f"[ERROR] --run 参数格式错误，请使用逗号分隔的数字，例如: --run 1,3,5")
            exit(1)

    elif args.start is not None:
        if args.start < 1 or args.start > len(EXPERIMENTS):
            print(f"[ERROR] 起始编号必须在 1-{len(EXPERIMENTS)} 范围内")
            exit(1)
        experiment_indices = list(range(args.start - 1, len(EXPERIMENTS)))
        print(f"[INFO] 从实验 {args.start} 开始运行")

    else:
        experiment_indices = list(range(len(EXPERIMENTS)))
        print(f"[INFO] 运行全部 {len(EXPERIMENTS)} 个实验")

    experiments_to_run = [EXPERIMENTS[i] for i in experiment_indices]

    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    log_file = os.path.join(PROJECT_ROOT, f"logs/key_dim_sweep_{timestamp}.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "w") as f:
        f.write(f"=== Key-Dimension Hyperparameter Sweep Log ===\n")
        f.write(f"Task: {TASK}\n")
        f.write(f"Max Iterations: {MAX_ITER}\n")
        f.write(f"Fixed: clip_param=0.15, learning_rate=3e-4\n")
        f.write(f"Total Experiments: {len(experiments_to_run)}\n")
        f.write(f"Experiment IDs: {[i + 1 for i in experiment_indices]}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")

    success_count = 0
    failure_count = 0

    for exp_idx in experiment_indices:
        idx = exp_idx + 1
        params = EXPERIMENTS[exp_idx]
        S = params["S"]
        E = params["E"]
        B = params["B"]
        H = params["H"]

        exp_name = f"exp{idx}_S{S}_E{E}_B{B}_H{H}"

        print(f"\n{'='*60}")
        print(f"Experiment #{idx}/{len(experiments_to_run)}: {exp_name}")
        print(f"{'='*60}")
        print(f"Parameters: S={S}, E={E}, B={B}, H={H}")
        print(f"Fixed: clip_param=0.15, learning_rate=3e-4")

        modify_config_file(S, E, B, H)
        result = run_training()

        if result.returncode == 0:
            success_count += 1
            log_result(idx, params, "SUCCESS", log_file)
            print(f"[OK] Completed: {exp_name}")
        else:
            failure_count += 1
            log_result(idx, params, "FAILED", log_file, f"returncode={result.returncode}")
            print(f"[FAIL] {exp_name} failed with return code {result.returncode}")
            print(f"[INFO] Continuing to next experiment...")

    summary = (
        f"\n{'='*60}\n"
        f"Sweep Complete!\n"
        f"Total: {len(experiments_to_run)} | Success: {success_count} | Failed: {failure_count}\n"
        f"Ended: {datetime.now()}\n"
        f"{'='*60}\n"
    )
    print(summary)

    with open(log_file, "a") as f:
        f.write(summary)

    print(f"\nResults logged to: {log_file}")
