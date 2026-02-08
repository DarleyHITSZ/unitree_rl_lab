#!/usr/bin/env python3
"""超参数搜索脚本：遍历clip_param和learning_rate的9种组合"""

import re
import subprocess
import sys
from datetime import datetime

CONFIG_FILE = "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
CLIP_PARAMS = [0.15, 0.2, 0.3]
LEARNING_RATES = [3e-4, 1e-3, 1e-4]
MAX_ITER = str(5000)
TASK = "Unitree-G1-29dof-Walkflat"
CONDA_ENV = "env_isaacsim"

def modify_config_file(clip_param, learning_rate):
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    content = re.sub(
        r'(clip_param=)([\d.e+-]+)',
        f'clip_param={clip_param}',
        content
    )
    content = re.sub(
        r'(learning_rate=)([\d.e+-]+)',
        f'learning_rate={learning_rate}',
        content
    )
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)

def run_training():
    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "--no-capture-output",
        "python", "scripts/rsl_rl/train.py",
        "--task", TASK,
        "--max_iterations", MAX_ITER,
        "--headless",
    ]
    print(f"[{datetime.now()}] Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

if __name__ == "__main__":
    for clip_param in CLIP_PARAMS:
        for learning_rate in LEARNING_RATES:
            name = f"{TASK}_clip{clip_param}_lr{learning_rate}"
            print(f"\n{'='*60}")
            print(f"Experiment: {name}")
            print(f"{'='*60}")

            modify_config_file(clip_param, learning_rate)
            result = run_training()

            if result.returncode != 0:
                print(f"Failed: {name}")
                sys.exit(1)

            print(f"Completed: {name}")
