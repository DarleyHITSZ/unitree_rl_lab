# AGENTS.md - Unitree RL Lab Development Guide

This document provides guidelines for AI agents working on the Unitree RL Lab codebase.

## Project Overview

Unitree RL Lab is a reinforcement learning framework for Unitree robots (Go2, H1, G1-29dof), built on IsaacLab and Isaac Sim. It contains locomotion/mimic environments, RSL-RL integration, and Sim2sim/Sim2real pipelines.

## Essential Commands

```bash
# Installation & Setup
./unitree_rl_lab.sh -i                    # Install in editable mode
pip install -e source/unitree_rl_lab/      # Direct pip install

# Environment Management
./unitree_rl_lab.sh -l                    # List available environments
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity  # Train (headless)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity  # Play/visualize

# Manual Commands
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

## Build, Lint & Test Commands

```bash
# Run all pre-commit hooks (lint + format + spell check)
pre-commit run --all-files

# Run specific linters
pre-commit run black --all-files              # Format code
pre-commit run flake8 --all-files             # Lint code (with flake8-simplify, flake8-return)
pre-commit run isort --all-files              # Sort imports
pre-commit run pyupgrade --all-files          # Upgrade Python syntax
pre-commit run codespell --all-files          # Spell check
pre-commit run rst-directive-colons --all-files  # RST formatting

# Manual linting/formatting
black --line-length 120 --preview <files>
flake8 <files>
isort --profile black --filter-files <files>
pyright <files>

# Type checking
pyright source/unitree_rl_lab/

# Test training (manual verification - no formal unit tests)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity
# Run play/inference to visualize trained policies
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

Note: Pre-commit excludes `deploy/` and `.vscode/` directories.

## Code Style

### Imports (in order)
1. FUTURE → 2. STDLIB (including numpy, torch, gymnasium, scipy) → 3. THIRDPARTY (isaacsim, pxr, omni) → 4. ISAACLABPARTY (isaaclab, isaaclab_tasks) → 5. FIRSTPARTY (unitree_rl_lab) → 6. LOCALFOLDER

```python
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import gymnasium as gym
import isaaclab.sim as sim_utils
from unitree_rl_lab.tasks.locomotion import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
```

### Formatting
- **Line length**: 120 characters
- **Formatter**: Black with `--preview` flag
- **Indentation**: 4 spaces (no tabs)
- **Python version**: 3.10+ (pyupgrade enforces --py37-plus)
- **Line breaks**: Break before binary operators (W503 ignored)
- **List formatting**: Use `# fmt: off` / `# fmt: on` around multi-line lists requiring custom formatting

### Type Annotations
- **Type checker**: pyright (`typeCheckingMode = "basic"`, Python 3.10)
- **Syntax**: Use `int | float` not `Union[int, float]`
- **Configs**: Use `@configclass` decorator from IsaacLab
- Ignore rules: `reportGeneralTypeIssues`, `reportMissingImports`, `reportMissingModuleSource` (CI compatibility)

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `RobotEnvCfg` |
| Config classes | PascalCase + `Cfg` | `RobotSceneCfg` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Functions/variables | snake_case | `reset_joints()` |
| Private methods | `_snake_case()` | `_process_obs()` |
| Files | snake_case | `velocity_env_cfg.py` |

### Docstrings
- **Style**: Google style
- **Required for**: All public classes, functions, methods
- **First line**: Imperative mood

```python
def compute_reward(self) -> float:
    """Compute the reward for the current timestep.

    Args:
        self: Robot state with joint positions and velocities.

    Returns:
        The computed reward value.
    """
```

### Error Handling
- Use specific exception types (no bare `except:`)
- Include informative error messages
- Let exceptions propagate for configuration validation

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for training.")
```

### Configuration Classes
```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the robot environment."""
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
```

### MDP Components
Organize into separate classes: `SceneCfg`, `EventCfg`, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `RewardsCfg`, `TerminationsCfg`, `CurriculumCfg`.

### File Structure
```
source/unitree_rl_lab/unitree_rl_lab/
├── assets/           # Robot configurations
├── tasks/
│   ├── locomotion/   # Locomotion environments
│   └── mimic/        # Motion capture tracking
└── utils/            # Utility functions
```

## Commit Style

Follow conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(locomotion): add velocity tracking reward for G1 robot`

## Common Issues
1. **Import errors**: Ensure Isaac Sim/Lab is activated
2. **CUDA errors**: Training requires CUDA-compatible GPU
3. **RSL-RL**: Version 2.3.1+ required for distributed training
4. **Missing robots**: Set `UNITREE_MODEL_DIR` or `UNITREE_ROS_DIR`
