# AGENTS.md - Unitree RL Lab Development Guide

Guidelines for AI agents working on the Unitree RL Lab codebase - a reinforcement learning framework for Unitree robots (Go2, H1, G1-29dof) built on IsaacLab and Isaac Sim.

## Essential Commands

```bash
# Installation
./unitree_rl_lab.sh -i                    # Install in editable mode
pip install -e source/unitree_rl_lab/

# Environments
./unitree_rl_lab.sh -l                    # List environments
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity  # Train (headless)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity  # Play/visualize

# Manual training
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

## Build, Lint & Test Commands

```bash
# All pre-commit hooks (run before committing)
pre-commit run --all-files

# Run on specific files
pre-commit run black --files path/to/file.py
pre-commit run flake8 --files path/to/file.py
pre-commit run isort --files path/to/file.py

# Individual linters (single file)
black --line-length 120 --preview source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py
flake8 source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py
isort --profile black --filter-files source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py
pyright source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py

# Type check entire package
pyright source/unitree_rl_lab/

# Spell check
pre-commit run codespell --all-files
```

**Note**: This project has no traditional unit tests. Verification is done through training/play scripts:
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 64 --max_iterations 10
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

Pre-commit excludes `deploy/` and `.vscode/`.

## Code Style

### Imports (ordered top-to-bottom, separated by blank lines)
```python
# 1. FUTURE
from __future__ import annotations

# 2. STDLIB
import math
import os
from typing import TYPE_CHECKING

# 3. THIRDPARTY (torch, numpy, gymnasium treated as stdlib per pyproject.toml)
import gymnasium as gym
import torch

# 4. ISAACLABPARTY
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 5. FIRSTPARTY
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# 6. LOCALFOLDER (config)
# (if applicable)

# TYPE_CHECKING imports at end
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
```

### Formatting
- **Line length**: 120
- **Formatter**: Black `--preview`
- **Indentation**: 4 spaces (no tabs)
- **Python**: 3.10+ (pyupgrade `--py37-plus` - note: project uses this but targets 3.10)
- **Line breaks**: Before binary operators (W503 ignored)
- **Custom lists**: Wrap with `# fmt: off` / `# fmt: on`
- **Imports**: isort with `--profile black --filter-files`

### Type Annotations
- **Checker**: pyright (`typeCheckingMode = "basic"`, Python 3.10)
- **Syntax**: `int | float` (not `Union[int, float]`)
- **Configs**: `@configclass` from IsaacLab
- **Ignored**: `reportGeneralTypeIssues`, `reportMissingImports`, `reportMissingModuleSource`

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `RobotEnvCfg` |
| Config classes | PascalCase + `Cfg` | `RobotSceneCfg`, `ObservationsCfg` |
| Constants | UPPER_SNAKE_CASE | `UNITREE_MODEL_DIR`, `NATURAL_FREQ` |
| Functions/variables | snake_case | `reset_joints()`, `base_reward` |
| Private methods | `_snake_case()` | `_process_obs()` |
| Files | snake_case | `velocity_env_cfg.py` |
| Module-level singletons | UPPER_SNAKE_CASE | `UNITREE_G1_29DOF_CFG` |

### Docstrings (Google style, imperative mood)
```python
def track_lin_vel_xy_adaptive(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Adaptive linear velocity tracking reward based on terrain type.

    On flat terrain: weight = 1.0
    On slope terrain: weight = 0.5 (reduced to prioritize stability).

    Args:
        env: The environment instance.
        command_name: Name of the velocity command to track.
        std: Standard deviation for exponential reward kernel.

    Returns:
        Adaptive velocity tracking reward for each environment.
    """
```

### Error Handling
- Specific exception types (no bare `except:`)
- Use `RuntimeError` for configuration/state errors
- Informative error messages with context

```python
# Good: Specific exception with context
if contact_sensor.cfg.track_air_time is False:
    raise RuntimeError("Activate ContactSensor's track_air_time!")

# Good: Validation with clear message
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for training.")
```

### Configuration Classes
```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
```

### MDP Components
Environment configs organize into these @configclass sections:
- `RobotSceneCfg`: Terrain, robot, sensors, lights
- `EventCfg`: Randomization (startup/reset/interval modes)
- `CommandsCfg`: Velocity commands
- `ActionsCfg`: Joint position actions
- `ObservationsCfg`: Policy and critic observation groups
- `RewardsCfg`: Task rewards, regularization penalties
- `TerminationsCfg`: Episode termination conditions
- `CurriculumCfg`: Curriculum learning terms

### File Structure
```
source/unitree_rl_lab/unitree_rl_lab/
├── __init__.py
├── assets/robots/           # Robot configurations (unitree.py, unitree_actuators.py)
├── tasks/
│   ├── __init__.py
│   ├── locomotion/          # Velocity-tracking environments
│   │   ├── __init__.py
│   │   ├── agents/          # PPO agent configs
│   │   ├── mdp/             # Rewards, observations, commands, curriculums
│   │   └── robots/{go2,h1,g1}/  # Per-robot environment configs
│   └── mimic/               # Motion capture tracking
└── utils/                   # export_deploy_cfg, parser_cfg
```

## Commit Style

Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(locomotion): add velocity tracking reward for G1 robot`

## Environment Variables

```bash
UNITREE_MODEL_DIR=/path/to/unitree_model    # USD robot models
UNITREE_ROS_DIR=/path/to/unitree_ros        # URDF robot descriptions
ISAACLAB_PATH=/path/to/IsaacLab             # Isaac Lab installation
```

## Common Issues

1. **Import errors**: Ensure Isaac Sim/Lab environment is activated (`conda activate env_isaaclab`)
2. **CUDA errors**: Training requires CUDA-compatible GPU
3. **RSL-RL version**: Version 2.3.1+ required for distributed training
4. **Missing robots**: Set `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
5. **pre-commit fails**: Run `pip install pre-commit && pre-commit install` to set up hooks
