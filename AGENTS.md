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
# All pre-commit hooks
pre-commit run --all-files

# Individual linters
pre-commit run black --all-files      # Format
pre-commit run flake8 --all-files     # Lint
pre-commit run isort --all-files      # Sort imports
pre-commit run pyupgrade --all-files   # Python syntax
pre-commit run codespell --all-files  # Spell check

# Manual
black --line-length 120 --preview <files>
flake8 <files>
isort --profile black --filter-files <files>
pyright <files>
pyright source/unitree_rl_lab/

# Training tests (manual verification)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

Pre-commit excludes `deploy/` and `.vscode/`.

## Code Style

### Imports (ordered: FUTURE → STDLIB → THIRDPARTY → ISAACLABPARTY → FIRSTPARTY → LOCALFOLDER)
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
- **Line length**: 120
- **Formatter**: Black `--preview`
- **Indentation**: 4 spaces (no tabs)
- **Python**: 3.10+ (pyupgrade `--py37-plus`)
- **Line breaks**: Before binary operators (W503 ignored)
- **Custom lists**: Wrap with `# fmt: off` / `# fmt: on`

### Type Annotations
- **Checker**: pyright (`typeCheckingMode = "basic"`, Python 3.10)
- **Syntax**: `int | float` (not `Union[int, float]`)
- **Configs**: `@configclass` from IsaacLab
- **Ignored**: `reportGeneralTypeIssues`, `reportMissingImports`, `reportMissingModuleSource`

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `RobotEnvCfg` |
| Config classes | PascalCase + `Cfg` | `RobotSceneCfg` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Functions/variables | snake_case | `reset_joints()` |
| Private methods | `_snake_case()` | `_process_obs()` |
| Files | snake_case | `velocity_env_cfg.py` |

### Docstrings (Google style, imperative mood)
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
- Specific exception types (no bare `except:`)
- Informative error messages
- Propagate exceptions for configuration validation

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
Organize into: `SceneCfg`, `EventCfg`, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `RewardsCfg`, `TerminationsCfg`, `CurriculumCfg`.

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

Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example: `feat(locomotion): add velocity tracking reward for G1 robot`

## Common Issues

1. **Import errors**: Ensure Isaac Sim/Lab is activated
2. **CUDA errors**: Training requires CUDA-compatible GPU
3. **RSL-RL**: Version 2.3.1+ required for distributed training
4. **Missing robots**: Set `UNITREE_MODEL_DIR` or `UNITREE_ROS_DIR`
