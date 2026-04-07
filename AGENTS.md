# AGENTS.md - Unitree RL Lab Development Guide

Guidelines for AI agents working on the Unitree RL Lab codebase - a reinforcement learning framework for Unitree robots (Go2, H1, G1-29dof) built on IsaacLab and Isaac Sim.

## Essential Commands

```bash
# Installation
./unitree_rl_lab.sh -i                    # Install in editable mode

# Environments
./unitree_rl_lab.sh -l                    # List environments
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity  # Train (headless)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity  # Play/visualize

# Manual training with options
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless --num_envs 64
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity

# SIMBICON controller debug / MSLPO parameter search
python scripts/mslpo/simbicon_debug.py --num_envs 4 --headless --max_steps 2000
python scripts/mslpo/qlearn_search.py --episodes 1000 --headless
```

## Build, Lint & Test Commands

```bash
# All pre-commit hooks (run before committing)
pre-commit run --all-files

# Run specific linter on a single file
pre-commit run black --files path/to/file.py
pre-commit run flake8 --files path/to/file.py
pre-commit run isort --files path/to/file.py

# Individual linters (single file)
black --line-length 120 --preview path/to/file.py
flake8 path/to/file.py
isort --profile black --filter-files path/to/file.py
pyright path/to/file.py

# Type check entire package
pyright source/unitree_rl_lab/
```

**Note**: This project has no traditional unit tests. Quick verification:
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 64 --max_iterations 10
```

Pre-commit excludes `deploy/` and `.vscode/`.

## Isaac Sim Runtime Constraints

- All IsaacLab imports (`omni`, `isaaclab`, `isaaclab_tasks`) **require** the Isaac Sim runtime. They must be placed **after** `AppLauncher` in scripts.
- Scripts must call `AppLauncher(args_cli)` and `simulation_app = app_launcher.app` before any IsaacLab imports.
- Use `env.unwrapped` to access the `ManagerBasedRLEnv` (the gym wrapper doesn't expose scene/device).
- Use `env.reset()` on the gym wrapper (not `env.unwrapped.reset()`) to avoid `ResetNeeded` errors.
- `env.step()` returns 5 values: `(obs, rewards, terminated, truncated, extras)` — all tensors.
- IsaacLab auto-resets terminated/truncated environments on the next `step()` call.
- `torch.inference_mode()` tensors cannot be modified inplace outside the context — always `.clone()` before mutation.

## Critical API Notes

### `find_bodies()` returns a tuple, not a list
```python
# WRONG - this returns a tuple
body_ids = robot.find_bodies("pattern")  # -> tuple[list[int], list[str]]

# CORRECT
body_ids, body_names = robot.find_bodies("pattern")
foot_id = body_ids[0] if body_ids else -1
```

### Contact sensor body ordering differs from articulation body ordering
```python
# WRONG - articulation body IDs don't match sensor body IDs
net_forces[:, articulation_body_id, :]

# CORRECT - resolve by name from the sensor's own body list
sensor_body_names = contact_sensor.body_names
for i, name in enumerate(sensor_body_names):
    if "left_ankle_roll" in name:
        left_foot_sensor_id = i
```

### `net_forces_w` shape is `(num_envs, num_bodies, 3)` per sensor
When using `prim_path="{ENV_REGEX_NS}/Robot/.*"` (one sensor, all bodies), index with `[env_idx, body_idx, :]`.

### Python module names starting with digits
`29dof` is a valid package name but cannot be imported with `from ... import` syntax. Use `importlib.import_module()`:
```python
import importlib
mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.some_config")
```

### Use `robot.data.default_joint_pos` for action computation
Never hardcode default joint positions. Always read from the live articulation:
```python
default_pos = robot.data.default_joint_pos  # (num_envs, num_joints)
action = (target - default_pos) / scale
```

## Code Style

### Imports (ordered top-to-bottom, separated by blank lines)
```python
# 1. FUTURE
from __future__ import annotations

# 2. STDLIB
import math
from typing import TYPE_CHECKING

# 3. THIRDPARTY (torch, numpy, gymnasium treated as stdlib per pyproject.toml)
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

# TYPE_CHECKING imports at end
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
```

**Note**: Some IsaacLab imports may require try/except for compatibility:
```python
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
```

### Formatting
- **Line length**: 120
- **Formatter**: Black `--preview`
- **Indentation**: 4 spaces (no tabs)
- **Python**: 3.10+ (pyupgrade uses `--py37-plus` but project targets 3.10)
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
| Constants | UPPER_SNAKE_CASE | `UNITREE_MODEL_DIR`, `COBBLESTONE_ROAD_CFG` |
| Functions/variables | snake_case | `reset_joints()`, `base_reward` |
| Private methods | `_snake_case()` | `_process_obs()` |
| Files | snake_case | `velocity_env_cfg.py` |
| Module-level singletons | UPPER_SNAKE_CASE | `UNITREE_G1_29DOF_CFG` |

### Docstrings (Google style)
```python
def track_lin_vel_xy_adaptive(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Adaptive linear velocity tracking reward based on terrain type.

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
if contact_sensor.cfg.track_air_time is False:
    raise RuntimeError("Activate ContactSensor's track_air_time!")
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

### Flake8 Ignored Rules
- `E402`: Module level import not at top of file
- `E501`: Line too long (handled by black)
- `W503`: Line break before binary operator
- `E203`: Whitespace before ':' (conflicts with black)
- `D401`: First line should be in imperative mood
- `R504/R505`: Unnecessary variable/elif after return
- `SIM102/SIM117`: Nested if-statement / merge with statements

## File Structure
```
source/unitree_rl_lab/unitree_rl_lab/
├── assets/robots/           # Robot configurations (unitree.py, unitree_actuators.py)
├── controllers/             # Traditional controllers (SIMBICON gait controller)
│   └── simbicon/            # FSM, PD, balance feedback, G1 joint mapping, param search
├── tasks/
│   ├── locomotion/          # Velocity-tracking environments
│   │   ├── agents/          # PPO agent configs (rsl_ppo_cfg_*.py)
│   │   ├── mdp/             # Rewards, observations, commands, curriculums
│   │   └── robots/{go2,h1,g1}/  # Per-robot environment configs
│   └── mimic/               # Motion capture tracking
└── utils/                   # export_deploy_cfg, parser_cfg

scripts/
├── list_envs.py
├── rsl_rl/                  # Training scripts (train.py, play.py, cli_args.py)
└── mslpo/                   # MSLPO parameter search (qlearn_search.py, simbicon_debug.py)
```

## G1-29dof Joint Layout

```
Indices 0-5:   Left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Indices 6-11:  Right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Indices 12-14: Waist     (yaw, roll, pitch)
Indices 15-21: Left arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
Indices 22-28: Right arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
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
4. **Missing robots**: Set `UNITREE_MODEL_DIR` or `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
5. **pre-commit fails**: Run `pip install pre-commit && pre-commit install`
6. **InferenceMode errors**: Tensors created inside `torch.inference_mode()` cannot be modified inplace outside it. Always `.clone()` before mutation in reset methods.
7. **Gym wrapper vs unwrapped**: Always call `env.reset()` (not `unwrapped.reset()`) and `env.step()` on the wrapper. Access scene/device via `env.unwrapped`.
