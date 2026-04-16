# AGENTS.md — Unitree RL Lab

RL framework for Unitree robots (Go2, H1, G1-29dof) on IsaacLab / Isaac Sim.

## Commands

```bash
# Setup
./unitree_rl_lab.sh -i                                          # editable install
conda activate env_isaaclab                                     # must be active for all Isaac Sim work

# Train / Play
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity         # train (headless flag auto-added)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity         # play/visualize

# Lint (run before committing — no traditional unit tests)
pre-commit run --all-files
black --line-length 120 --preview <file>
isort --profile black --filter-files <file>
flake8 <file>                                                   # config in .flake8
pyright <file>                                                  # config in pyproject.toml [tool.pyright]
pyright source/unitree_rl_lab/                                  # type-check whole package
```

```bash
# Offline pipelines (no Isaac Sim / GPU needed)
python scripts/mocap/batch_convert_cmu.py --input_dir data/human_gait/subject_07 --output_dir data/human_gait/processed/cmu_subject_07 --subject_id subject_07
python scripts/mocap/build_human_g1_candidates.py --skip_analysis --skip_plots
python scripts/mslpo/fuse_pose_library.py --skip_plots
python scripts/mslpo/build_stage1_pose_library.py

# SIMBICON controller / MSLPO parameter search (requires Isaac Sim)
python scripts/mslpo/simbicon_debug.py --num_envs 4 --headless --max_steps 2000
python scripts/mslpo/qlearn_search.py --episodes 1000 --headless
```

Pre-commit excludes `deploy/` and `.vscode/`. `pyright` has `reportMissingImports = "none"` — Isaac Sim import errors are expected and safe to ignore.

## Architecture

```
source/unitree_rl_lab/unitree_rl_lab/       # installable package
├── assets/robots/                          # unitree.py, unitree_actuators.py
├── controllers/simbicon/                   # SIMBICON gait controller + pose library pipeline
│   ├── pose_library_builder.py             #   stage1: SIMBICON params → 160 poses
│   └── pose_library_fusion.py              #   stage1 + human mocap → expanded library
├── tasks/
│   ├── locomotion/                         # velocity-tracking RL envs
│   │   ├── agents/                         # PPO configs
│   │   ├── mdp/                            # rewards, obs, commands, curriculums
│   │   └── robots/{go2,h1,g1}/            # per-robot env configs
│   └── mimic/                              # motion capture tracking
└── utils/

scripts/
├── rsl_rl/                                 # train.py, play.py (require Isaac Sim)
├── mslpo/                                  # fuse_pose_library.py, build_stage1_pose_library.py
├── mocap/                                  # CMU → NPZ → G1 mapping (pure Python, offline)
└── mimic/

data/human_gait/                            # raw + processed mocap (not in git)
outputs/pose_library/                       # pose_library.npy, expanded, reports
```

**Key import boundary:** `scripts/mocap/` uses `sys.path.insert(0, parent_dir)` to import sibling modules (e.g. `from human_to_g1_mapping_config import ...`). `scripts/mslpo/` uses `sys.path.insert` pointing at `source/unitree_rl_lab/` to import `unitree_rl_lab.controllers.simbicon.*`. Neither depends on Isaac Sim.

## Isaac Sim Runtime Constraints

- All IsaacLab imports (`omni`, `isaaclab`, `isaaclab_tasks`) **require** Isaac Sim runtime. They must be placed **after** `AppLauncher(args_cli)` and `simulation_app = app_launcher.app`.
- Use `env.unwrapped` for scene/device. Call `env.reset()` on the **wrapper** (not `unwrapped.reset()`) to avoid `ResetNeeded` errors.
- `env.step()` returns 5 values: `(obs, rewards, terminated, truncated, extras)` — all tensors.
- IsaacLab auto-resets terminated/truncated environments on the next `step()`.
- `torch.inference_mode()` tensors cannot be modified inplace outside the context — `.clone()` before mutation in reset methods.

## Critical API Pitfalls

- **`find_bodies()` returns `tuple[list[int], list[str]]`**, not a list. Unpack both elements.
- **Contact sensor body IDs ≠ articulation body IDs.** Resolve by name from `contact_sensor.body_names`.
- **`net_forces_w` shape** when using `prim_path="{ENV_REGEX_NS}/Robot/.*"`: `(num_envs, num_bodies, 3)` per sensor. Index as `[env_idx, body_idx, :]`.
- **`29dof` in import paths**: `from ... import` fails on digit-starting module names. Use `importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.some_config")`.
- **Never hardcode default joint positions.** Always read from `robot.data.default_joint_pos`.

## Code Style

- **Line length 120**, Black `--preview`, isort `--profile black --filter-files`.
- **Python 3.10+**: use `int | float` not `Union`. pyupgrade runs with `--py37-plus` but project targets 3.10.
- **Import order** (custom isort sections in `pyproject.toml`): future → stdlib → **torch/numpy (in `extra_standard_library`)** → third-party (isaacsim, pxr, warp, etc.) → **`ISAACLABPARTY` (isaaclab, isaaclab_tasks, isaaclab_rl, isaaclab_mimic, isaaclab_assets)** → first-party (`unitree_rl_lab`) → local (`config`). Agents must use this exact section ordering or isort will reorganize imports incorrectly.
- **IsaacLab import compat**: some utils need `try/except` (e.g. `quat_apply_inverse` may be `quat_rotate_inverse` in `rewards.py`).
- **Config classes**: use `@configclass` from IsaacLab, fields grouped as `scene`, `observations`, `actions`, `commands`, `rewards`, `terminations`, `events`, `curriculum`.
- **No comments unless asked.**
- **Docstrings**: Google style.
- **`codespell`** runs in pre-commit — watch for typos in docstrings and comments.

### Flake8 ignored rules

`E402 E501 W503 E203 D401 R504 R505 SIM102 SIM117` — defined in `.flake8`.

## G1-29dof Joint Layout (29 DOF)

```
Idx 0-5:   Left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Idx 6-11:  Right leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
Idx 12-14: Waist      (yaw, roll, pitch)
Idx 15-21: Left arm   (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
Idx 22-28: Right arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
```

**Knee limits**: `[0.0, 2.09]` rad. Knee mapping uses robust affine remap (`KNEE_MAPPING_CONFIG` in `scripts/mocap/human_to_g1_mapping_config.py`) to avoid saturation at 0.0 boundary.

## Pose Library Pipeline (offline)

```
CMU ASF/AMC → batch_convert_cmu.py → data/human_gait/processed/*.npz
    → build_human_g1_candidates.py → outputs/pose_library/human_g1_candidate_poses.npy (240, 29)
    → build_stage1_pose_library.py → outputs/pose_library/pose_library.npy (160, 29)
    → fuse_pose_library.py         → outputs/pose_library/pose_library_expanded.npy (216, 29)
```

The fusion pipeline (`pose_library_fusion.py`) filters by joint saturation, morphology, deduplicates (stage1 vs human, human vs human), scores (quality + novelty − limit_penalty), then applies a three-stage phase budget.

## Robot Model Paths

`UNITREE_MODEL_DIR` and `UNITREE_ROS_DIR` are **hardcoded constants** in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:20-21`, not environment variables. They must be edited in that file to point at the local clone of the robot models.

`ISAACLAB_PATH` is a real env var set by the conda activate script (`_ut_setup_conda_env` in `unitree_rl_lab.sh`). It must be defined before running `./unitree_rl_lab.sh -i`.

## Gitignored Artifacts

`*.pt`, `*.npz`, `env.yaml`, `scripts/*.sh` are gitignored. Do not try to commit trained model weights, processed mocap data, or env configs. `scripts/new_train.sh` and `scripts/resume_train.sh` exist locally but are ignored by git.

## Commit Style

Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.  
Example: `feat(locomotion): add velocity tracking reward for G1 robot`
