# AGENTS.md — Unitree RL Lab

IsaacLab / Isaac Sim framework with H1_2 flat locomotion and G1-29dof locomotion/MSLPO workflows. Other Unitree robot asset configurations are available in `assets/robots/unitree.py`.

## Commands

```bash
# Setup
conda activate env_isaacsim                                     # required for all Isaac Sim work
./unitree_rl_lab.sh -i                                          # editable install; ISAACLAB_PATH must be set

# Train / Play
./unitree_rl_lab.sh -l
./unitree_rl_lab.sh -t --task Unitree-H1-2-Velocity --num_envs 512
./unitree_rl_lab.sh -p --task Unitree-H1-2-Velocity --num_envs 4
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity         # train (headless flag auto-added)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity         # play/visualize

# Offline H1_2 numerical tests (no Isaac Sim / GPU required)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_h1_2_mdp.py

# Short H1_2 integration run (requires Isaac Sim)
./unitree_rl_lab.sh -t --task Unitree-H1-2-Velocity --num_envs 512 --max_iterations 15 --run_name validation

# Lint (run before committing)
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
├── assets/robots/                          # unitree.py, unitree_actuators.py, h1_2.py + h1_2/ assets
├── controllers/simbicon/                   # SIMBICON gait controller + pose library pipeline
│   ├── pose_library_builder.py             #   stage1: SIMBICON params → 160 poses
│   └── pose_library_fusion.py              #   stage1 + human mocap → expanded library
├── tasks/
│   ├── locomotion/                         # velocity-tracking RL envs
│   │   ├── agents/                         # PPO configs
│   │   ├── mdp/                            # rewards, obs, commands, curriculums
│   │   └── robots/{h1_2,g1}/            # per-robot env configs
│   └── mimic/                              # motion capture tracking
└── utils/

scripts/
├── rsl_rl/                                 # train.py, play.py (require Isaac Sim)
├── mslpo/                                  # fuse_pose_library.py, build_stage1_pose_library.py
├── mocap/                                  # CMU → NPZ → G1 mapping (pure Python, offline)
└── mimic/

data/human_gait/                            # raw + processed mocap (not in git)
outputs/pose_library/                       # pose_library.npy, expanded, reports
tests/test_h1_2_mdp.py                      # offline H1_2 numerical tests
doc/h1_2_training.md                        # source pin, parameters, backend adaptations
```

**Key import boundary:** `scripts/mocap/` uses `sys.path.insert(0, parent_dir)` to import sibling modules (e.g. `from human_to_g1_mapping_config import ...`). `scripts/mslpo/` uses `sys.path.insert` pointing at `source/unitree_rl_lab/` to import `unitree_rl_lab.controllers.simbicon.*`. Neither depends on Isaac Sim.

## Isaac Sim Runtime Constraints

- Simulator-dependent imports (`omni`, IsaacLab assets/envs/tasks, and simulator extensions) must follow `AppLauncher(args_cli)` and `simulation_app = app_launcher.app`. Importing `isaaclab.app.AppLauncher` and registration-only modules before launch is required by the entry scripts.
- Use `env.unwrapped` for scene/device. Call `env.reset()` on the **wrapper** (not `unwrapped.reset()`) to avoid `ResetNeeded` errors.
- The Gym environment `env.step()` returns five values: `(obs, rewards, terminated, truncated, extras)`. Observations and extras are dictionaries; rewards and termination flags are tensors. `RslRlVecEnvWrapper.step()` returns four values.
- `ManagerBasedRLEnv` automatically resets ended environments during `step()`, before returning the next observations.
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

## H1_2 Official mjlab Migration

The existing `Unitree-H1-2-Velocity` task uses the official **27-joint flat** scheme. Update this task in place unless the user requests a separate variant.

- Upstream: `unitreerobotics/unitree_rl_mjlab`, pinned commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`; adapter semantics also follow its `mjlab==1.2.0` dependency.
- Configuration: `tasks/locomotion/robots/h1_2/velocity_env_cfg.py`; PPO: `rsl_rl_ppo_cfg.py` in the same directory. Numerical terms live in `mdp.py`; manager/contact adapters in `terms.py`; environment/action/randomization adapters in `env.py`.
- Defaults: training `num_envs=4096`, play `num_envs=32`. CLI `--num_envs` overrides either; 512 training environments were validated on the local 8 GB GPU. Do not silently change defaults when only editing documentation.
- Actions: 27, in `H1_2_JOINT_NAMES` order. Actor/critic observations: 92/107. Use explicit ordered joint resolution; articulation internal order differs from the action order.
- PPO: feedforward MLP `[512, 256, 128]`, ELU, actor/critic observation normalization enabled, noise std 1.0, seed 42, 24 rollout steps, 10001 updates, save interval 100. No LSTM or history stacking.
- Timing: simulation dt 0.005, decimation 4, gait period 0.6 s. Training episodes last 20 s. Play episodes last 1e9 s, disable corruption/pushes/curriculum, and retain startup randomization.
- Preserve all 16 rewards and signed total rewards, including termination weight -200. Do not restore the old nonnegative reward clipping or 12-joint model.
- Per-actuator action scale is `0.25 * effort_limit / stiffness`. Default target offsets come from `robot.data.default_joint_pos`. Encoder bias is subtracted from action targets; joint observations are unbiased by default.
- Bundled model: `assets/robots/h1_2/h1_2.xml` and exactly 28 referenced meshes. Preserve both license notices and `SOURCE.md`. H1_2 does not use the external model-directory constants in `unitree.py`.
- The MJCF spawn callback enables `isaacsim.asset.importer.mjcf`, removes its extra non-rigid `worldBody` articulation root, and applies collision exclusions. Imported body paths are under `{ENV_REGEX_NS}/Robot/pelvis/<body>`.
- Foot sites are `(0.04, 0, -0.04)` in ankle-roll link frames. Include angular velocity cross offset when calculating site velocity. The torso-mounted IMU has its own offset and frame.
- `root_angmom` represents the torso-and-arms subtree about its COM, excluding pelvis/legs. PhysX default inertia tensors are expressed in link actor frames.
- Use `FilteredContactSensor` for ground-only contact timers, first contacts, and normal-plus-tangential force histories. Built-in contact timers use unfiltered net forces. A filtered sensor must target one body; create separate sensors for each foot/self-contact body.
- Keep the fixed-pose kinematic floor, top z=0. The installed GPU backend rejected filtered contacts against the default static plane and static cuboid; the kinematic collider was validated with nonzero ground contacts.
- Scene self-contact fields are added before the single `configclass` conversion. Decorating before adding dynamic fields omits them from the instantiated scene.
- Keep tensor terms importable offline so `tests/test_h1_2_mdp.py` can run without launching Isaac Sim. Simulator-dependent classes belong in runtime modules.

Old models, migration scratch files, and short validation checkpoints/logs were removed. Do not reference the deleted `scripts/tests/h1_2_smoke.py`, `/tmp` source snapshots, or assume that a pretrained checkpoint exists. Retain permanent tests and source assets; avoid recreating historical variants or committing runtime artifacts. Historical short-run validation proves execution, not gait convergence.

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

For robots configured in `unitree.py`, `UNITREE_MODEL_DIR` and `UNITREE_ROS_DIR` are **hardcoded constants** in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:20-21`, not environment variables. They must be edited in that file to point at the local clone of the robot models.

`ISAACLAB_PATH` is a real env var set by the conda activate script (`_ut_setup_conda_env` in `unitree_rl_lab.sh`). It must be defined before running `./unitree_rl_lab.sh -i`.

## Gitignored Artifacts

`*.pt`, `*.npz`, `env.yaml`, and runtime log directories are gitignored. Do not commit trained weights, processed mocap data, exported runtime configs, or conversion caches. H1_2 model meshes have a narrow `.gitignore` exception and are tracked with Git LFS via `.gitattributes`; they are required source assets. Check actual ignore rules before assuming a shell script is ignored.

## Commit Style

Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.  
Example: `feat(locomotion): add velocity tracking reward for G1 robot`
