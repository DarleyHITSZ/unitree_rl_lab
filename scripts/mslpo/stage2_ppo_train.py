"""MSLPO Stage 2: discrete-action PPO training over the expanded pose library.

Trains a Categorical PPO policy that selects from 216 discrete poses at each
simulation step.  The selected pose is converted to a joint-position target
and sent to the G1-29dof robot through the standard ``JointPositionAction``
pipeline.

Usage::

    python scripts/mslpo/stage2_ppo_train.py \\
        --task Unitree-G1-29dof-Stage2-PPO \\
        --num_envs 4096 --headless \\
        --max_iterations 50000

    python scripts/mslpo/stage2_ppo_train.py \\
        --task Unitree-G1-29dof-Stage2-PPO \\
        --checkpoint outputs/stage2_ppo/stage2_ppo.pt \\
        --max_iterations 50000 --headless
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MSLPO Stage 2: discrete-action PPO training.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Stage2-PPO")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=50000)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--print_interval", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="outputs/stage2_ppo")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_ratio", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--update_epochs", type=int, default=5)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[256, 256, 128])
parser.add_argument("--activation", type=str, default="elu")
parser.add_argument(
    "--pose_library_path",
    type=str,
    default="outputs/pose_library/pose_library_expanded.npy",
)
parser.add_argument(
    "--pose_library_meta_path",
    type=str,
    default="outputs/pose_library/pose_library_expanded_meta.json",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Everything below runs inside Isaac Sim ----

import gymnasium as gym
import importlib
import torch

import isaaclab_tasks  # noqa: F401

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.categorical_ppo import (
    CategoricalActorCritic,
    CategoricalPPO,
    CategoricalRolloutStorage,
)
from unitree_rl_lab.controllers.simbicon.discrete_pose_library import DiscretePoseLibrary

stage2_cfg_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.stage2_ppo_env_cfg")

TASK_NAME = args_cli.task


def get_flat_obs(env_output) -> torch.Tensor:
    """Extract the flat policy observation tensor from env output.

    Handles both ``env.get_observations()`` (returns dict) and ``env.step()``
    (returns 5-tuple where first element is dict).
    """
    if isinstance(env_output, tuple):
        obs_dict = env_output[0]
    else:
        obs_dict = env_output
    if isinstance(obs_dict, dict):
        return obs_dict["policy"]
    return obs_dict


def apply_pose_action(
    action_indices: torch.Tensor,
    pose_library: DiscretePoseLibrary,
    default_joint_pos: torch.Tensor,
) -> torch.Tensor:
    """Map discrete action indices to env-compatible joint position actions.

    With ``JointPositionActionCfg(scale=1.0, use_default_offset=True)`` the
    env processes actions as ``processed = action * 1.0 + default``.  To send
    ``target_pos`` as the final joint target, ``action = target - default``.
    """
    indices = action_indices.squeeze(-1) if action_indices.dim() > 1 else action_indices
    target_poses = pose_library.get_pose(indices)
    return target_poses - default_joint_pos


def main() -> None:
    device = args_cli.device or "cuda:0"

    # ---- Environment ----
    env_cfg = stage2_cfg_mod.Stage2EnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    # ---- Initial reset to determine obs_dim dynamically ----
    obs_dict, _ = env.reset()
    obs_flat = get_flat_obs((obs_dict, _))
    obs_dim = obs_flat.shape[-1]
    print(f"[STAGE2] Detected obs_dim={obs_dim}")

    # ---- Pose library ----
    pose_lib = DiscretePoseLibrary(
        library_path=args_cli.pose_library_path,
        meta_path=args_cli.pose_library_meta_path,
        device=device,
    )
    num_discrete_actions = pose_lib.num_actions
    print(f"[STAGE2] Pose library: {num_discrete_actions} actions, {pose_lib.num_joints} joints")

    # ---- Default joint positions ----
    robot = unwrapped.scene["robot"]
    default_joint_pos = robot.data.default_joint_pos.clone().to(device)

    # ---- Policy & PPO ----
    torch.manual_seed(args_cli.seed)

    num_envs = unwrapped.num_envs

    policy = CategoricalActorCritic(
        num_obs=obs_dim,
        num_actions=num_discrete_actions,
        actor_hidden_dims=args_cli.actor_hidden_dims,
        critic_hidden_dims=args_cli.critic_hidden_dims,
        activation=args_cli.activation,
    ).to(device)

    storage = CategoricalRolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=args_cli.num_steps_per_env,
        obs_dim=obs_dim,
        device=device,
    )

    ppo = CategoricalPPO(
        policy=policy,
        storage=storage,
        lr=args_cli.lr,
        gamma=args_cli.gamma,
        gae_lambda=args_cli.gae_lambda,
        clip_ratio=args_cli.clip_ratio,
        entropy_coef=args_cli.entropy_coef,
        value_coef=args_cli.value_coef,
        max_grad_norm=args_cli.max_grad_norm,
        update_epochs=args_cli.update_epochs,
        num_mini_batches=args_cli.num_mini_batches,
        device=device,
    )

    # ---- Resume from checkpoint ----
    start_iteration = 0
    if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0)
        print(f"[STAGE2] Resumed from {args_cli.checkpoint} at iteration {start_iteration}")

    # ---- Logging setup ----
    log_dir = os.path.join(args_cli.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir)
        use_tb = True
    except ImportError:
        writer = None
        use_tb = False

    # ---- Training bookkeeping ----
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    ep_infos: list[dict] = []
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=device)
    best_mean_reward = float("-inf")

    # ---- Save config ----
    import json

    config_dict = {
        "task": TASK_NAME,
        "num_envs": num_envs,
        "obs_dim": obs_dim,
        "num_discrete_actions": num_discrete_actions,
        "num_steps_per_env": args_cli.num_steps_per_env,
        "lr": args_cli.lr,
        "gamma": args_cli.gamma,
        "gae_lambda": args_cli.gae_lambda,
        "clip_ratio": args_cli.clip_ratio,
        "entropy_coef": args_cli.entropy_coef,
        "value_coef": args_cli.value_coef,
        "max_grad_norm": args_cli.max_grad_norm,
        "update_epochs": args_cli.update_epochs,
        "num_mini_batches": args_cli.num_mini_batches,
        "actor_hidden_dims": args_cli.actor_hidden_dims,
        "critic_hidden_dims": args_cli.critic_hidden_dims,
        "activation": args_cli.activation,
        "pose_library_path": args_cli.pose_library_path,
        "seed": args_cli.seed,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"[STAGE2] Training started: {num_envs} envs, {obs_dim} obs dim")
    print(
        f"[STAGE2] PPO: lr={args_cli.lr}, clip={args_cli.clip_ratio}, "
        f"entropy={args_cli.entropy_coef}, epochs={args_cli.update_epochs}"
    )
    print(f"[STAGE2] Max iterations: {args_cli.max_iterations}, " f"steps/env: {args_cli.num_steps_per_env}")
    print(f"[STAGE2] Logging to: {log_dir}")

    start_time = time.time()

    for iteration in range(start_iteration, args_cli.max_iterations):
        if not simulation_app.is_running():
            print("[WARN] Simulation app stopped.")
            break

        ep_infos.clear()

        for _step in range(args_cli.num_steps_per_env):
            action_indices = ppo.act(obs_flat)

            env_actions = apply_pose_action(action_indices, pose_lib, default_joint_pos)

            obs_dict, rewards, terminated, truncated, extras = env.step(env_actions)
            dones = terminated | truncated

            obs_flat = get_flat_obs((obs_dict, extras))

            ppo.process_env_step(obs_flat, rewards, dones, extras)

            cur_reward_sum += rewards
            cur_episode_length += 1

            if "episode" in extras:
                ep_infos.append(extras["episode"])
            elif "log" in extras:
                ep_infos.append(extras["log"])

            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                for rid in reset_ids:
                    rewbuffer.append(cur_reward_sum[rid].item())
                    lenbuffer.append(cur_episode_length[rid].item())
                cur_reward_sum[reset_ids] = 0.0
                cur_episode_length[reset_ids] = 0.0

        ppo.compute_returns(obs_flat)

        with torch.no_grad():
            unique_actions = torch.unique(storage.actions[: storage.num_transitions_per_env]).numel()

        loss_dict = ppo.update()

        # ---- Logging ----
        mean_reward = sum(rewbuffer) / max(len(rewbuffer), 1)
        mean_ep_len = sum(lenbuffer) / max(len(lenbuffer), 1)

        if use_tb:
            writer.add_scalar("Loss/value_loss", loss_dict["value_loss"], iteration)
            writer.add_scalar("Loss/surrogate_loss", loss_dict["surrogate_loss"], iteration)
            writer.add_scalar("Loss/entropy", loss_dict["entropy"], iteration)
            writer.add_scalar("Perf/mean_reward", mean_reward, iteration)
            writer.add_scalar("Perf/mean_episode_length", mean_ep_len, iteration)
            writer.add_scalar("Policy/unique_actions_used", unique_actions, iteration)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

        if (iteration + 1) % args_cli.print_interval == 0 or iteration == start_iteration:
            elapsed = time.time() - start_time
            fps = (iteration - start_iteration + 1) * args_cli.num_steps_per_env * num_envs / max(elapsed, 1e-6)
            print(
                f"[ITER {iteration+1:6d}/{args_cli.max_iterations}] "
                f"reward={mean_reward:+.2f} "
                f"ep_len={mean_ep_len:.0f} "
                f"v_loss={loss_dict['value_loss']:.4f} "
                f"surr_loss={loss_dict['surrogate_loss']:.4f} "
                f"entropy={loss_dict['entropy']:.4f} "
                f"unique={unique_actions}/{num_discrete_actions} "
                f"best={best_mean_reward:+.2f} "
                f"fps={fps:.0f} "
                f"time={elapsed:.0f}s"
            )

        # ---- Save checkpoint ----
        if (iteration + 1) % args_cli.save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"stage2_ppo_{iteration+1}.pt")
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "iteration": iteration + 1,
                    "config": config_dict,
                },
                ckpt_path,
            )
            print(f"[SAVE] Checkpoint: {ckpt_path}")

    # ---- Final save ----
    final_path = os.path.join(log_dir, "stage2_ppo_final.pt")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "iteration": args_cli.max_iterations,
            "config": config_dict,
        },
        final_path,
    )
    print(f"\n[STAGE2] Training complete. Final model: {final_path}")
    print(f"[STAGE2] Best mean reward: {best_mean_reward:+.2f}")
    print(f"[STAGE2] Total time: {time.time() - start_time:.1f}s")

    if use_tb:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
