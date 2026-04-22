"""MSLPO Stage 2: evaluate a trained discrete-action PPO policy.

Loads a checkpoint, runs evaluation episodes, and prints summary metrics:
average reward, forward velocity, lateral offset, fall rate, survival time.

Usage::

    python scripts/mslpo/stage2_ppo_play.py \\
        --checkpoint outputs/stage2_ppo/.../stage2_ppo_final.pt

    python scripts/mslpo/stage2_ppo_play.py \\
        --checkpoint outputs/stage2_ppo/.../stage2_ppo_final.pt \\
        --num_envs 4 --num_episodes 20
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MSLPO Stage 2: evaluate trained PPO policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Stage2-PPO")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--real_time", action="store_true", default=False)
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
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.algorithms.categorical_ppo import CategoricalActorCritic
from unitree_rl_lab.controllers.simbicon.discrete_pose_library import DiscretePoseLibrary


def get_flat_obs(env_output) -> torch.Tensor:
    if isinstance(env_output, tuple):
        obs_dict = env_output[0]
    else:
        obs_dict = env_output
    if isinstance(obs_dict, dict):
        return obs_dict["policy"]
    return obs_dict


def apply_pose_action(action_indices, pose_library, default_joint_pos):
    indices = action_indices.squeeze(-1) if action_indices.dim() > 1 else action_indices
    target_poses = pose_library.get_pose(indices)
    return target_poses - default_joint_pos


def evaluate_stage2_policy(
    policy: CategoricalActorCritic,
    env,
    pose_library: DiscretePoseLibrary,
    default_joint_pos: torch.Tensor,
    num_episodes: int,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Evaluate policy over multiple episodes and return summary metrics.

    Args:
        policy: Trained CategoricalActorCritic.
        env: The gymnasium environment (wrapped).
        pose_library: Discrete pose library.
        default_joint_pos: Default joint positions (num_envs, 29).
        num_episodes: Total episodes to evaluate.
        device: Torch device.

    Returns:
        Dict with mean_reward, mean_forward_vel, mean_lateral_offset,
        fall_rate, mean_survival_time.
    """
    policy.eval()
    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt

    episode_rewards = []
    episode_forward_vels = []
    episode_lateral_offsets = []
    episode_survival_times = []
    episode_falls = []

    completed = 0
    cur_reward = torch.zeros(num_envs, device=device)
    cur_forward_vel_sum = torch.zeros(num_envs, device=device)
    cur_lateral_sum = torch.zeros(num_envs, device=device)
    cur_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    obs_dict, _ = env.reset()
    obs_flat = get_flat_obs((obs_dict, _))

    initial_y = unwrapped.scene["robot"].data.root_pos_w[:, 1].clone()

    while completed < num_episodes:
        with torch.no_grad():
            action_indices = policy.act_inference(obs_flat)

        env_actions = apply_pose_action(action_indices, pose_library, default_joint_pos)
        obs_dict, rewards, terminated, truncated, extras = env.step(env_actions)
        dones = terminated | truncated

        obs_flat = get_flat_obs((obs_dict, extras))

        root_pos = unwrapped.scene["robot"].data.root_pos_w
        root_lin_vel = unwrapped.scene["robot"].data.root_lin_vel_b

        cur_reward += rewards
        cur_forward_vel_sum += root_lin_vel[:, 0].abs()
        cur_lateral_sum += (root_pos[:, 1] - initial_y).abs()
        cur_steps += 1

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            for rid in reset_ids:
                if completed >= num_episodes:
                    break
                steps = cur_steps[rid].item()
                if steps == 0:
                    continue
                episode_rewards.append(cur_reward[rid].item())
                episode_forward_vels.append(cur_forward_vel_sum[rid].item() / steps)
                episode_lateral_offsets.append(cur_lateral_sum[rid].item() / steps)
                episode_survival_times.append(steps * dt)
                episode_falls.append(terminated[rid].item())
                completed += 1

            cur_reward[reset_ids] = 0.0
            cur_forward_vel_sum[reset_ids] = 0.0
            cur_lateral_sum[reset_ids] = 0.0
            cur_steps[reset_ids] = 0
            initial_y[reset_ids] = unwrapped.scene["robot"].data.root_pos_w[reset_ids, 1].clone()

    rewards_arr = np.array(episode_rewards[:num_episodes])
    vels_arr = np.array(episode_forward_vels[:num_episodes])
    lat_arr = np.array(episode_lateral_offsets[:num_episodes])
    time_arr = np.array(episode_survival_times[:num_episodes])
    fall_arr = np.array(episode_falls[:num_episodes])

    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "mean_forward_vel": float(vels_arr.mean()),
        "mean_lateral_offset": float(lat_arr.mean()),
        "mean_survival_time": float(time_arr.mean()),
        "fall_rate": float(fall_arr.mean()),
        "num_episodes": len(rewards_arr),
    }


def main() -> None:
    device = args_cli.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    pose_lib = DiscretePoseLibrary(
        library_path=args_cli.pose_library_path,
        meta_path=args_cli.pose_library_meta_path,
        device=device,
    )

    num_actions = config.get("num_discrete_actions", pose_lib.num_actions)
    actor_dims = config.get("actor_hidden_dims", [256, 256, 128])
    critic_dims = config.get("critic_hidden_dims", [256, 256, 128])
    activation = config.get("activation", "elu")

    # Use play config (fewer envs, wider velocity range)
    import importlib

    stage2_cfg_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.stage2_ppo_env_cfg")
    env_cfg = stage2_cfg_mod.Stage2PlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Determine obs_dim: prefer checkpoint config, else dynamic from env
    obs_dim = config.get("obs_dim")
    if obs_dim is None:
        obs_dict, _ = env.reset()
        obs_flat_tmp = get_flat_obs((obs_dict, _))
        obs_dim = obs_flat_tmp.shape[-1]
    print(f"[STAGE2-PLAY] obs_dim={obs_dim}")

    policy = CategoricalActorCritic(
        num_obs=obs_dim,
        num_actions=num_actions,
        actor_hidden_dims=actor_dims,
        critic_hidden_dims=critic_dims,
        activation=activation,
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    print(f"[STAGE2-PLAY] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[STAGE2-PLAY] Iteration: {ckpt.get('iteration', '?')}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos.clone().to(device)

    # ---- Interactive play loop (if GUI) ----
    if not args_cli.headless:
        print("[STAGE2-PLAY] Interactive mode — close window to exit.")
        obs_dict, _ = env.reset()
        obs_flat = get_flat_obs((obs_dict, _))
        dt = env.unwrapped.step_dt
        step = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.no_grad():
                action_indices = policy.act_inference(obs_flat)
            env_actions = apply_pose_action(action_indices, pose_lib, default_joint_pos)
            obs_dict, rewards, terminated, truncated, extras = env.step(env_actions)
            obs_flat = get_flat_obs((obs_dict, extras))
            step += 1
            if args_cli.video and step >= args_cli.video_length:
                break
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    else:
        # ---- Headless evaluation ----
        print(f"[STAGE2-PLAY] Running {args_cli.num_episodes} evaluation episodes...")
        metrics = evaluate_stage2_policy(
            policy,
            env,
            pose_lib,
            default_joint_pos,
            num_episodes=args_cli.num_episodes,
            device=device,
        )

        print(f"\n{'='*60}")
        print("  Stage 2 PPO Evaluation Results")
        print(f"{'='*60}")
        print(f"  Episodes:          {metrics['num_episodes']}")
        print(f"  Mean Reward:        {metrics['mean_reward']:+.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Forward Vel:   {metrics['mean_forward_vel']:.4f} m/s")
        print(f"  Mean Lateral Offset:{metrics['mean_lateral_offset']:.4f} m")
        print(f"  Mean Survival Time: {metrics['mean_survival_time']:.2f} s")
        print(f"  Fall Rate:          {metrics['fall_rate']:.2%}")
        print(f"{'='*60}")

        results_path = os.path.join(os.path.dirname(args_cli.checkpoint), "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[STAGE2-PLAY] Results saved to: {results_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
