# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Number of episodes to evaluate in headless mode. Defaults to num_envs.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    headless: bool = args_cli.headless  # type: ignore[reportUnknownVariableType]
    timestep = 0
    # evaluation metrics (always allocated; only populated/exported in headless mode)
    num_episodes = args_cli.num_episodes if args_cli.num_episodes is not None else env.num_envs
    robot = env.unwrapped.scene["robot"]
    ep_forward_vel_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
    ep_lateral_drift_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
    ep_step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    completed = 0
    all_forward_vels: list[float] = []
    all_lateral_drifts: list[float] = []
    all_successes: list[bool] = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rew, dones, extras = env.step(actions)
        if headless:
            ep_forward_vel_sum += robot.data.root_lin_vel_b[:, 0].abs()
            ep_lateral_drift_sum += robot.data.root_lin_vel_b[:, 1].abs()
            ep_step_count += 1

            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                for rid in reset_ids:
                    if completed >= num_episodes:
                        break
                    steps = ep_step_count[rid].item()
                    if steps > 0:
                        all_forward_vels.append(ep_forward_vel_sum[rid].item() / steps)
                        all_lateral_drifts.append(ep_lateral_drift_sum[rid].item() / steps)
                        all_successes.append(not bool(env.unwrapped.reset_terminated[rid].item()))
                        completed += 1
                ep_forward_vel_sum[reset_ids] = 0.0
                ep_lateral_drift_sum[reset_ids] = 0.0
                ep_step_count[reset_ids] = 0
            if completed >= num_episodes:
                break
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # export evaluation metrics
    if headless and len(all_forward_vels) > 0:
        fwd_arr = np.array(all_forward_vels)
        lat_arr = np.array(all_lateral_drifts)
        succ_arr = np.array(all_successes, dtype=float)
        metrics = {
            "num_episodes": int(len(all_forward_vels)),
            "mean_forward_vel": float(fwd_arr.mean()),
            "std_forward_vel": float(fwd_arr.std()),
            "mean_lateral_drift_vel": float(lat_arr.mean()),
            "std_lateral_drift_vel": float(lat_arr.std()),
            "success_rate": float(succ_arr.mean()),
        }
        print(f"\n{'='*60}")
        print("  Evaluation Results")
        print(f"{'='*60}")
        print(f"  Episodes:               {metrics['num_episodes']}")
        print(f"  Mean Forward Vel:        {metrics['mean_forward_vel']:.4f} +/- {metrics['std_forward_vel']:.4f} m/s")
        print(
            f"  Mean Lateral Drift Vel:  {metrics['mean_lateral_drift_vel']:.4f} +/- {metrics['std_lateral_drift_vel']:.4f} m/s"
        )
        print(f"  Success Rate:            {metrics['success_rate']:.2%}")
        print(f"{'='*60}")
        results_path = os.path.join(os.path.dirname(resume_path), "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Results saved to: {results_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
