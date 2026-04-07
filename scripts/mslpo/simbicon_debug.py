"""Debug script for the SIMBICON bipedal gait controller.

Runs the SIMBICON controller on the G1-29dof robot in the Isaac Sim
environment and prints diagnostic information at regular intervals.

Usage:
    python scripts/simbicon_debug.py --num_envs 4 --headless
    python scripts/simbicon_debug.py --num_envs 1   # with GUI
"""

from __future__ import annotations

import argparse
import time
import torch
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

parser = argparse.ArgumentParser(description="Debug the SIMBICON gait controller.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max simulation steps.")
parser.add_argument("--print_interval", type=int, default=50, help="Print debug info every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib

import isaaclab_tasks  # noqa: F401

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.controllers.simbicon.g1_joint_map import G1JointIndices
from unitree_rl_lab.controllers.simbicon.simbicon_cfg import SimbiconCfg
from unitree_rl_lab.controllers.simbicon.simbicon_controller import SimbiconController, SimbiconStateData

_simbicon_cfg_mod = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.simbicon_debug_env_cfg"
)
SimbiconPlayEnvCfg = _simbicon_cfg_mod.SimbiconPlayEnvCfg

TASK_NAME = "Unitree-G1-29dof-Simbicon-Debug"


def extract_state_data(env: ManagerBasedRLEnv, joint_indices: G1JointIndices) -> SimbiconStateData:
    """Extract robot state data from the IsaacLab environment.

    Args:
        env: The ManagerBasedRLEnv instance.
        joint_indices: Resolved G1 joint and body indices.

    Returns:
        SimbiconStateData with current robot state.
    """
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]

    root_state = robot.data.root_state_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    body_pos = robot.data.body_pos_w

    contact_data = contact_sensor.data
    net_forces = contact_data.net_forces_w

    left_force = torch.zeros(env.num_envs, device=env.device)
    right_force = torch.zeros(env.num_envs, device=env.device)

    if net_forces is not None and net_forces.shape[1] > 0:
        if joint_indices.left_foot_sensor_id >= 0:
            left_force = net_forces[:, joint_indices.left_foot_sensor_id, :].norm(dim=1)
        if joint_indices.right_foot_sensor_id >= 0:
            right_force = net_forces[:, joint_indices.right_foot_sensor_id, :].norm(dim=1)

    return SimbiconStateData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_state[:, :3],
        root_quat=root_state[:, 3:7],
        root_lin_vel=root_state[:, 7:10],
        root_ang_vel=root_state[:, 10:13],
        body_pos=body_pos,
        left_foot_contact=left_force > 1.0,
        right_foot_contact=right_force > 1.0,
        left_foot_force=left_force,
        right_foot_force=right_force,
    )


def print_debug_info(
    step: int,
    controller: SimbiconController,
    state: SimbiconStateData,
    last_targets: torch.Tensor,
) -> None:
    """Print SIMBICON controller debug information.

    Args:
        step: Current simulation step.
        controller: The SIMBICON controller instance.
        state: Current robot state data.
        last_targets: Last computed 29-DOF joint targets.
    """
    info = controller.get_debug_info()

    print(f"\n{'='*60}")
    print(f"  SIMBICON Debug - Step {step}")
    print(f"{'='*60}")

    if info["fsm_state"] is not None:
        print(f"  FSM State:        {info['fsm_state_names'][0]}")
        print(f"  State Timer:      {info['state_timer'][0].item():.3f} s")
        print(f"  Phase:            {info['phase'][0].item():.3f}")
        print(f"  Step Count:       {info['step_count'][0].item()}")

        support_map = {0: "LEFT", 1: "RIGHT", 2: "BOTH"}
        swing_map = {0: "LEFT", 1: "RIGHT", 2: "NONE"}
        print(f"  Support Foot:     {support_map[info['support_foot'][0].item()]}")
        print(f"  Swing Foot:       {swing_map[info['swing_foot'][0].item()]}")

    print(f"  COM Position:     ({state.root_pos[0, 0]:.3f}, {state.root_pos[0, 1]:.3f}, {state.root_pos[0, 2]:.3f})")
    print(
        f"  COM Velocity:     ({state.root_lin_vel[0, 0]:.3f}, {state.root_lin_vel[0, 1]:.3f}, {state.root_lin_vel[0, 2]:.3f})"
    )

    print(f"  Left Contact:     {state.left_foot_contact[0].item()}  (force={state.left_foot_force[0].item():.1f} N)")
    print(f"  Right Contact:    {state.right_foot_contact[0].item()}  (force={state.right_foot_force[0].item():.1f} N)")

    if info.get("correction") is not None:
        print(f"  Balance Correction: {info['correction'][0].item():.4f} rad")

    joint_names = [
        "L_hip_pitch",
        "L_hip_roll",
        "L_hip_yaw",
        "L_knee",
        "L_ankle_pitch",
        "L_ankle_roll",
        "R_hip_pitch",
        "R_hip_roll",
        "R_hip_yaw",
        "R_knee",
        "R_ankle_pitch",
        "R_ankle_roll",
    ]
    print("  Joint Targets (env 0, controllable 12 DOF):")
    targets = last_targets
    for j, name in enumerate(joint_names):
        jid = controller.joint_indices.controllable_joint_ids[j]
        cur = state.joint_pos[0, jid].item()
        tgt = targets[0, jid].item()
        print(f"    {name:16s}: cur={cur:+.4f}  tgt={tgt:+.4f}")

    print(f"{'='*60}")


def main() -> None:
    """Run the SIMBICON controller debug loop."""
    env_cfg = SimbiconPlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    env = gym.make(TASK_NAME, cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped

    cfg = SimbiconCfg(continuous_walking=True, max_steps=100)
    controller = SimbiconController(cfg)
    controller.initialize(unwrapped.scene["robot"], unwrapped.num_envs, unwrapped.device)

    obs, _ = env.reset()
    controller.reset()
    joint_indices = controller.joint_indices

    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    joint_indices.resolve_from_contact_sensor(contact_sensor.body_names)
    print(f"[INFO] Contact sensor bodies: {contact_sensor.num_bodies}")
    print(f"[INFO] Left foot sensor ID: {joint_indices.left_foot_sensor_id}")
    print(f"[INFO] Right foot sensor ID: {joint_indices.right_foot_sensor_id}")

    dt = unwrapped.step_dt
    print(f"[INFO] SIMBICON Debug started: {args_cli.num_envs} envs, dt={dt:.4f}s")
    print(f"[INFO] Controller config: continuous_walking={cfg.continuous_walking}")
    print(f"[INFO] Running for {args_cli.max_steps} steps...")

    step = 0
    last_targets = torch.zeros(unwrapped.num_envs, 29, device=unwrapped.device)
    while simulation_app.is_running() and step < args_cli.max_steps:
        start_time = time.time()

        with torch.inference_mode():
            state_data = extract_state_data(unwrapped, joint_indices)
            last_targets = controller.step(state_data, dt)
            actions = controller.compute_actions_from_targets(last_targets)
            obs, rewards, terminated, truncated, extras = env.step(actions)

        step += 1

        if step % args_cli.print_interval == 0:
            with torch.inference_mode():
                print_debug_info(step, controller, state_data, last_targets)
            print(f"  Reward (env 0): {rewards[0].item():.4f}")  # type: ignore[index]

        reset_mask = terminated | truncated
        if reset_mask.any():
            reset_ids = torch.where(reset_mask)[0]
            print(f"\n[RESET] Env(s) {reset_ids.tolist()} reset at step {step}")
            with torch.inference_mode():
                controller.reset(reset_ids)

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0 and not args_cli.headless:
            time.sleep(sleep_time)

    print(f"\n[INFO] SIMBICON Debug finished after {step} steps.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
