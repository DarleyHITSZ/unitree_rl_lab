"""MSLPO Phase 1: Q-learning gait parameter search script.

Runs Q-learning to search for optimal SIMBICON gait parameters (HL, Ls, Lswb,
Lforward) on the G1-29dof robot. Outputs top-5 parameter sets, Q-table
checkpoint, and per-episode log.

Usage:
    python scripts/qlearn_search.py --task Unitree-G1-29dof-Simbicon-Debug \
        --num_envs 1 --max_steps 2000 --episodes 500 --headless

    python scripts/qlearn_search.py --task Unitree-G1-29dof-Simbicon-Debug \
        --num_envs 1 --max_steps 2000 --episodes 100 --output_dir outputs/my_run
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

parser = argparse.ArgumentParser(description="MSLPO Phase 1: Q-learning gait parameter search.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Simbicon-Debug", help="Environment task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max simulation steps per episode.")
parser.add_argument("--episodes", type=int, default=1000, help="Number of Q-learning episodes.")
parser.add_argument("--output_dir", type=str, default="outputs/qlearn_search", help="Output directory.")
parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning learning rate.")
parser.add_argument("--gamma", type=float, default=0.95, help="Q-learning discount factor.")
parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate.")
parser.add_argument("--epsilon_end", type=float, default=0.05, help="Minimum exploration rate.")
parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay factor per episode.")
parser.add_argument("--k_v", type=float, default=1.0, help="Velocity reward coefficient.")
parser.add_argument("--k_y", type=float, default=3.0, help="Lateral offset penalty coefficient.")
parser.add_argument("--k_alive", type=float, default=10.0, help="Survival reward coefficient.")
parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N episodes.")
parser.add_argument("--print_interval", type=int, default=10, help="Print progress every N episodes.")
parser.add_argument("--top_k", type=int, default=5, help="Number of top results to track.")
parser.add_argument("--checkpoint", type=str, default=None, help="Load Q-table from checkpoint.")
parser.add_argument(
    "--no_set_params", action="store_true", help="Skip set_gait_params (diagnostic: test default controller)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib

import isaaclab_tasks  # noqa: F401

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.controllers.simbicon.simbicon_cfg import SimbiconCfg
from unitree_rl_lab.controllers.simbicon.simbicon_controller import SimbiconController
from unitree_rl_lab.controllers.simbicon.simbicon_param_search import (
    GaitParameterSpace,
    SimbiconQLearningSearcher,
    SparseQTable,
    VelocityStateDiscretizer,
)

_simbicon_cfg_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.simbicon_debug_env_cfg")
SimbiconPlayEnvCfg = _simbicon_cfg_mod.SimbiconPlayEnvCfg

TASK_NAME = args_cli.task


def main() -> None:
    """Run the Q-learning gait parameter search."""
    env_cfg = SimbiconPlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    env = gym.make(TASK_NAME, cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped

    cfg = SimbiconCfg(continuous_walking=True, max_steps=100)
    controller = SimbiconController(cfg)
    controller.initialize(unwrapped.scene["robot"], unwrapped.num_envs, unwrapped.device)

    joint_indices = controller.joint_indices

    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    joint_indices.resolve_from_contact_sensor(contact_sensor.body_names)

    dt = unwrapped.step_dt

    print("[INFO] Q-learning search started")
    print(f"[INFO] Task: {TASK_NAME}")
    print(f"[INFO] Environments: {args_cli.num_envs}, dt={dt:.4f}s")
    print(f"[INFO] Episodes: {args_cli.episodes}, max_steps: {args_cli.max_steps}")
    print(f"[INFO] Output dir: {args_cli.output_dir}")
    print(f"[INFO] Alpha={args_cli.alpha}, Gamma={args_cli.gamma}")
    print(f"[INFO] Epsilon: {args_cli.epsilon_start} -> {args_cli.epsilon_end} (decay={args_cli.epsilon_decay})")

    param_space = GaitParameterSpace()
    state_discretizer = VelocityStateDiscretizer()

    print(f"[INFO] Action space: {param_space.total_actions:,} combinations")
    print(
        f"[INFO] State space: {state_discretizer.num_bins} velocity bins "
        f"[{state_discretizer.vel_min}, {state_discretizer.vel_max}] m/s"
    )

    q_table = SparseQTable(default_q=0.0)
    resume_episode = 0
    if args_cli.checkpoint and args_cli.checkpoint != "":
        q_table, saved_epsilon, saved_episode = SparseQTable.load(args_cli.checkpoint)
        resume_episode = saved_episode
        print(f"[INFO] Loaded Q-table checkpoint: {q_table.size()} entries")
        print(f"[INFO] Resuming from episode {saved_episode}, epsilon={saved_epsilon:.4f}")

    searcher = SimbiconQLearningSearcher(
        param_space=param_space,
        state_discretizer=state_discretizer,
        q_table=q_table,
        alpha=args_cli.alpha,
        gamma=args_cli.gamma,
        epsilon_start=args_cli.epsilon_start,
        epsilon_end=args_cli.epsilon_end,
        epsilon_decay=args_cli.epsilon_decay,
        top_k=args_cli.top_k,
    )

    if resume_episode > 0:
        searcher._epsilon = saved_epsilon
        searcher._episode = resume_episode

    best_reward = float("-inf")
    start_time = time.time()
    total_episodes = args_cli.episodes

    for ep in range(resume_episode, total_episodes):
        if not simulation_app.is_running():
            print("[WARN] Simulation app stopped, ending search.")
            break

        ep_start = time.time()
        result = searcher.run_episode(
            env,
            controller,
            joint_indices,
            args_cli.max_steps,
            k_v=args_cli.k_v,
            k_y=args_cli.k_y,
            k_alive=args_cli.k_alive,
            skip_set_params=args_cli.no_set_params,
        )
        ep_elapsed = time.time() - ep_start

        if result.total_reward > best_reward:
            best_reward = result.total_reward

        if (ep + 1) % args_cli.print_interval == 0 or ep == resume_episode:
            top = searcher.get_top_results()
            top_score = top[0].score if top else 0.0
            elapsed = time.time() - start_time
            print(
                f"[EP {ep+1:4d}/{total_episodes}] "
                f"params=({result.hl},{result.ls},{result.lswb},{result.lforward}) "
                f"vel={result.avg_forward_velocity:+.3f} lat={result.avg_lateral_offset:.3f} "
                f"steps={result.step_count} "
                f"r={result.total_reward:+.2f} "
                f"eps={searcher._epsilon:.3f} "
                f"Q={searcher.q_table.size()} "
                f"best={best_reward:+.2f} "
                f"top1={top_score:+.2f} "
                f"time={elapsed:.0f}s "
                f"ep_time={ep_elapsed:.1f}s"
            )

        if (ep + 1) % args_cli.save_interval == 0:
            searcher.save_checkpoint(args_cli.output_dir)
            print(f"[SAVE] Checkpoint saved at episode {ep+1}")

    searcher.save_checkpoint(args_cli.output_dir)

    total_time = time.time() - start_time
    print(f"\n[INFO] Search completed: {searcher._episode} episodes in {total_time:.1f}s")
    print(f"[INFO] Best reward: {best_reward:+.2f}")
    print(f"[INFO] Q-table entries: {searcher.q_table.size()}")
    print(f"[INFO] Outputs saved to: {args_cli.output_dir}")

    top_results = searcher.get_top_results()
    print(f"\n{'='*60}")
    print(f"  Top-{args_cli.top_k} Gait Parameters")
    print(f"{'='*60}")
    print(
        f"  {'Rank':>4s}  {'HL':>4s}  {'Ls':>4s}  {'Lswb':>4s}  {'Lfwd':>4s}  "
        f"{'AvgVel':>8s}  {'AvgLat':>7s}  {'Alive':>6s}  {'Succ':>4s}  {'Reward':>8s}"
    )
    print(f"  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  " f"{'-'*8}  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*8}")
    for rank, r in enumerate(top_results, 1):
        print(
            f"  {rank:4d}  {r.hl:4d}  {r.ls:4d}  {r.lswb:4d}  {r.lforward:4d}  "
            f"{r.avg_forward_velocity:+8.3f}  {r.avg_lateral_offset:7.3f}  "
            f"{r.alive_time:6.2f}  {'Y' if r.success else 'N':>4s}  "
            f"{r.total_reward:+8.2f}"
        )
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
