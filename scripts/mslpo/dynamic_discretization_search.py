"""Phase B+C: Dynamic discretization Q-learning search with uniform comparison.

Loads (or runs) a sensitivity scan, builds non-uniform discrete value lists
for each parameter, then runs Q-learning search using the dynamic parameter
space.  Finally runs a uniform-space comparison with the same episode budget
and outputs a side-by-side report.

Usage:
    # With pre-existing scan:
    python scripts/mslpo/dynamic_discretization_search.py --headless \
        --episodes 200 --sensitivity_scan outputs/qlearn_search/parameter_sensitivity_scan.json

    # Run scan inline first:
    python scripts/mslpo/dynamic_discretization_search.py --headless --episodes 200
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

parser = argparse.ArgumentParser(description="Phase B+C: dynamic discretization Q-learning search.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Simbicon-Debug")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--episodes", type=int, default=200, help="Episodes for each search (dynamic + uniform).")
parser.add_argument("--sensitivity_scan", type=str, default=None, help="Path to pre-existing scan JSON.")
parser.add_argument(
    "--base_param_mode",
    type=str,
    default="median",
    choices=["mean", "median", "best"],
    help="Base param derivation mode (used only when running scan inline).",
)
parser.add_argument("--num_rollouts", type=int, default=3, help="Rollouts per scan point (inline scan only).")
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--epsilon_start", type=float, default=1.0)
parser.add_argument("--epsilon_end", type=float, default=0.05)
parser.add_argument("--epsilon_decay", type=float, default=0.995)
parser.add_argument("--k_v", type=float, default=1.0)
parser.add_argument("--k_y", type=float, default=3.0)
parser.add_argument("--k_alive", type=float, default=10.0)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--print_interval", type=int, default=10)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="outputs/dynamic_discretization")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--sensitivity_score_metric",
    type=str,
    default="composite",
    choices=["composite", "total_reward", "time_to_fall", "early_forward_velocity"],
    help="Metric for sensitivity scan and discretization.",
)
parser.add_argument("--sensitivity_alpha", type=float, default=0.6, help="Weight for time_to_fall in composite score.")
parser.add_argument("--target_reward_threshold", type=float, default=15.0)
parser.add_argument("--target_forward_velocity_threshold", type=float, default=0.33)
parser.add_argument("--compare", action="store_true", help="Run uniform comparison and generate report.")
parser.add_argument("--scan_only", action="store_true", help="Run sensitivity scan only, then exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.controllers.simbicon.simbicon_cfg import SimbiconCfg
from unitree_rl_lab.controllers.simbicon.simbicon_controller import SimbiconController
from unitree_rl_lab.controllers.simbicon.simbicon_param_search import (
    DynamicDiscretizer,
    DynamicParameterSpace,
    GaitParameterSpace,
    ParameterSensitivityAnalyzer,
    SimbiconQLearningSearcher,
    SparseQTable,
    VelocityStateDiscretizer,
    compare_uniform_vs_dynamic,
)

_simbicon_cfg_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.simbicon_debug_env_cfg")
SimbiconPlayEnvCfg = _simbicon_cfg_mod.SimbiconPlayEnvCfg


def _create_env_controller():
    env_cfg = SimbiconPlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped

    cfg = SimbiconCfg(continuous_walking=True, max_steps=100)
    controller = SimbiconController(cfg)
    controller.initialize(unwrapped.scene["robot"], unwrapped.num_envs, unwrapped.device)

    joint_indices = controller.joint_indices
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    joint_indices.resolve_from_contact_sensor(contact_sensor.body_names)

    return env, controller, joint_indices


def _run_search(param_space, num_episodes, label):
    q_table = SparseQTable(default_q=0.0)
    state_discretizer = VelocityStateDiscretizer()
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

    start_time = time.time()
    best_reward = float("-inf")
    for ep in range(num_episodes):
        if not simulation_app.is_running():
            print(f"[WARN] Simulation app stopped during {label} search.")
            break

        result = searcher.run_episode(
            env,
            controller,
            joint_indices,
            args_cli.max_steps,
            k_v=args_cli.k_v,
            k_y=args_cli.k_y,
            k_alive=args_cli.k_alive,
        )

        if result.total_reward > best_reward:
            best_reward = result.total_reward

        if (ep + 1) % args_cli.print_interval == 0:
            top = searcher.get_top_results()
            top_score = top[0].score if top else 0.0
            elapsed = time.time() - start_time
            print(
                f"[{label} EP {ep+1:4d}/{num_episodes}] "
                f"params=({result.hl},{result.ls},{result.lswb},{result.lforward}) "
                f"r={result.total_reward:+.2f} "
                f"eps={searcher._epsilon:.3f} "
                f"Q={searcher.q_table.size()} "
                f"best={best_reward:+.2f} "
                f"top1={top_score:+.2f} "
                f"time={elapsed:.0f}s"
            )

    search_time = time.time() - start_time
    return searcher, search_time


env = None
controller = None
joint_indices = None


def main() -> None:
    global env, controller, joint_indices

    os.makedirs(args_cli.output_dir, exist_ok=True)

    env, controller, joint_indices = _create_env_controller()
    dt = env.unwrapped.step_dt  # type: ignore[union-attr]
    print(f"[INFO] Dynamic discretization search (dt={dt:.4f}s)")
    print(f"[INFO] Episodes per search: {args_cli.episodes}")

    scan_time = 0.0

    # === Phase A: Load or run sensitivity scan ===
    scan_results = None
    if args_cli.sensitivity_scan and os.path.exists(args_cli.sensitivity_scan):
        print(f"[INFO] Loading sensitivity scan from {args_cli.sensitivity_scan}")
        with open(args_cli.sensitivity_scan) as f:
            scan_results = json.load(f)
    else:
        print("[INFO] No scan file provided, running sensitivity scan inline...")
        analyzer = ParameterSensitivityAnalyzer(
            env=env,
            controller=controller,
            joint_indices=joint_indices,
            max_steps=args_cli.max_steps,
            k_v=args_cli.k_v,
            k_y=args_cli.k_y,
            k_alive=args_cli.k_alive,
            num_rollouts=args_cli.num_rollouts,
            base_param_mode=args_cli.base_param_mode,
            sensitivity_score_metric=args_cli.sensitivity_score_metric,
            sensitivity_alpha=args_cli.sensitivity_alpha,
        )
        scan_start = time.time()
        scan_results = analyzer.scan_all_parameters()
        scan_time = time.time() - scan_start
        scan_path = f"{args_cli.output_dir}/../qlearn_search/parameter_sensitivity_scan.json"
        analyzer.save_results(scan_results, scan_path)
        print(f"[INFO] Inline scan completed in {scan_time:.1f}s")

    if args_cli.scan_only:
        _metric_key_map = {
            "composite": "composite_score",
            "time_to_fall": "mean_time_to_fall",
            "total_reward": "mean_total_reward",
            "early_forward_velocity": "mean_early_window_forward_velocity",
        }
        metric_key = _metric_key_map.get(args_cli.sensitivity_score_metric, "composite_score")
        total_points = sum(len(v) for v in scan_results.values()) if isinstance(scan_results, dict) else 0
        print(f"\n{'='*70}")
        print(f"  Sensitivity Scan Summary ({total_points} points, {scan_time:.1f}s)")
        print(f"  Ranking metric: {metric_key}")
        print(f"{'='*70}")
        for pname, pts in scan_results.items():
            best = max(pts, key=lambda d: d[metric_key])
            worst = min(pts, key=lambda d: d[metric_key])
            print(
                f"  {pname:>10s}: best_val={best['param_value']:4d} "
                f"{metric_key}={best[metric_key]:+.4f} | "
                f"worst_val={worst['param_value']:4d} "
                f"{metric_key}={worst[metric_key]:+.4f} | "
                f"reward_range={best['mean_total_reward'] - worst['mean_total_reward']:.4f} "
                f"ttf_best={best['mean_time_to_fall']:.4f}s "
                f"fall_rate={best['fall_rate']:.2f}"
            )
        print(f"{'='*70}")
        env.close()
        print("[INFO] Scan-only mode complete.")
        return

    # === Phase B: Build dynamic discretization ===
    _metric_map = {
        "composite": "composite_score",
        "time_to_fall": "mean_time_to_fall",
        "total_reward": "mean_total_reward",
        "early_forward_velocity": "mean_early_window_forward_velocity",
    }
    disc_metric = _metric_map.get(args_cli.sensitivity_score_metric, "composite_score")
    alpha = args_cli.sensitivity_alpha

    discretizer_before = DynamicDiscretizer(metric=disc_metric, sensitivity_alpha=alpha, high_value_pct=50.0)
    values_before = discretizer_before.build_all(scan_results)
    space_before = DynamicParameterSpace(
        hl_values=values_before["HL"],
        ls_values=values_before["Ls"],
        lswb_values=values_before["Lswb"],
        lforward_values=values_before["Lforward"],
    )
    before_size = space_before.total_actions

    discretizer = DynamicDiscretizer(metric=disc_metric, sensitivity_alpha=alpha)
    dynamic_values = discretizer.build_all(scan_results)
    config_path = f"{args_cli.output_dir}/dynamic_discretization_config.json"
    discretizer.save_config(dynamic_values, scan_results, config_path)

    dynamic_space = DynamicParameterSpace(
        hl_values=dynamic_values["HL"],
        ls_values=dynamic_values["Ls"],
        lswb_values=dynamic_values["Lswb"],
        lforward_values=dynamic_values["Lforward"],
    )

    uniform_space = GaitParameterSpace()
    print(f"[INFO] Uniform action space:   {uniform_space.total_actions:,}")
    print(f"[INFO] Dynamic action space:   {dynamic_space.total_actions:,}")
    reduction = (1.0 - dynamic_space.total_actions / uniform_space.total_actions) * 100
    print(f"[INFO] Action space reduction: {reduction:.1f}%")

    # === Phase C: Dynamic Q-learning search ===
    print(f"\n{'='*60}")
    print(f"  Running DYNAMIC discretization search ({args_cli.episodes} episodes)")
    print(f"{'='*60}")
    dynamic_searcher, dynamic_search_time = _run_search(dynamic_space, args_cli.episodes, label="DYN")

    dynamic_output = args_cli.output_dir

    top_path = os.path.join(dynamic_output, "top5_pose_params_dynamic.json")
    top_results = dynamic_searcher.get_top_results()
    top_data = []
    for r in top_results:
        top_data.append(
            {
                "rank": len(top_data) + 1,
                "HL": r.hl,
                "Ls": r.ls,
                "Lswb": r.lswb,
                "Lforward": r.lforward,
                "avg_forward_velocity": round(r.avg_forward_velocity, 6),
                "avg_lateral_offset": round(r.avg_lateral_offset, 6),
                "alive_time": round(r.alive_time, 4),
                "success": r.success,
                "total_reward": round(r.total_reward, 4),
                "score": round(r.score, 4),
            }
        )
    with open(top_path, "w") as f:
        json.dump(top_data, f, indent=2)

    with open(os.path.join(dynamic_output, "dynamic_episode_log.json"), "w") as f:
        json.dump(dynamic_searcher._episode_log, f, indent=2)

    with open(os.path.join(dynamic_output, "q_table_checkpoint_dynamic.json"), "w") as f:
        json.dump(dynamic_searcher.q_table.to_dict_with_meta(dynamic_searcher._epsilon, dynamic_searcher._episode), f)

    print(f"\n  Dynamic search completed in {dynamic_search_time:.1f}s")

    # === Optional: Comparison ===
    if args_cli.compare:
        print(f"\n{'='*60}")
        print(f"  Running UNIFORM discretization search ({args_cli.episodes} episodes)")
        print(f"{'='*60}")
        uniform_searcher, uniform_search_time = _run_search(uniform_space, args_cli.episodes, label="UNI")

        with open(os.path.join(dynamic_output, "uniform_episode_log.json"), "w") as f:
            json.dump(uniform_searcher._episode_log, f, indent=2)

        print(f"\n  Uniform search completed in {uniform_search_time:.1f}s")

        report = compare_uniform_vs_dynamic(
            uniform_episode_log=uniform_searcher._episode_log,
            dynamic_episode_log=dynamic_searcher._episode_log,
            uniform_search_time=uniform_search_time,
            dynamic_search_time=dynamic_search_time,
            scan_time=scan_time,
            uniform_space_size=uniform_space.total_actions,
            dynamic_space_size=dynamic_space.total_actions,
            dynamic_space_size_before_tightening=before_size,
            target_reward_threshold=args_cli.target_reward_threshold,
            target_forward_velocity_threshold=args_cli.target_forward_velocity_threshold,
        )

        report_path = os.path.join(dynamic_output, "dynamic_vs_uniform_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print("  DYNAMIC vs UNIFORM Comparison Report")
        print(f"{'='*70}")
        print(f"  {'Metric':<40s} {'Uniform':>12s} {'Dynamic':>12s}")
        print(f"  {'-'*40} {'-'*12} {'-'*12}")
        print(
            f"  {'Action space size':<40s} {report['uniform_action_space_size']:>12,} {report['dynamic_action_space_size_after_tightening']:>12,}"
        )
        print(
            f"  {'Action space (before tightening)':<40s} {'':>12s} {report['dynamic_action_space_size_before_tightening']:>12,}"
        )
        print(f"  {'Action space reduction':<40s} {'':>12s} {report['action_space_reduction_percent']:>11.1f}%")
        u_conv = report["uniform_convergence_episodes"]
        d_conv = report["dynamic_convergence_episodes"]
        print(f"  {'Convergence episode':<40s} {str(u_conv):>12s} {str(d_conv):>12s}")
        u_ep_r = report["uniform_episodes_to_target_reward"]
        d_ep_r = report["dynamic_episodes_to_target_reward"]
        print(
            f"  {'Episodes to target reward':<40s} {str(u_ep_r):>12s} {str(d_ep_r):>12s} "
            f"(>={report['target_reward_threshold']})"
        )
        u_ep_v = report["uniform_episodes_to_target_velocity"]
        d_ep_v = report["dynamic_episodes_to_target_velocity"]
        print(
            f"  {'Episodes to target velocity':<40s} {str(u_ep_v):>12s} {str(d_ep_v):>12s} "
            f"(>={report['target_forward_velocity_threshold']})"
        )
        print(
            f"  {'Search time only (s)':<40s} {report['uniform_search_time_only']:>12.1f} {report['dynamic_search_time_only']:>12.1f}"
        )
        print(f"  {'Total time incl. scan (s)':<40s} {'N/A':>12s} {report['total_time_including_scan']:>12.1f}")
        print(
            f"  {'Best total reward':<40s} {report['uniform_best_total_reward']:>+12.4f} {report['dynamic_best_total_reward']:>+12.4f}"
        )
        print(
            f"  {'Best forward velocity':<40s} "
            f"{report['uniform_best_forward_velocity']:>+12.6f} "
            f"{report['dynamic_best_forward_velocity']:>+12.6f}"
        )
        print(
            f"  {'Best lateral offset':<40s} "
            f"{report['uniform_best_lateral_offset']:>12.6f} "
            f"{report['dynamic_best_lateral_offset']:>12.6f}"
        )
        print(
            f"  {'Best time_to_fall (s)':<40s} "
            f"{report['uniform_best_time_to_fall']:>12.4f} "
            f"{report['dynamic_best_time_to_fall']:>12.4f}"
        )
        print(
            f"  {'Best early fwd velocity':<40s} "
            f"{report['uniform_best_early_forward_velocity']:>+12.6f} "
            f"{report['dynamic_best_early_forward_velocity']:>+12.6f}"
        )
        print(
            f"  {'Success rate':<40s} {report['uniform_success_rate']:>12.2%} {report['dynamic_success_rate']:>12.2%}"
        )
        print(f"  {'Efficiency improvement':<40s} {'':>12s} {report['efficiency_improvement_percent']:>11.1f}%")
        print(f"  {'Assessment':<40s} {report['dynamic_search_assessment']}")
        print(f"{'='*70}")

    print(f"\n  Dynamic Top-{args_cli.top_k}:")
    for entry in top_data:
        print(
            f"    #{entry['rank']}: HL={entry['HL']} Ls={entry['Ls']} "
            f"Lswb={entry['Lswb']} Lfwd={entry['Lforward']} "
            f"reward={entry['total_reward']:+.4f} vel={entry['avg_forward_velocity']:+.4f}"
        )

    print(f"\n  All outputs saved to: {args_cli.output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
