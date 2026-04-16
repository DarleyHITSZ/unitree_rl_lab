"""Build, inspect, and visualize the Stage 1 pose library.

Generates a discrete pose library (160 poses x 29 DOFs) from the top-5
gait parameter sets discovered by Q-learning in Stage 1, validates
integrity, and produces visualization plots.

This script runs purely offline -- no Isaac Sim or GPU required.

Usage:
    python scripts/mslpo/build_stage1_pose_library.py

    python scripts/mslpo/build_stage1_pose_library.py \
        --top5 outputs/qlearn_search/top5_pose_params.json

    python scripts/mslpo/build_stage1_pose_library.py --skip_plots
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
from collections import Counter
from typing import Any

from unitree_rl_lab.controllers.simbicon.pose_library_builder import (
    CORE_STATES,
    Stage1PoseLibraryBuilder,
    load_top5_params,
)

# ---------------------------------------------------------------------------
# Step 1: Build
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    print("\n" + "=" * 60)
    print("  Step 1: Build Pose Library")
    print("=" * 60)

    print("\n[CONFIG]")
    print(f"  Top-5 path:  {args.top5}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Num phases:  {args.num_phases}")
    print(f"  Core states: {args.core_states}")

    params_list = load_top5_params(args.top5)
    print(f"\n[INFO] Loaded {len(params_list)} parameter groups from {args.top5}")
    print(f"  Total poses to generate: {len(params_list) * len(args.core_states) * args.num_phases}")

    for i, p in enumerate(params_list):
        print(
            f"  Group {i}: HL={p['HL']}, Ls={p['Ls']}, Lswb={p['Lswb']}, "
            f"Lforward={p['Lforward']}, reward={p.get('total_reward', 0):.2f}"
        )

    builder = Stage1PoseLibraryBuilder(num_phases=args.num_phases, core_states=args.core_states)
    poses, meta = builder.build(params_list, args.output_dir)

    print("\n[RESULT]")
    print(f"  Pose array shape: {poses.shape}")
    print(f"  Metadata entries: {len(meta)}")
    print(f"  Action space size: {poses.shape[0]}")

    print("\n[STATISTICS]")
    print(f"  Joint range: [{poses.min():.4f}, {poses.max():.4f}] rad")
    print(f"  Per-joint std: min={poses.std(axis=0).min():.4f}, max={poses.std(axis=0).max():.4f}")

    print("\n[DONE] Pose library built successfully.")
    return poses, meta


# ---------------------------------------------------------------------------
# Step 2: Inspect
# ---------------------------------------------------------------------------


def run_inspect(output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 2: Inspect Pose Library")
    print("=" * 60)

    npy_path = os.path.join(output_dir, "pose_library.npy")
    meta_path = os.path.join(output_dir, "pose_library_meta.json")

    poses = np.load(npy_path)
    print(f"\n[SHAPE] {poses.shape} (dtype={poses.dtype})")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(f"[META]  {len(meta)} entries")

    print(f"\n{'='*60}")
    print("  Integrity Checks")
    print(f"{'='*60}")

    checks_passed = 0
    checks_failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        else:
            checks_failed += 1
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)

    check("Shape is (160, 29)", poses.shape == (160, 29), f"got {poses.shape}")
    check("Dtype is float32", poses.dtype == np.float32, f"got {poses.dtype}")
    check("No NaN values", not np.any(np.isnan(poses)))
    check("No Inf values", not np.any(np.isinf(poses)))

    finite_poses = poses[np.isfinite(poses)]
    if len(finite_poses) > 0:
        check(
            "Joint range within +/-2.0 rad",
            np.abs(finite_poses).max() <= 2.0,
            f"max abs = {np.abs(finite_poses).max():.4f} rad",
        )

    check("Meta length is 160", len(meta) == 160, f"got {len(meta)}")

    group_counts = Counter(e.get("param_group_idx", -1) for e in meta)
    for g in range(5):
        check(f"Param group {g} has 32 poses", group_counts.get(g, 0) == 32, f"got {group_counts.get(g, 0)}")

    state_counts = Counter(e.get("fsm_state", "") for e in meta)
    check(
        "STEP_RIGHT_WITH_LEFT_FRONT has 80 poses",
        state_counts.get("STEP_RIGHT_WITH_LEFT_FRONT", 0) == 80,
        f"got {state_counts.get('STEP_RIGHT_WITH_LEFT_FRONT', 0)}",
    )
    check(
        "STEP_LEFT_WITH_RIGHT_FRONT has 80 poses",
        state_counts.get("STEP_LEFT_WITH_RIGHT_FRONT", 0) == 80,
        f"got {state_counts.get('STEP_LEFT_WITH_RIGHT_FRONT', 0)}",
    )

    action_indices = [e.get("action_idx", -1) for e in meta]
    check("Action indices are 0..159", sorted(action_indices) == list(range(160)))
    check("Action indices are unique", len(set(action_indices)) == 160)

    phase_values = [e.get("phase_value", -1) for e in meta]
    check("All phase values in [0, 1]", all(0.0 <= pv <= 1.0 + 1e-6 for pv in phase_values))

    print(f"\n  Result: {checks_passed} passed, {checks_failed} failed")

    print(f"\n{'='*60}")
    print("  Joint Statistics (degrees)")
    print(f"{'='*60}")

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
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "L_sh_pitch",
        "L_sh_roll",
        "L_sh_yaw",
        "L_elbow",
        "L_wr_roll",
        "L_wr_pitch",
        "L_wr_yaw",
        "R_sh_pitch",
        "R_sh_roll",
        "R_sh_yaw",
        "R_elbow",
        "R_wr_roll",
        "R_wr_pitch",
        "R_wr_yaw",
    ]
    print(f"  {'Joint':16s}  {'Min':>8s}  {'Max':>8s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"  {'-'*56}")
    for j in range(29):
        col = poses[:, j]
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        print(
            f"  {name:16s}  {np.degrees(col.min()):+8.2f}  "
            f"{np.degrees(col.max()):+8.2f}  {np.degrees(col.mean()):+8.2f}  "
            f"{np.degrees(col.std()):8.2f}"
        )

    print("\n[DONE] Inspection complete.")


# ---------------------------------------------------------------------------
# Step 3: Visualize
# ---------------------------------------------------------------------------


def _get_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _plot_group_phase_curves(poses: np.ndarray, meta: list[dict], param_group_idx: int, output_dir: str, plt) -> None:
    group_poses = []
    group_meta = []
    for i, entry in enumerate(meta):
        if entry["param_group_idx"] == param_group_idx:
            group_poses.append(poses[i])
            group_meta.append(entry)

    group_poses = np.array(group_poses)

    hl = group_meta[0]["HL"]
    ls = group_meta[0]["Ls"]
    lswb = group_meta[0]["Lswb"]
    lforward = group_meta[0]["Lforward"]
    rank = group_meta[0].get("rank", "?")
    reward = group_meta[0].get("total_reward", 0)

    leg_joints = [
        ("L_hip_pitch", 0),
        ("L_hip_roll", 1),
        ("L_knee", 3),
        ("L_ankle_pitch", 4),
        ("R_hip_pitch", 6),
        ("R_hip_roll", 7),
        ("R_knee", 9),
        ("R_ankle_pitch", 10),
    ]

    state_order = ["STEP_RIGHT_WITH_LEFT_FRONT", "STEP_LEFT_WITH_RIGHT_FRONT"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Group {param_group_idx} (rank={rank}): "
        f"HL={hl}, Ls={ls}, Lswb={lswb}, Lfwd={lforward}, "
        f"reward={reward:.2f}",
        fontsize=13,
    )

    for col, (joint_name, joint_idx) in enumerate(leg_joints):
        ax = axes[col % 2, col // 2]
        for s, state_name in enumerate(state_order):
            state_entries = [e for e in group_meta if e["fsm_state"] == state_name]
            if not state_entries:
                continue
            state_indices = [e["action_idx"] for e in state_entries]
            state_poses_col = poses[state_indices, joint_idx]
            phases = [e["phase_value"] for e in state_entries]
            label = "STEP_R/L" if s == 0 else "STEP_L/R"
            ax.plot(phases, np.degrees(state_poses_col), "o-", label=label, markersize=3)
        ax.set_title(joint_name)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"group_{param_group_idx}_phase_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_all_groups_heatmap(poses: np.ndarray, output_dir: str, plt) -> None:
    leg_indices = list(range(12))
    leg_names = [
        "L_hp",
        "L_hr",
        "L_hy",
        "L_kn",
        "L_ap",
        "L_ar",
        "R_hp",
        "R_hr",
        "R_hy",
        "R_kn",
        "R_ap",
        "R_ar",
    ]
    leg_poses = poses[:, leg_indices]

    fig, ax = plt.subplots(figsize=(14, 20))
    im = ax.imshow(np.degrees(leg_poses), aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Leg Joint")
    ax.set_ylabel("Pose Index (action_idx)")
    ax.set_xticks(range(12))
    ax.set_xticklabels(leg_names, fontsize=9)

    for g in range(5):
        y_start = g * 32
        y_end = y_start + 32
        ax.axhline(y=y_start, color="white", linewidth=1.5)
        if g > 0:
            ax.axhline(y=y_start - 0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.text(-0.8, (y_start + y_end) / 2, f"G{g}", fontsize=10, ha="right", va="center", fontweight="bold")

    ax.text(-0.8, -3, "Group", fontsize=10, ha="right", va="center", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Joint Angle (deg)")

    plt.tight_layout()
    path = os.path.join(output_dir, "all_groups_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_group_comparison(poses: np.ndarray, meta: list[dict], output_dir: str, plt) -> None:
    state_name = "STEP_RIGHT_WITH_LEFT_FRONT"
    joint_idx = 3
    joint_name = "left_knee"

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for g in range(5):
        state_entries = [e for e in meta if e["param_group_idx"] == g and e["fsm_state"] == state_name]
        if not state_entries:
            continue
        state_indices = [e["action_idx"] for e in state_entries]
        phases = [e["phase_value"] for e in state_entries]
        values = np.degrees(poses[state_indices, joint_idx])
        hl = state_entries[0]["HL"]
        ls = state_entries[0]["Ls"]
        rank = state_entries[0].get("rank", "?")
        ax.plot(phases, values, "o-", label=f"G{g} (rank={rank}, HL={hl}, Ls={ls})", color=colors[g], markersize=3)

    ax.set_title(f"{joint_name} across groups -- {state_name}")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Angle (deg)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "group_comparison_knee.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def run_visualize(output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 3: Visualize Pose Library")
    print("=" * 60)

    plt = _get_matplotlib()
    if plt is None:
        print("\n[SKIP] matplotlib not installed, skipping visualization.")
        print("  Install with: pip install matplotlib")
        return

    npy_path = os.path.join(output_dir, "pose_library.npy")
    meta_path = os.path.join(output_dir, "pose_library_meta.json")

    poses = np.load(npy_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n[INFO] Loaded poses: {poses.shape}, meta: {len(meta)} entries")
    print(f"[INFO] Plots directory: {plots_dir}")

    print("\n[INFO] Generating per-group phase curves...")
    num_groups = len(set(e.get("param_group_idx", -1) for e in meta))
    for g in range(num_groups):
        _plot_group_phase_curves(poses, meta, g, plots_dir, plt)

    print("\n[INFO] Generating all-groups heatmap...")
    _plot_all_groups_heatmap(poses, plots_dir, plt)

    print("\n[INFO] Generating group comparison...")
    _plot_group_comparison(poses, meta, plots_dir, plt)

    print(f"\n[DONE] All plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, inspect, and visualize the Stage 1 pose library.",
    )
    parser.add_argument(
        "--top5",
        type=str,
        default="outputs/dynamic_discretization/top5_pose_params_dynamic.json",
        help="Path to top5_pose_params.json from Stage 1.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pose_library",
        help="Directory for all outputs (npy, meta, plots).",
    )
    parser.add_argument(
        "--num_phases",
        type=int,
        default=16,
        help="Number of phase samples per FSM state (default: 16).",
    )
    parser.add_argument(
        "--core_states",
        type=str,
        nargs="+",
        default=CORE_STATES,
        help="FSM states to include (default: STEP_RIGHT_WITH_LEFT_FRONT STEP_LEFT_WITH_RIGHT_FRONT).",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip visualization plots (no matplotlib dependency).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MSLPO Stage 1 -- Pose Library Pipeline")
    print("=" * 60)

    poses, meta = run_build(args)

    run_inspect(args.output_dir)

    if not args.skip_plots:
        run_visualize(args.output_dir)

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  Output dir: {args.output_dir}")
    print(f"  Poses:      {args.output_dir}/pose_library.npy ({poses.shape})")
    print(f"  Metadata:   {args.output_dir}/pose_library_meta.json ({len(meta)} entries)")
    if not args.skip_plots:
        print(f"  Plots:      {args.output_dir}/plots/")
    print()


if __name__ == "__main__":
    main()
