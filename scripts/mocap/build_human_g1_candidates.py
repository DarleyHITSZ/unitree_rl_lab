"""Build, inspect, and visualize the human-gait-based G1 candidate poses.

Reads processed CMU Mocap .npz files, extracts lower-body gait cycles,
maps them to G1-29dof joint space (29 DOF), validates integrity, and
produces visualization plots.  Outputs are aligned with the stage1 pose
library format (16 phases per cycle).

This script runs purely offline -- no Isaac Sim or GPU required.

Usage::

    python scripts/mocap/build_human_g1_candidates.py

    python scripts/mocap/build_human_g1_candidates.py \\
        --input_dir data/human_gait/processed/cmu_subject_07 \\
        --output_dir outputs/pose_library

    python scripts/mocap/build_human_g1_candidates.py --analyze_only
    python scripts/mocap/build_human_g1_candidates.py --skip_analysis
    python scripts/mocap/build_human_g1_candidates.py --skip_plots
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_gait_to_g1 import (
    CMULowerBodySelector,
    GaitCycleSegmenter,
    HumanToG1PoseMapper,
    ProcessedCMUGaitLoader,
    analyze_cmu_axis_importance,
    build_human_g1_candidate_dataset,
    resample_cycle_to_16_phases,
)
from human_to_g1_mapping_config import (
    G1_JOINT_LIMITS,
    G1_JOINT_NAMES,
    NUM_PHASES,
    NUM_TOTAL_JOINTS,
)

JOINT_DISPLAY_NAMES: list[str] = [
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


# ---------------------------------------------------------------------------
# Step 1: Build
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> tuple[np.ndarray, list[dict]]:
    print("\n" + "=" * 60)
    print("  Step 1: Build Human G1 Candidate Poses")
    print("=" * 60)

    print("\n[CONFIG]")
    print(f"  Input dir:   {args.input_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Num phases:  {args.num_phases}")

    poses, meta = build_human_g1_candidate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_phases=args.num_phases,
        run_analysis=not args.skip_analysis,
    )

    print("\n[RESULT]")
    print(f"  Pose array shape: {poses.shape}")
    print(f"  Metadata entries: {len(meta)}")
    print("\n[DONE] Build step complete.")
    return poses, meta


# ---------------------------------------------------------------------------
# Step 2: Inspect
# ---------------------------------------------------------------------------


def run_inspect(output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 2: Inspect Human G1 Candidate Poses")
    print("=" * 60)

    npy_path = os.path.join(output_dir, "human_g1_candidate_poses.npy")
    meta_path = os.path.join(output_dir, "human_g1_candidate_poses_meta.json")

    if not os.path.exists(npy_path):
        print(f"\n[ERROR] File not found: {npy_path}")
        sys.exit(1)

    poses = np.load(npy_path)
    print(f"\n[SHAPE] {poses.shape} (dtype={poses.dtype})")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(f"[META]  {len(meta)} entries")

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

    print(f"\n{'=' * 60}")
    print("  Integrity Checks")
    print(f"{'=' * 60}")

    check("Shape is (N, 29)", len(poses.shape) == 2 and poses.shape[1] == 29, f"got {poses.shape}")
    check("Dtype is float32", poses.dtype == np.float32, f"got {poses.dtype}")
    check("No NaN values", not np.any(np.isnan(poses)))
    check("No Inf values", not np.any(np.isinf(poses)))
    check("Meta length matches poses", len(meta) == poses.shape[0], f"meta={len(meta)}, poses={poses.shape[0]}")

    finite_poses = poses[np.isfinite(poses)]
    if len(finite_poses) > 0:
        check(
            "Joint range within +/-2.0 rad",
            np.abs(finite_poses).max() <= 2.0,
            f"max abs = {np.abs(finite_poses).max():.4f} rad",
        )

    source_counts = Counter(e.get("source_npz", "") for e in meta)
    print(f"\n  Sources: {dict(source_counts)}")

    cycle_counts = Counter(e.get("cycle_id", -1) for e in meta)
    print(f"  Cycle IDs: {len(cycle_counts)} distinct cycles")

    method_counts = Counter(e.get("cycle_method", "") for e in meta)
    print(f"  Cycle methods: {dict(method_counts)}")

    clip_ratios = [e.get("clip_ratio", 0.0) for e in meta]
    clipped_count = sum(1 for r in clip_ratios if r > 0)
    check(
        "Clipped poses",
        clipped_count < len(meta),
        f"{clipped_count}/{len(meta)} poses clipped ({100*clipped_count/len(meta):.1f}%)",
    )

    support_counts = Counter(e.get("support_foot", "") for e in meta)
    print(f"  Support foot: {dict(support_counts)}")

    all_have_upper_body_default = all(e.get("upper_body_filled_with_default", False) for e in meta)
    check("All poses have upper_body_filled_with_default", all_have_upper_body_default)

    phase_values = [e.get("phase_value", -1) for e in meta]
    check("All phase values in [0, 1]", all(0.0 <= pv <= 1.0 + 1e-6 for pv in phase_values))

    print(f"\n  Result: {checks_passed} passed, {checks_failed} failed")

    print(f"\n{'=' * 60}")
    print("  Joint Statistics (degrees)")
    print(f"{'=' * 60}")
    print(f"  {'Joint':16s}  {'Min':>8s}  {'Max':>8s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"  {'-' * 56}")
    for j in range(min(poses.shape[1], NUM_TOTAL_JOINTS)):
        col = poses[:, j]
        name = JOINT_DISPLAY_NAMES[j] if j < len(JOINT_DISPLAY_NAMES) else f"joint_{j}"
        print(
            f"  {name:16s}  {np.degrees(col.min()):+8.2f}  "
            f"{np.degrees(col.max()):+8.2f}  {np.degrees(col.mean()):+8.2f}  "
            f"{np.degrees(col.std()):8.2f}"
        )

    print(f"\n{'=' * 60}")
    print("  Clip Ratio Distribution")
    print(f"{'=' * 60}")
    if clip_ratios:
        cr_arr = np.array(clip_ratios)
        print(f"  Mean:   {cr_arr.mean():.6f}")
        print(f"  Max:    {cr_arr.max():.6f}")
        print(f"  Clipped poses: {clipped_count}/{len(meta)} ({100*clipped_count/len(meta):.1f}%)")

    print(f"\n{'=' * 60}")
    print("  Sample Metadata (first 3 poses)")
    print(f"{'=' * 60}")
    for entry in meta[:3]:
        print(
            f"  pose_id={entry['pose_id']}, source={entry['source_npz']}, "
            f"cycle={entry['cycle_id']}, phase={entry['phase_index']}, "
            f"clip_ratio={entry['clip_ratio']:.4f}, "
            f"support={entry['support_foot']}, swing={entry['swing_foot']}"
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


def _plot_human_vs_g1_comparison(motions, selector, segmenter, mapper, plots_dir, plt):
    motion = motions[0]
    lower_body = selector.select(motion)
    cycles = segmenter.segment(lower_body)
    if not cycles:
        print("  [SKIP] No cycles found for comparison plot.")
        return

    cycle = cycles[0]
    frames, _, _ = resample_cycle_to_16_phases(lower_body, cycle)

    cmu_start = cycle["start_frame"]
    cmu_end = cycle["end_frame"]
    orig_time = np.arange(cmu_start, cmu_end + 1) / float(lower_body["fps"])
    phase_time = np.linspace(orig_time[0], orig_time[-1], NUM_PHASES)

    comparisons = [
        ("Left Hip Pitch", "lfemur_rx", 0),
        ("Left Knee", "ltibia_rx", 3),
        ("Left Ankle Pitch", "lfoot_rx", 4),
        ("Right Hip Pitch", "rfemur_rx", 6),
        ("Right Knee", "rtibia_rx", 9),
        ("Right Ankle Pitch", "rfoot_rx", 10),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"CMU Raw vs G1 Mapped -- {motion['source_npz']} cycle 0",
        fontsize=14,
    )

    for idx, (title, cmu_ch, g1_idx) in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        raw = lower_body["channels"][cmu_ch][cmu_start : cmu_end + 1]
        ax.plot(orig_time, np.degrees(raw), "b-", alpha=0.5, linewidth=1, label="CMU raw (deg)")

        mapped_vals = []
        for frame in frames:
            pose, _ = mapper.map_frame_to_g1_pose(frame)
            mapped_vals.append(float(pose[g1_idx]))

        ax.plot(phase_time, np.degrees(mapped_vals), "ro-", markersize=5, linewidth=1.5, label="G1 mapped (deg)")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "human_vs_g1_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_phase_discretization(motions, selector, segmenter, mapper, plots_dir, plt):
    motion = motions[0]
    lower_body = selector.select(motion)
    cycles = segmenter.segment(lower_body)
    if not cycles:
        return

    cycle = cycles[0]
    frames, _, phase_values = resample_cycle_to_16_phases(lower_body, cycle)

    cmu_start = cycle["start_frame"]
    cmu_end = cycle["end_frame"]
    norm_phase = np.linspace(0, 1, cmu_end - cmu_start + 1)
    phase_vals_16 = np.array(phase_values)

    key_joints = [
        ("L hip pitch", 0, "lfemur_rx"),
        ("L knee", 3, "ltibia_rx"),
        ("L ankle pitch", 4, "lfoot_rx"),
        ("R hip pitch", 6, "rfemur_rx"),
        ("R knee", 9, "rtibia_rx"),
        ("R ankle pitch", 10, "rfoot_rx"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Phase Discretization -- Continuous vs 16-Phase Sampling", fontsize=14)

    for idx, (name, g1_idx, cmu_ch) in enumerate(key_joints):
        ax = axes[idx // 2, idx % 2]
        raw = lower_body["channels"][cmu_ch][cmu_start : cmu_end + 1]
        ax.plot(norm_phase, np.degrees(raw), "b-", alpha=0.4, linewidth=1, label="Continuous (raw)")

        mapped_16 = []
        for frame in frames:
            pose, _ = mapper.map_frame_to_g1_pose(frame)
            mapped_16.append(float(pose[g1_idx]))
        mapped_16 = np.array(mapped_16)

        ax.plot(phase_vals_16, np.degrees(mapped_16), "ro-", markersize=6, linewidth=1.5, label="16-phase (mapped)")
        ax.set_title(name)
        ax.set_xlabel("Normalized phase")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "phase_discretization.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_joint_limit_distribution(poses, plots_dir, plt):
    leg_joints = [
        ("L hip pitch", 0),
        ("L hip roll", 1),
        ("L knee", 3),
        ("L ankle pitch", 4),
        ("R hip pitch", 6),
        ("R hip roll", 7),
        ("R knee", 9),
        ("R ankle pitch", 10),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Joint Value Distribution vs Limits", fontsize=14)

    for idx, (name, g1_idx) in enumerate(leg_joints):
        ax = axes[idx // 4, idx % 4]
        col = poses[:, g1_idx]
        ax.hist(np.degrees(col), bins=30, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)

        joint_name = G1_JOINT_NAMES[g1_idx]
        if joint_name in G1_JOINT_LIMITS:
            lo, hi = G1_JOINT_LIMITS[joint_name]
            ax.axvline(
                np.degrees(lo), color="red", linestyle="--", linewidth=1.5, label=f"limit ({np.degrees(lo):.0f} deg)"
            )
            ax.axvline(np.degrees(hi), color="red", linestyle="--", linewidth=1.5)

        ax.set_title(name)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "joint_limit_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def run_visualize(input_dir: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 3: Visualize Human G1 Mapping")
    print("=" * 60)

    plt = _get_matplotlib()
    if plt is None:
        print("\n[SKIP] matplotlib not installed, skipping visualization.")
        print("  Install with: pip install matplotlib")
        return

    npy_path = os.path.join(output_dir, "human_g1_candidate_poses.npy")
    if not os.path.exists(npy_path):
        print(f"\n[ERROR] Pose file not found: {npy_path}")
        return

    poses = np.load(npy_path)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    loader = ProcessedCMUGaitLoader()
    motions = loader.load_directory(input_dir)
    selector = CMULowerBodySelector()
    segmenter = GaitCycleSegmenter()
    mapper = HumanToG1PoseMapper()

    print("\n[INFO] Generating CMU vs G1 comparison plot...")
    _plot_human_vs_g1_comparison(motions, selector, segmenter, mapper, plots_dir, plt)

    print("[INFO] Generating phase discretization plot...")
    _plot_phase_discretization(motions, selector, segmenter, mapper, plots_dir, plt)

    print("[INFO] Generating joint limit distribution plot...")
    _plot_joint_limit_distribution(poses, plots_dir, plt)

    print(f"\n[DONE] All plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, inspect, and visualize human-gait-based G1-29dof candidate poses.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/human_gait/processed/cmu_subject_07",
        help="Directory with processed CMU .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pose_library",
        help="Output directory for .npy, .json, and plots.",
    )
    parser.add_argument(
        "--num_phases",
        type=int,
        default=16,
        help="Number of phases per gait cycle (default: 16, aligned with stage1).",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only run axis analysis, then exit.",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip axis analysis, use current mapping config.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip visualization plots (no matplotlib dependency).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Human Gait -> G1 Candidate Pose Pipeline")
    print("=" * 60)

    if args.analyze_only:
        loader = ProcessedCMUGaitLoader()
        motions = loader.load_directory(args.input_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        analyze_cmu_axis_importance(motions, output_path=os.path.join(args.output_dir, "cmu_axis_analysis.json"))
        print("\n[DONE] Analysis complete. Exiting.")
        return

    poses, meta = run_build(args)

    run_inspect(args.output_dir)

    if not args.skip_plots:
        run_visualize(args.input_dir, args.output_dir)

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  Output dir: {args.output_dir}")
    print(f"  Poses:      {args.output_dir}/human_g1_candidate_poses.npy ({poses.shape})")
    print(f"  Metadata:   {args.output_dir}/human_g1_candidate_poses_meta.json ({len(meta)} entries)")
    if not args.skip_plots:
        print(f"  Plots:      {args.output_dir}/plots/")
    print()


if __name__ == "__main__":
    main()
