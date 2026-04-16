"""Fuse stage1 pose library with human mocap candidates, then inspect and compare.

Runs the full fusion pipeline:
  1. Fuse     -- filter, diagnose, score, dedup, budget, merge, export
  2. Inspect  -- integrity checks on the expanded library
  3. Compare  -- original stage1 vs expanded
  4. Visualize -- key joint overlay, phase coverage, score distribution, diagnostics

This script runs purely offline -- no Isaac Sim or GPU required.

Usage::

    python scripts/mslpo/fuse_pose_library.py

    python scripts/mslpo/fuse_pose_library.py \\
        --dedup_s1_threshold 0.40 \\
        --dedup_hh_threshold 0.25 \\
        --max_per_phase 3 \\
        --max_human_total 80

    python scripts/mslpo/fuse_pose_library.py --skip_plots
    python scripts/mslpo/fuse_pose_library.py --skip_inspect
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "source" / "unitree_rl_lab"))

from unitree_rl_lab.controllers.simbicon.pose_library_fusion import (
    G1_JOINT_LIMITS,
    G1_JOINT_NAMES,
    KEY_JOINT_INDICES,
    KEY_JOINT_NAMES_SHORT,
    NUM_TOTAL_JOINTS,
    FusionConfig,
    PoseDistanceMetric,
    PoseLibraryLoader,
    build_expanded_pose_library,
    save_fusion_report,
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
# Step 1: Fuse
# ---------------------------------------------------------------------------


def run_fuse(args: argparse.Namespace) -> tuple[np.ndarray, list[dict], dict]:
    print("\n" + "=" * 60)
    print("  Step 1: Fuse Pose Libraries")
    print("=" * 60)

    config = FusionConfig(
        stage1_vs_human_dedup_threshold=args.dedup_s1_threshold,
        human_vs_human_dedup_threshold=args.dedup_hh_threshold,
        max_human_poses_per_phase=args.max_per_phase,
        hard_max_human_total=args.max_human_total,
    )
    print("\n[CONFIG]")
    print(f"  stage1_vs_human threshold:       {config.stage1_vs_human_dedup_threshold}")
    print(f"  human_vs_human threshold:        {config.human_vs_human_dedup_threshold}")
    print(f"  cross_phase multiplier:          {config.cross_phase_dedup_multiplier}")
    print(f"  max per phase:                   {config.max_human_poses_per_phase}")
    print(f"  soft target:                     {config.soft_target_human_total}")
    print(f"  hard max total:                  {config.hard_max_human_total}")
    print(f"  score weights:                   q={config.w_quality}, n={config.w_novelty}, p={config.w_limit_penalty}")
    print(f"  novelty normalization:           {config.novelty_normalization}")
    print(f"  half_cycle asymmetry:            {config.use_half_cycle_asymmetry}")

    poses, meta, report = build_expanded_pose_library(
        stage1_dir=args.stage1_dir,
        human_dir=args.human_dir,
        output_dir=args.output_dir,
        config=config,
    )

    report_path = save_fusion_report(report, args.output_dir)
    print(f"\n[FUSION REPORT] saved to {report_path}")

    print(f"\n  Stage1:       {report.get('stage1_count', '?')}")
    print(f"  Human input:  {report.get('human_input', '?')}")
    print(f"  Human final:  {report.get('human_final', '?')}")
    print(f"  Total:        {report.get('final_total', '?')}")

    return poses, meta, report


# ---------------------------------------------------------------------------
# Step 2: Inspect
# ---------------------------------------------------------------------------


def run_inspect(output_dir: str) -> bool:
    print("\n" + "=" * 60)
    print("  Step 2: Inspect Expanded Pose Library")
    print("=" * 60)

    npy_path = os.path.join(output_dir, "pose_library_expanded.npy")
    meta_path = os.path.join(output_dir, "pose_library_expanded_meta.json")

    if not os.path.exists(npy_path):
        print(f"\n[ERROR] File not found: {npy_path}")
        return False

    poses = np.load(npy_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

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

    action_indices = [e.get("action_idx", -1) for e in meta]
    expected_indices = list(range(len(meta)))
    check(
        "action_idx continuous 0..N-1",
        action_indices == expected_indices,
        f"expected 0..{len(meta) - 1}, got {action_indices[:5]}...{action_indices[-3:]}",
    )

    source_counts = Counter(e.get("source", "") for e in meta)
    print(f"\n  Source breakdown: {dict(source_counts)}")

    n_stage1 = source_counts.get("stage1", 0)
    check("All stage1 poses preserved", n_stage1 == 160, f"got {n_stage1}")
    check("Total >= 160", len(meta) >= 160, f"got {len(meta)}")

    phase_values = [e.get("phase_value", -1) for e in meta]
    check(
        "All phase_value in [0, 1]",
        all(0.0 <= pv <= 1.0 + 1e-6 for pv in phase_values),
    )

    phase_indices = [e.get("phase_index", -1) for e in meta]
    check(
        "All phase_index in [0, 15]",
        all(0 <= pi <= 15 for pi in phase_indices),
    )

    print(f"\n{'=' * 60}")
    print("  Joint Limit Checks")
    print(f"{'=' * 60}")

    n_over_limit = 0
    for j, name in enumerate(G1_JOINT_NAMES):
        if j >= poses.shape[1]:
            break
        lo, hi = G1_JOINT_LIMITS[name]
        col = poses[:, j]
        violations = int(np.sum((col < lo - 1e-6) | (col > hi + 1e-6)))
        if violations > 0:
            n_over_limit += violations
            print(f"  [WARN] {name}: {violations} poses outside [{lo:.2f}, {hi:.2f}]")
    check("All joints within limits", n_over_limit == 0, f"{n_over_limit} violations total")

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

    print(f"\n  Result: {checks_passed} passed, {checks_failed} failed")
    print("\n[DONE] Inspection complete.")
    return checks_failed == 0


# ---------------------------------------------------------------------------
# Step 3: Compare
# ---------------------------------------------------------------------------


def run_compare(stage1_dir: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 3: Compare Stage1 vs Expanded Library")
    print("=" * 60)

    s1_poses, s1_meta = PoseLibraryLoader.load_stage1(stage1_dir)

    expanded_path = os.path.join(output_dir, "pose_library_expanded.npy")
    expanded_meta_path = os.path.join(output_dir, "pose_library_expanded_meta.json")
    if not os.path.exists(expanded_path):
        print(f"\n[ERROR] Expanded library not found: {expanded_path}")
        return

    ex_poses = np.load(expanded_path).astype(np.float64)
    with open(expanded_meta_path, "r") as f:
        ex_meta = json.load(f)

    n_s1 = len(s1_meta)
    n_ex = len(ex_meta)
    source_counts = Counter(e.get("source", "") for e in ex_meta)
    n_human_in_ex = source_counts.get("human_cmu", 0)

    report_path = os.path.join(output_dir, "pose_library_expanded_report.json")
    human_input = 240
    report_data: dict = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_data = json.load(f)
        human_input = report_data.get("human_input", 240)

    retention_rate = n_human_in_ex / human_input * 100 if human_input > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  Original stage1:         {n_s1}")
    print(f"  Expanded total:          {n_ex}")
    print(f"  Human poses added:       {n_human_in_ex}")
    print(f"  Human input:             {human_input}")
    print(f"  Retention rate:          {retention_rate:.1f}%")
    print(f"  Stage1 proportion:       {n_s1 / n_ex * 100:.1f}%")
    print(f"  Human proportion:        {n_human_in_ex / n_ex * 100:.1f}%")

    soft_target = report_data.get("soft_target_human_total", 56)
    soft_reached = report_data.get("soft_target_reached", False)
    print(f"  Soft target ({soft_target}):   {'reached' if soft_reached else 'NOT reached'}")

    s1_poses_f = s1_poses
    human_mask = np.array([e.get("source") == "human_cmu" for e in ex_meta])
    if human_mask.any():
        human_poses = ex_poses[human_mask]
    else:
        human_poses = np.zeros((0, NUM_TOTAL_JOINTS))

    print(f"\n{'=' * 60}")
    print("  Key Joint Distribution (mean +/- std, degrees)")
    print(f"{'=' * 60}")
    print(f"  {'Joint':16s}  {'S1 mean':>9s}  {'S1 std':>8s}  {'Hu mean':>9s}  {'Hu std':>8s}")
    print(f"  {'-' * 64}")
    for ki, jidx in enumerate(KEY_JOINT_INDICES):
        name = KEY_JOINT_NAMES_SHORT[ki]
        s1_col = s1_poses_f[:, jidx]
        s1_m = np.degrees(s1_col.mean())
        s1_s = np.degrees(s1_col.std())
        if len(human_poses) > 0:
            hu_col = human_poses[:, jidx]
            hu_m = np.degrees(hu_col.mean())
            hu_s = np.degrees(hu_col.std())
        else:
            hu_m, hu_s = 0.0, 0.0
        print(f"  {name:16s}  {s1_m:+9.2f}  {s1_s:8.2f}  {hu_m:+9.2f}  {hu_s:8.2f}")

    print(f"\n{'=' * 60}")
    print("  Per-Phase Human Pose Count")
    print(f"{'=' * 60}")
    phase_human_counts: dict[int, int] = {}
    phase_s1_counts: dict[int, int] = {}
    for e in ex_meta:
        pi = e.get("phase_index", -1)
        if e.get("source") == "human_cmu":
            phase_human_counts[pi] = phase_human_counts.get(pi, 0) + 1
        else:
            phase_s1_counts[pi] = phase_s1_counts.get(pi, 0) + 1
    print(f"  {'Phase':>5s}  {'Stage1':>7s}  {'Human':>6s}  {'Total':>5s}  {'Status':>12s}")
    print(f"  {'-' * 42}")
    budget = report_data.get("budget_report", {})
    cand_counts = budget.get("phase_candidate_counts", {})
    for p in range(16):
        s1c = phase_s1_counts.get(p, 0)
        hc = phase_human_counts.get(p, 0)
        n_cand = cand_counts.get(str(p), "?")
        if hc == 0 and n_cand == 0:
            status = "NO CAND"
        elif hc == 0 and n_cand != 0:
            status = "EMPTIED"
        else:
            status = "OK"
        print(f"  {p:5d}  {s1c:7d}  {hc:6d}  {s1c + hc:5d}  {status:>12s}")

    print(f"\n{'=' * 60}")
    print("  Stage1 Distance Diagnostics")
    print(f"{'=' * 60}")
    dist_stats = report_data.get("nearest_stage1_distance_stats", {})
    if dist_stats:
        print(
            f"  Human→Stage1 min distance: mean={dist_stats.get('mean', '?')}, "
            f"p10={dist_stats.get('percentiles', {}).get('p10', '?')}, "
            f"p50={dist_stats.get('percentiles', {}).get('p50', '?')}, "
            f"p90={dist_stats.get('percentiles', {}).get('p90', '?')}"
        )

    nn = report_data.get("novelty_normalization_info", {})
    if nn:
        print(
            f"  Novelty normalization: {nn.get('method', '?')}, "
            f"lower={nn.get('lower', '?')}, upper={nn.get('upper', '?')}"
        )

    ss = report_data.get("score_stats", {})
    if ss:
        for sk in ("quality_score", "novelty_score", "joint_margin_penalty", "final_score"):
            sv = ss.get(sk, {})
            print(f"  {sk:25s}: mean={sv.get('mean', '?')}, " f"min={sv.get('min', '?')}, max={sv.get('max', '?')}")

    print("\n[DONE] Comparison complete.")


# ---------------------------------------------------------------------------
# Step 4: Visualize
# ---------------------------------------------------------------------------


def _get_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _plot_key_joint_comparison(
    s1_poses: np.ndarray,
    human_poses: np.ndarray,
    plots_dir: str,
    plt,
) -> None:
    n_joints = len(KEY_JOINT_INDICES)
    fig, axes = plt.subplots(2, n_joints // 2, figsize=(20, 10))
    fig.suptitle("Key Joint Distribution: Stage1 vs Human Poses", fontsize=14)

    for ki, jidx in enumerate(KEY_JOINT_INDICES):
        ax = axes[ki // (n_joints // 2), ki % (n_joints // 2)]
        name = KEY_JOINT_NAMES_SHORT[ki]

        s1_vals = np.degrees(s1_poses[:, jidx])
        ax.hist(s1_vals, bins=25, alpha=0.6, color="steelblue", label="stage1", edgecolor="black", linewidth=0.5)

        if len(human_poses) > 0:
            hu_vals = np.degrees(human_poses[:, jidx])
            ax.hist(hu_vals, bins=20, alpha=0.6, color="coral", label="human_cmu", edgecolor="black", linewidth=0.5)

        ax.set_title(name)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "fusion_key_joint_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_phase_coverage(
    ex_meta: list[dict],
    plots_dir: str,
    plt,
) -> None:
    s1_counts = [0] * 16
    hu_counts = [0] * 16

    for e in ex_meta:
        pi = e.get("phase_index", -1)
        if 0 <= pi < 16:
            if e.get("source") == "human_cmu":
                hu_counts[pi] += 1
            else:
                s1_counts[pi] += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(16)
    width = 0.4
    ax.bar(x - width / 2, s1_counts, width, label="stage1", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, hu_counts, width, label="human_cmu", color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Phase Index")
    ax.set_ylabel("Pose Count")
    ax.set_title("Per-Phase Pose Coverage (Stage1 vs Human)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(plots_dir, "fusion_phase_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_score_distribution(
    ex_meta: list[dict],
    plots_dir: str,
    plt,
) -> None:
    quality_scores = [e.get("quality_score", 0) for e in ex_meta if e.get("source") == "human_cmu"]
    novelty_scores = [e.get("novelty_score", 0) for e in ex_meta if e.get("source") == "human_cmu"]
    final_scores = [e.get("final_score", 0) for e in ex_meta if e.get("source") == "human_cmu"]

    if not quality_scores:
        print("  [SKIP] No human scores to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Human Pose Score Distribution", fontsize=14)

    for ax, data, name, color in [
        (axes[0], quality_scores, "Quality Score", "steelblue"),
        (axes[1], novelty_scores, "Novelty Score", "coral"),
        (axes[2], final_scores, "Final Score", "seagreen"),
    ]:
        ax.hist(data, bins=20, alpha=0.8, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        mean_val = np.mean(data)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "fusion_score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_nearest_stage1_distance_hist(
    stage1_dir: str,
    output_dir: str,
    plots_dir: str,
    plt,
) -> None:
    diag_path = os.path.join(output_dir, "nearest_stage1_distances.json")
    if not os.path.exists(diag_path):
        print("  [SKIP] nearest_stage1_distances.json not found.")
        return

    s1_poses, _ = PoseLibraryLoader.load_stage1(stage1_dir)
    try:
        h_poses, _ = PoseLibraryLoader.load_human_candidates(output_dir)
    except FileNotFoundError:
        h_poses = None

    if h_poses is None:
        h_npy = os.path.join(output_dir, "human_g1_candidate_poses.npy")
        if not os.path.exists(h_npy):
            print("  [SKIP] No human poses for distance histogram.")
            return
        h_poses = np.load(h_npy).astype(np.float64)

    metric = PoseDistanceMetric()
    dists = metric.pairwise_distances(h_poses, s1_poses)
    min_dists = dists.min(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(min_dists, bins=30, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    p10 = np.percentile(min_dists, 10)
    p50 = np.percentile(min_dists, 50)
    p90 = np.percentile(min_dists, 90)
    ax.axvline(p10, color="green", linestyle="--", linewidth=1.5, label=f"p10={p10:.3f}")
    ax.axvline(p50, color="orange", linestyle="--", linewidth=1.5, label=f"p50={p50:.3f}")
    ax.axvline(p90, color="red", linestyle="--", linewidth=1.5, label=f"p90={p90:.3f}")
    ax.set_xlabel("Min Distance to Nearest Stage1 Pose (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Human → Nearest Stage1 Distance Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "human_to_stage1_nearest_distance_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_phase_candidate_vs_kept(
    output_dir: str,
    plots_dir: str,
    plt,
) -> None:
    report_path = os.path.join(output_dir, "pose_library_expanded_report.json")
    if not os.path.exists(report_path):
        print("  [SKIP] Report not found for candidate vs kept plot.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    budget = report.get("budget_report", {})
    cand = budget.get("phase_candidate_counts", {})
    kept = budget.get("phase_kept_counts", {})

    cand_vals = [cand.get(str(p), 0) for p in range(16)]
    kept_vals = [kept.get(str(p), 0) for p in range(16)]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(16)
    width = 0.35
    ax.bar(x - width / 2, cand_vals, width, label="Candidates", color="lightgray", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, kept_vals, width, label="Kept", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Phase Index")
    ax.set_ylabel("Count")
    ax.set_title("Per-Phase: Candidates (post-dedup) vs Kept (post-budget)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(plots_dir, "phase_candidate_vs_kept.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_joint_margin_penalty(
    output_dir: str,
    plots_dir: str,
    plt,
) -> None:
    meta_path = os.path.join(output_dir, "pose_library_expanded_meta.json")
    if not os.path.exists(meta_path):
        print("  [SKIP] Expanded metadata not found.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    penalties = [e.get("joint_margin_penalty", 0) for e in meta if e.get("source") == "human_cmu"]
    if not penalties:
        print("  [SKIP] No human poses with margin penalty data.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(penalties, bins=20, alpha=0.7, color="coral", edgecolor="black", linewidth=0.5)
    mean_val = np.mean(penalties)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.3f}")
    ax.set_xlabel("Joint Margin Penalty")
    ax.set_ylabel("Count")
    ax.set_title("Joint Margin Penalty Distribution (Human Poses)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "joint_margin_penalty_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def run_visualize(stage1_dir: str, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("  Step 4: Visualize Fusion Results")
    print("=" * 60)

    plt = _get_matplotlib()
    if plt is None:
        print("\n[SKIP] matplotlib not installed.")
        print("  Install with: pip install matplotlib")
        return

    expanded_path = os.path.join(output_dir, "pose_library_expanded.npy")
    if not os.path.exists(expanded_path):
        print(f"\n[ERROR] Expanded library not found: {expanded_path}")
        return

    ex_poses = np.load(expanded_path).astype(np.float64)
    with open(os.path.join(output_dir, "pose_library_expanded_meta.json"), "r") as f:
        ex_meta = json.load(f)

    s1_poses, _ = PoseLibraryLoader.load_stage1(stage1_dir)

    human_mask = np.array([e.get("source") == "human_cmu" for e in ex_meta])
    human_poses = ex_poses[human_mask] if human_mask.any() else np.zeros((0, NUM_TOTAL_JOINTS))

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n[INFO] Generating key joint comparison...")
    _plot_key_joint_comparison(s1_poses, human_poses, plots_dir, plt)

    print("[INFO] Generating phase coverage plot...")
    _plot_phase_coverage(ex_meta, plots_dir, plt)

    print("[INFO] Generating score distribution plot...")
    _plot_score_distribution(ex_meta, plots_dir, plt)

    print("[INFO] Generating human→stage1 distance histogram...")
    _plot_nearest_stage1_distance_hist(stage1_dir, output_dir, plots_dir, plt)

    print("[INFO] Generating phase candidate vs kept plot...")
    _plot_phase_candidate_vs_kept(output_dir, plots_dir, plt)

    print("[INFO] Generating joint margin penalty distribution...")
    _plot_joint_margin_penalty(output_dir, plots_dir, plt)

    print(f"\n[DONE] All plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse stage1 and human pose libraries, inspect, compare, and visualize.",
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        default="outputs/pose_library",
        help="Directory containing pose_library.npy and meta.",
    )
    parser.add_argument(
        "--human_dir",
        type=str,
        default="outputs/pose_library",
        help="Directory containing human_g1_candidate_poses.npy and meta.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pose_library",
        help="Output directory for expanded library.",
    )
    parser.add_argument(
        "--dedup_s1_threshold",
        type=float,
        default=0.40,
        help="Stage1 vs human dedup threshold (weighted L2, rad).",
    )
    parser.add_argument(
        "--dedup_hh_threshold",
        type=float,
        default=0.25,
        help="Human vs human within-phase dedup threshold (weighted L2, rad).",
    )
    parser.add_argument(
        "--max_per_phase",
        type=int,
        default=3,
        help="Max human poses per phase index.",
    )
    parser.add_argument(
        "--max_human_total",
        type=int,
        default=80,
        help="Hard max total human poses.",
    )
    parser.add_argument("--skip_plots", action="store_true", help="Skip visualization.")
    parser.add_argument("--skip_inspect", action="store_true", help="Skip inspection step.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pose Library Fusion Pipeline")
    print("=" * 60)

    poses, meta, report = run_fuse(args)

    if not args.skip_inspect:
        run_inspect(args.output_dir)

    run_compare(args.stage1_dir, args.output_dir)

    if not args.skip_plots:
        run_visualize(args.stage1_dir, args.output_dir)

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  Output dir: {args.output_dir}")
    print(f"  Poses:      {args.output_dir}/pose_library_expanded.npy ({poses.shape})")
    print(f"  Metadata:   {args.output_dir}/pose_library_expanded_meta.json ({len(meta)} entries)")
    print(f"  Report:     {args.output_dir}/pose_library_expanded_report.json")
    print(f"  Diagnostics:{args.output_dir}/nearest_stage1_distances.json")
    if not args.skip_plots:
        print(f"  Plots:      {args.output_dir}/plots/")
    print()


if __name__ == "__main__":
    main()
