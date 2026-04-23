"""Generate Stage 1 analysis report with data tables, statistics, and figures.

Reads all existing Stage 1 experimental outputs (Q-learning, dynamic
discretization, sensitivity scan, pose library, fusion report) and produces
a unified set of CSV / JSON / Markdown summaries and publication-ready
figures for the thesis chapter "Stage 1 Experimental Results and Analysis".

This script runs purely offline -- no Isaac Sim or GPU required.

Usage:
    python scripts/mslpo/build_stage1_analysis_report.py
    python scripts/mslpo/build_stage1_analysis_report.py --project_root /path/to/unitree_rl_lab
"""

from __future__ import annotations

import argparse
import csv
import json
import matplotlib
import statistics
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_BASE / "analysis_stage1"
FIGURES_DIR = ANALYSIS_DIR / "figures"

DPI = 150
FIG_WIDE = (10, 5)
FIG_SQUARE = (7, 5)
FIG_PIPELINE = (14, 4.5)

COLORS = {
    "primary": "#2C5F8A",
    "secondary": "#D45B3F",
    "tertiary": "#5BA35B",
    "quaternary": "#E8A838",
    "quinary": "#8E6BB0",
    "gray": "#888888",
    "light_gray": "#CCCCCC",
    "bg": "#F5F5F5",
}

PARAM_COLORS = {
    "HL": COLORS["primary"],
    "Ls": COLORS["secondary"],
    "Lswb": COLORS["tertiary"],
    "Lforward": COLORS["quaternary"],
}

PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"], COLORS["quinary"]]


def _setup_chinese_font() -> str:
    candidates = [
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans SC",
        "AR PL UMing CN",
    ]
    from matplotlib.font_manager import fontManager

    available = {f.name for f in fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams.get(
                "font.sans-serif", ["DejaVu Sans"]
            )
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    warnings.warn("No Chinese font found; falling back to default sans-serif.")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "default"


FONT_NAME = ""


def _save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        warnings.warn(f"Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _safe_stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Part 1: Parameter Optimization Results
# ---------------------------------------------------------------------------


def _load_top_parameters() -> list[dict[str, Any]]:
    uniform_top5 = _load_json(OUTPUT_BASE / "qlearn_search" / "top5_pose_params.json") or []
    dynamic_top5 = _load_json(OUTPUT_BASE / "dynamic_discretization" / "top5_pose_params_dynamic.json") or []
    combined = []
    for item in uniform_top5:
        item = dict(item)
        item["source"] = "uniform"
        combined.append(item)
    for item in dynamic_top5:
        item = dict(item)
        item["source"] = "dynamic"
        combined.append(item)
    combined.sort(key=lambda x: x.get("total_reward", 0), reverse=True)
    for i, item in enumerate(combined):
        item["overall_rank"] = i + 1
    return combined


def _export_top_parameter_sets(params: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "top_parameter_sets.csv"
    fields = [
        "overall_rank",
        "source",
        "rank",
        "HL",
        "Ls",
        "Lswb",
        "Lforward",
        "total_reward",
        "avg_forward_velocity",
        "avg_lateral_offset",
        "alive_time",
    ]
    rename = {"rank": "source_rank"}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in params:
            out = {}
            for k in fields:
                key = rename.get(k, k)
                out[k] = row.get(key, row.get(k, ""))
            writer.writerow(out)
    json_path = ANALYSIS_DIR / "top_parameter_sets.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")


def _export_parameter_distribution_stats(params: list[dict[str, Any]]) -> None:
    top5 = params[:5]
    top10 = params[:10]
    result = {}
    for pname in ("HL", "Ls", "Lswb", "Lforward"):
        vals_top5 = [p[pname] for p in top5]
        vals_top10 = [p[pname] for p in top10 if p.get(pname) is not None]
        result[pname] = {
            "top5": _safe_stat(vals_top5),
            "top10": _safe_stat(vals_top10) if vals_top10 else {},
            "best_value": params[0].get(pname),
        }
    out_path = ANALYSIS_DIR / "parameter_distribution_stats.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")


def _plot_top5_parameter_comparison(params: list[dict[str, Any]]) -> None:
    top5 = params[:5]
    labels = [f"G{i+1}" for i in range(5)]
    param_names = ["HL", "Ls", "Lswb", "Lforward"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for i, pname in enumerate(param_names):
        vals = [p[pname] for p in top5]
        ax.bar(x + (i - 1.5) * width, vals, width, label=pname, color=PALETTE[i], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("参数组", fontsize=11)
    ax.set_ylabel("参数取值", fontsize=11)
    ax.set_title("Top-5 参数组对比", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig01_top5_parameter_comparison")


def _plot_top5_performance_comparison(params: list[dict[str, Any]]) -> None:
    top5 = params[:5]
    labels = [f"G{i+1}" for i in range(5)]
    metrics = [
        ("total_reward", "累计奖励 (total_reward)", COLORS["primary"]),
        ("avg_forward_velocity", "平均前向速度 (m/s)", COLORS["secondary"]),
        ("avg_lateral_offset", "平均横向偏移 (m)", COLORS["tertiary"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (key, ylabel, color) in zip(axes, metrics):
        vals = [p.get(key, 0) for p in top5]
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8
            )
    fig.suptitle("Top-5 参数组性能指标对比", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig02_top5_performance_comparison")


# ---------------------------------------------------------------------------
# Part 2: Dynamic Discretization Analysis
# ---------------------------------------------------------------------------


def _export_dynamic_vs_uniform() -> dict[str, Any]:
    report = _load_json(OUTPUT_BASE / "dynamic_discretization" / "dynamic_vs_uniform_report.json")
    if not report:
        warnings.warn("dynamic_vs_uniform_report.json not found, skipping.")
        return {}
    flat = {
        "uniform_action_space_size": report.get("uniform_action_space_size"),
        "dynamic_action_space_size_before_tightening": report.get("dynamic_action_space_size_before_tightening"),
        "dynamic_action_space_size_after_tightening": report.get("dynamic_action_space_size_after_tightening"),
        "action_space_reduction_percent": report.get("action_space_reduction_percent"),
        "uniform_best_total_reward": report.get("uniform_best_total_reward"),
        "dynamic_best_total_reward": report.get("dynamic_best_total_reward"),
        "uniform_best_forward_velocity": report.get("uniform_best_forward_velocity"),
        "dynamic_best_forward_velocity": report.get("dynamic_best_forward_velocity"),
        "uniform_best_lateral_offset": report.get("uniform_best_lateral_offset"),
        "dynamic_best_lateral_offset": report.get("dynamic_best_lateral_offset"),
        "uniform_best_time_to_fall": report.get("uniform_best_time_to_fall"),
        "dynamic_best_time_to_fall": report.get("dynamic_best_time_to_fall"),
        "uniform_episodes_to_target_reward": report.get("uniform_episodes_to_target_reward"),
        "dynamic_episodes_to_target_reward": report.get("dynamic_episodes_to_target_reward"),
        "uniform_episodes_to_target_velocity": report.get("uniform_episodes_to_target_velocity"),
        "dynamic_episodes_to_target_velocity": report.get("dynamic_episodes_to_target_velocity"),
        "uniform_search_time_only_s": report.get("uniform_search_time_only"),
        "dynamic_search_time_only_s": report.get("dynamic_search_time_only"),
        "dynamic_total_time_including_scan_s": report.get("total_time_including_scan"),
        "efficiency_improvement_percent": report.get("efficiency_improvement_percent"),
    }
    csv_path = ANALYSIS_DIR / "dynamic_vs_uniform_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    json_path = ANALYSIS_DIR / "dynamic_vs_uniform_summary.json"
    with open(json_path, "w") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
    return flat


def _export_dynamic_discretization_summary() -> dict[str, Any]:
    config = _load_json(OUTPUT_BASE / "dynamic_discretization" / "dynamic_discretization_config.json")
    if not config:
        return {}
    result = {}
    params_section = config.get("parameters", {})
    for pname, pconf in params_section.items():
        result[pname] = {
            "original_range": pconf.get("original_range"),
            "focus_zones": pconf.get("focus_zones"),
            "discrete_values": pconf.get("discrete_values"),
            "num_discrete_values": len(pconf.get("discrete_values", [])),
        }
    out_path = ANALYSIS_DIR / "dynamic_discretization_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")
    return result


def _plot_action_space_comparison(dv_report: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["均匀离散", "动态离散\n(紧缩后)"]
    uniform_size = dv_report.get("uniform_action_space_size") or 2149056
    dynamic_size = dv_report.get("dynamic_action_space_size_after_tightening") or 193648
    vals = [uniform_size, dynamic_size]
    bars = ax.bar(labels, vals, color=[COLORS["gray"], COLORS["primary"]], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,}", ha="center", va="bottom", fontsize=10)
    reduction = dv_report.get("action_space_reduction_percent")
    if reduction is not None:
        ax.annotate(
            f"缩减 {reduction:.1f}%",
            xy=(1, dynamic_size),
            xytext=(0.5, uniform_size * 0.6),
            fontsize=10,
            color=COLORS["secondary"],
            arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
            ha="center",
        )
    ax.set_ylabel("动作空间大小", fontsize=11)
    ax.set_title("动作空间规模对比", fontsize=13)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig03_action_space_comparison")


def _plot_episodes_to_target_reward(dv_report: dict) -> None:
    u_ep = dv_report.get("uniform_episodes_to_target_reward")
    d_ep = dv_report.get("dynamic_episodes_to_target_reward")
    if u_ep is None and d_ep is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["均匀离散", "动态离散"]
    vals = [u_ep if u_ep is not None else 0, d_ep if d_ep is not None else 0]
    bars = ax.bar(labels, vals, color=[COLORS["gray"], COLORS["primary"]], edgecolor="white", width=0.5)
    for bar, v, raw in zip(bars, vals, [u_ep, d_ep]):
        txt = str(raw) if raw is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 1), txt, ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("达到目标奖励的回合数", fontsize=11)
    ax.set_title("达到目标奖励回合数对比", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig04_episodes_to_target_reward")


def _plot_episodes_to_target_velocity(dv_report: dict) -> None:
    u_ep = dv_report.get("uniform_episodes_to_target_velocity")
    d_ep = dv_report.get("dynamic_episodes_to_target_velocity")
    if u_ep is None and d_ep is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["均匀离散", "动态离散"]
    vals = [u_ep if u_ep is not None else 0, d_ep if d_ep is not None else 0]
    bars = ax.bar(labels, vals, color=[COLORS["gray"], COLORS["primary"]], edgecolor="white", width=0.5)
    for bar, v, raw in zip(bars, vals, [u_ep, d_ep]):
        txt = str(raw) if raw is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 1), txt, ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("达到目标速度的回合数", fontsize=11)
    ax.set_title("达到目标速度回合数对比", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig05_episodes_to_target_velocity")


def _plot_best_performance_comparison(dv_report: dict) -> None:
    metrics = [
        ("uniform_best_total_reward", "dynamic_best_total_reward", "最优累计奖励"),
        ("uniform_best_forward_velocity", "dynamic_best_forward_velocity", "最优前向速度 (m/s)"),
        ("uniform_best_lateral_offset", "dynamic_best_lateral_offset", "最优横向偏移 (m)"),
        ("uniform_best_time_to_fall", "dynamic_best_time_to_fall", "最长存活时间 (s)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    labels = ["均匀", "动态"]
    for ax, (u_key, d_key, title) in zip(axes, metrics):
        u_val = dv_report.get(u_key)
        d_val = dv_report.get(d_key)
        plot_vals = [u_val if u_val is not None else 0, d_val if d_val is not None else 0]
        bars = ax.bar(labels, plot_vals, color=[COLORS["gray"], COLORS["primary"]], edgecolor="white", width=0.5)
        for bar, raw in zip(bars, [u_val, d_val]):
            txt = f"{raw:.4f}" if raw is not None else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0.001),
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("均匀 vs 动态离散化最优性能对比", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig06_best_performance_comparison")


# ---------------------------------------------------------------------------
# Part 3: Sensitivity Scan Curves
# ---------------------------------------------------------------------------


def _load_sensitivity_scan() -> dict[str, list[dict]]:
    scan = _load_json(OUTPUT_BASE / "qlearn_search" / "parameter_sensitivity_scan.json")
    if scan is None:
        scan = _load_json(OUTPUT_BASE / "dynamic_discretization" / "parameter_sensitivity_scan.json")
    return scan or {}


def _get_focus_zones(pname: str, dd_summary: dict) -> list[list[float]]:
    if pname in dd_summary:
        return dd_summary[pname].get("focus_zones", [])
    return []


def _plot_sensitivity_curve(
    pname: str,
    scan_data: list[dict],
    focus_zones: list[list[float]],
    fig_num: int,
) -> None:
    if not scan_data:
        return
    sorted_data = sorted(scan_data, key=lambda x: x["param_value"])
    x_vals = [d["param_value"] for d in sorted_data]
    composite = [d.get("composite_score", 0) for d in sorted_data]
    reward = [d.get("mean_total_reward", 0) for d in sorted_data]

    fig, ax1 = plt.subplots(figsize=FIG_SQUARE)
    color1 = COLORS["primary"]
    color2 = COLORS["secondary"]
    ax1.set_xlabel(f"{pname} 参数取值", fontsize=11)
    ax1.set_ylabel("综合评分 (composite_score)", color=color1, fontsize=10)
    (line1,) = ax1.plot(x_vals, composite, "o-", color=color1, markersize=4, linewidth=1.5, label="综合评分")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("平均累计奖励", color=color2, fontsize=10)
    (line2,) = ax2.plot(x_vals, reward, "s--", color=color2, markersize=4, linewidth=1.5, label="平均累计奖励")
    ax2.tick_params(axis="y", labelcolor=color2)

    for zone in focus_zones:
        if len(zone) >= 2:
            ax1.axvspan(zone[0], zone[-1], alpha=0.15, color=COLORS["tertiary"], label="重点区间")

    lines = [line1, line2]
    if focus_zones:
        zone_patch = mpatches.Patch(color=COLORS["tertiary"], alpha=0.15, label="重点区间")
        lines.append(zone_patch)
    ax1.legend(handles=lines, loc="upper right", fontsize=8)
    ax1.set_title(f"参数 {pname} 敏感性扫描曲线", fontsize=13)
    ax1.grid(alpha=0.3)
    _save_fig(fig, f"fig{fig_num:02d}_parameter_{pname}_sensitivity_curve")


# ---------------------------------------------------------------------------
# Part 4: Pose Library Analysis
# ---------------------------------------------------------------------------


def _export_original_pose_library_summary() -> dict[str, Any]:
    meta = _load_json(OUTPUT_BASE / "pose_library" / "pose_library_meta.json")
    if not meta:
        return {}
    num_params = len({m["param_group_idx"] for m in meta})
    num_states = len({m["fsm_state"] for m in meta})
    phases_per_state = len({m["phase_index"] for m in meta if m.get("fsm_state") == meta[0]["fsm_state"]})
    poses_per_group = Counter(m["param_group_idx"] for m in meta)
    phase_counts = Counter(m["phase_index"] for m in meta)
    groups_info = []
    for gidx in sorted(set(m["param_group_idx"] for m in meta)):
        entries = [m for m in meta if m["param_group_idx"] == gidx]
        groups_info.append(
            {
                "param_group_idx": gidx,
                "rank": entries[0].get("rank"),
                "HL": entries[0].get("HL"),
                "Ls": entries[0].get("Ls"),
                "Lswb": entries[0].get("Lswb"),
                "Lforward": entries[0].get("Lforward"),
                "total_reward": entries[0].get("total_reward"),
                "num_poses": len(entries),
            }
        )
    result = {
        "total_poses": len(meta),
        "num_parameter_groups": num_params,
        "num_core_states": num_states,
        "phases_per_state": phases_per_state,
        "poses_per_group": dict(poses_per_group),
        "phase_pose_counts": {str(k): v for k, v in sorted(phase_counts.items())},
        "parameter_groups": groups_info,
    }
    out_path = ANALYSIS_DIR / "original_pose_library_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")
    return result


def _export_expanded_pose_library_summary() -> dict[str, Any]:
    expanded_report = _load_json(OUTPUT_BASE / "pose_library" / "pose_library_expanded_report.json") or {}
    expanded_meta = _load_json(OUTPUT_BASE / "pose_library" / "pose_library_expanded_meta.json") or []

    stage1_entries = [m for m in expanded_meta if m.get("source") == "stage1"]
    human_entries = [m for m in expanded_meta if m.get("source") == "human_cmu"]

    stage1_phase_counts = Counter(m["phase_index"] for m in stage1_entries)
    human_phase_counts = Counter(m["phase_index"] for m in human_entries)
    all_phases = sorted(set(list(stage1_phase_counts.keys()) + list(human_phase_counts.keys())))
    phase_coverage = []
    for p in all_phases:
        phase_coverage.append(
            {
                "phase": p,
                "stage1_count": stage1_phase_counts.get(p, 0),
                "human_count": human_phase_counts.get(p, 0),
                "total": stage1_phase_counts.get(p, 0) + human_phase_counts.get(p, 0),
            }
        )

    budget_report = expanded_report.get("budget_report", {})
    s1_dedup = expanded_report.get("stage1_dedup_report", {})
    h_dedup = expanded_report.get("human_dedup_report", {})
    filter_report = expanded_report.get("filter_report", {})

    source_npz_counts = Counter(m.get("source_npz", "unknown") for m in human_entries)

    result = {
        "stage1_count": expanded_report.get("stage1_count", len(stage1_entries)),
        "human_candidate_input": expanded_report.get("human_input", 0),
        "human_final_count": expanded_report.get("human_final", len(human_entries)),
        "final_total": expanded_report.get("final_total", len(expanded_meta)),
        "source_ratio": {
            "stage1": len(stage1_entries),
            "human_cmu": len(human_entries),
            "stage1_percent": round(100 * len(stage1_entries) / max(len(expanded_meta), 1), 1),
            "human_cmu_percent": round(100 * len(human_entries) / max(len(expanded_meta), 1), 1),
        },
        "phase_coverage": phase_coverage,
        "phases_covered": budget_report.get("phases_covered", len(all_phases)),
        "phase_full_coverage": budget_report.get("phase_min_coverage_satisfied", len(all_phases) >= 14),
        "soft_target": budget_report.get("soft_target"),
        "soft_target_reached": budget_report.get("soft_target_reached"),
        "filtering_funnel": {
            "human_input": filter_report.get("total_input", 0),
            "validity_passed": filter_report.get("validity_passed", 0),
            "morphology_passed": filter_report.get("morphology_passed", 0),
            "final_kept_after_filter": filter_report.get("final_kept", 0),
            "stage1_dedup_absorbed": s1_dedup.get("absorbed", 0),
            "after_stage1_dedup": s1_dedup.get("kept", 0),
            "human_dedup_removed": h_dedup.get("removed", 0),
            "after_human_dedup": h_dedup.get("kept", 0),
            "budget_final": budget_report.get("final_count", 0),
        },
        "human_source_npz_distribution": dict(source_npz_counts),
        "dedup_summary": {
            "stage1_vs_human_absorbed": s1_dedup.get("absorbed", 0),
            "human_vs_human_removed": h_dedup.get("removed", 0),
            "within_phase_removed": h_dedup.get("within_phase_removed", 0),
            "cross_phase_removed": h_dedup.get("cross_phase_removed", 0),
        },
        "score_stats": expanded_report.get("score_stats", {}),
        "overlap_assessment": expanded_report.get("stage1_human_overlap_assessment", ""),
        "nearest_stage1_distance_stats": expanded_report.get("nearest_stage1_distance_stats", {}),
        "key_joint_saturated_by_joint": filter_report.get("key_joint_saturated_by_joint", {}),
    }
    out_path = ANALYSIS_DIR / "expanded_pose_library_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")
    return result


def _plot_library_size_comparison(exp_summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    stage1_count = exp_summary.get("stage1_count", 160)
    human_count = exp_summary.get("human_final_count", 56)
    total = exp_summary.get("final_total", 216)
    labels = ["原始姿态库\n(Stage1)", "人类步态\n增量", "扩充姿态库\n(总计)"]
    vals = [stage1_count, human_count, total]
    colors = [COLORS["primary"], COLORS["tertiary"], COLORS["secondary"]]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(v),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("姿态数量", fontsize=11)
    ax.set_title("原始姿态库与扩充姿态库规模对比", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig11_library_size_comparison")


def _plot_source_proportion(exp_summary: dict) -> None:
    stage1 = exp_summary.get("stage1_count", 160)
    human = exp_summary.get("human_final_count", 56)
    total = stage1 + human
    if total == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sizes = [stage1, human]
    labels = [f"Stage1 参数优化\n({stage1})", f"人类CMU步态\n({human})"]
    colors = [COLORS["primary"], COLORS["tertiary"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 10}
    )
    for t in autotexts:
        t.set_fontsize(11)
    ax.set_title("扩充姿态库来源占比", fontsize=13)
    _save_fig(fig, "fig12_source_proportion")


def _plot_phase_coverage(orig_summary: dict, exp_summary: dict) -> None:
    orig_meta = _load_json(OUTPUT_BASE / "pose_library" / "pose_library_meta.json") or []
    exp_meta = _load_json(OUTPUT_BASE / "pose_library" / "pose_library_expanded_meta.json") or []
    orig_s1 = [m for m in orig_meta if m.get("source") == "stage1"]
    exp_human = [m for m in exp_meta if m.get("source") == "human_cmu"]
    orig_phase_counts = Counter(m["phase_index"] for m in orig_s1)
    exp_human_phase_counts = Counter(m["phase_index"] for m in exp_human)
    all_phases = sorted(set(list(orig_phase_counts.keys()) + list(exp_human_phase_counts.keys())))
    x = np.arange(len(all_phases))
    width = 0.35
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    orig_vals = [orig_phase_counts.get(p, 0) for p in all_phases]
    exp_h_vals = [exp_human_phase_counts.get(p, 0) for p in all_phases]
    ax.bar(x - width / 2, orig_vals, width, label="原始 (Stage1)", color=COLORS["primary"], edgecolor="white")
    ax.bar(x + width / 2, exp_h_vals, width, label="扩充 (Human)", color=COLORS["tertiary"], edgecolor="white")
    ax.set_xlabel("步态相位 (Phase)", fontsize=11)
    ax.set_ylabel("姿态数量", fontsize=11)
    ax.set_title("Phase 覆盖柱状图（原始 vs 扩充）", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in all_phases])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig13_phase_coverage")


def _plot_human_filtering_funnel(exp_summary: dict) -> None:
    funnel = exp_summary.get("filtering_funnel", {})
    stages = [
        ("CMU 候选输入", funnel.get("human_input", 0)),
        ("有效性过滤", funnel.get("validity_passed", 0)),
        ("形态学过滤", funnel.get("final_kept_after_filter", 0)),
        ("Stage1 去重", funnel.get("after_stage1_dedup", 0)),
        ("Human 去重", funnel.get("after_human_dedup", 0)),
        ("预算控制\n(最终保留)", funnel.get("budget_final", 0)),
    ]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    if all(v == 0 for v in vals):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(labels))
    colors_funnel = plt.cm.Blues(np.linspace(0.85, 0.35, len(labels)))
    bars = ax.barh(y_pos, vals, color=colors_funnel, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=10)
    ax.set_xlabel("姿态数量", fontsize=11)
    ax.set_title("人类步态姿态筛选流程（漏斗图）", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(vals) * 1.15)
    _save_fig(fig, "fig14_human_filtering_funnel")


def _plot_key_joint_saturation(exp_summary: dict) -> None:
    joint_data = exp_summary.get("key_joint_saturated_by_joint", {})
    if not joint_data:
        return
    joint_names = list(joint_data.keys())
    saturated_counts = [joint_data[j].get("count_saturated", 0) for j in joint_names]
    mean_margins = [joint_data[j].get("mean_margin", 0) for j in joint_names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(joint_names, saturated_counts, color=COLORS["primary"], edgecolor="white")
    ax1.set_xlabel("饱和姿态数", fontsize=10)
    ax1.set_title("关键关节饱和计数", fontsize=12)
    ax1.grid(axis="x", alpha=0.3)
    ax2.barh(joint_names, mean_margins, color=COLORS["tertiary"], edgecolor="white")
    ax2.set_xlabel("平均关节余量 (rad)", fontsize=10)
    ax2.set_title("关键关节平均余量", fontsize=12)
    ax2.grid(axis="x", alpha=0.3)
    fig.suptitle("关键关节饱和统计", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig15_key_joint_saturation")


# ---------------------------------------------------------------------------
# Part 5: Pipeline Overview Figure
# ---------------------------------------------------------------------------


def _plot_pipeline_overview(params: list[dict], dv_report: dict, exp_summary: dict) -> None:
    fig, ax = plt.subplots(figsize=FIG_PIPELINE)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    node_x = [0.5, 1.5, 2.5, 3.5, 4.5]
    node_y = 0.8
    box_w = 0.85
    box_h = 0.9

    best_reward = params[0].get("total_reward", 0) if params else 0
    uniform_size = dv_report.get("uniform_action_space_size", 2149056)
    dynamic_size = dv_report.get("dynamic_action_space_size_after_tightening", 193648)
    reduction = dv_report.get("action_space_reduction_percent", 0)
    human_input = exp_summary.get("filtering_funnel", {}).get("human_input", 240)
    human_final = exp_summary.get("human_final_count", 56)
    final_total = exp_summary.get("final_total", 216)

    nodes = [
        ("参数优化\n(Q-learning)", f"Top-5 参数组\nBest reward={best_reward:.2f}"),
        ("动态离散化\n(敏感性引导)", f"{uniform_size:,}\n→ {dynamic_size:,}\n(↓{reduction:.1f}%)"),
        ("原始姿态库\n(Stage1)", "5组×2状态×16相位\n= 160 姿态"),
        ("人类步态扩充\n(CMU MoCap)", f"{human_input} → {human_final}\n筛选+去重"),
        ("扩充姿态库\n(最终)", f"总计 {final_total} 姿态\n(160+{human_final})"),
    ]

    node_colors = [COLORS["primary"], COLORS["quaternary"], COLORS["tertiary"], COLORS["secondary"], COLORS["quinary"]]

    for i, ((title, detail), nx) in enumerate(zip(nodes, node_x)):
        rect = mpatches.FancyBboxPatch(
            (nx - box_w / 2, node_y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=node_colors[i],
            alpha=0.15,
            edgecolor=node_colors[i],
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(nx, node_y + 0.2, title, ha="center", va="center", fontsize=9, fontweight="bold", color=node_colors[i])
        ax.text(nx, node_y - 0.2, detail, ha="center", va="center", fontsize=7, color="#333333")

    arrow_labels = ["敏感性扫描", "参数→姿态", "融合+评分", "去重+预算"]
    for i in range(4):
        x_start = node_x[i] + box_w / 2 + 0.02
        x_end = node_x[i + 1] - box_w / 2 - 0.02
        ax.annotate(
            "",
            xy=(x_end, node_y),
            xytext=(x_start, node_y),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8),
        )
        ax.text(
            (x_start + x_end) / 2,
            node_y + 0.6,
            arrow_labels[i],
            ha="center",
            va="center",
            fontsize=7.5,
            color="#555555",
        )

    ax.text(2.5, -1.0, "MSLPO 第一阶段完整流程结果总览", ha="center", va="center", fontsize=14, fontweight="bold")
    _save_fig(fig, "fig00_stage1_pipeline_overview")


# ---------------------------------------------------------------------------
# Part 6: Master Report
# ---------------------------------------------------------------------------

FIGURE_SECTION_MAP = [
    ("fig00", "第一阶段整体流程结果总览图", "§X.1 概述 / 章节开头"),
    ("fig01", "Top-5 参数组对比", "§X.2 参数优化结果分析"),
    ("fig02", "Top-5 性能指标对比", "§X.2 参数优化结果分析"),
    ("fig03", "动作空间规模对比", "§X.3 动态间隔离散化效果分析"),
    ("fig04", "达到目标奖励回合数对比", "§X.3 动态间隔离散化效果分析"),
    ("fig05", "达到目标速度回合数对比", "§X.3 动态间隔离散化效果分析"),
    ("fig06", "最优性能指标对比", "§X.3 动态间隔离散化效果分析"),
    ("fig07", "参数 HL 敏感性扫描曲线", "§X.3 动态间隔离散化效果分析"),
    ("fig08", "参数 Ls 敏感性扫描曲线", "§X.3 动态间隔离散化效果分析"),
    ("fig09", "参数 Lswb 敏感性扫描曲线", "§X.3 动态间隔离散化效果分析"),
    ("fig10", "参数 Lforward 敏感性扫描曲线", "§X.3 动态间隔离散化效果分析"),
    ("fig11", "姿态库规模对比", "§X.4 姿态库生成与扩充效果分析"),
    ("fig12", "姿态来源占比", "§X.4 姿态库生成与扩充效果分析"),
    ("fig13", "Phase 覆盖柱状图", "§X.4 姿态库生成与扩充效果分析"),
    ("fig14", "人类步态筛选漏斗图", "§X.4 姿态库生成与扩充效果分析"),
    ("fig15", "关键关节饱和统计", "§X.4 姿态库生成与扩充效果分析"),
]


def _build_markdown_report(
    params: list[dict],
    dv_report: dict,
    dd_summary: dict,
    orig_summary: dict,
    exp_summary: dict,
    scan_data: dict,
) -> str:
    lines = []
    lines.append("# MSLPO 第一阶段实验结果总报告\n")
    lines.append(f"自动生成，输出目录: `{ANALYSIS_DIR.relative_to(PROJECT_ROOT)}`\n")

    # --- Section 1 ---
    lines.append("## 1. 参数优化结果摘要\n")
    if params:
        lines.append("### 1.1 Top 参数组合\n")
        lines.append(
            "| 排名 | 来源 | HL | Ls | Lswb | Lforward | total_reward | avg_forward_velocity | avg_lateral_offset |"
        )
        lines.append(
            "|------|------|-----|-----|------|----------|-------------|---------------------|-------------------|"
        )
        for p in params[:10]:
            lines.append(
                f"| {p['overall_rank']} | {p.get('source','')} | {p.get('HL','')} | {p.get('Ls','')} "
                f"| {p.get('Lswb','')} | {p.get('Lforward','')} | {p.get('total_reward',0):.4f} "
                f"| {p.get('avg_forward_velocity',0):.4f} | {p.get('avg_lateral_offset',0):.4f} |"
            )
        lines.append("")
        best = params[0]
        lines.append(
            f"- **最优参数组**: HL={best.get('HL')}, Ls={best.get('Ls')}, Lswb={best.get('Lswb')}, Lforward={best.get('Lforward')}"
        )
        lines.append(f"- **最优累计奖励**: {best.get('total_reward',0):.4f}")
        lines.append(f"- **最优前向速度**: {best.get('avg_forward_velocity',0):.4f} m/s")
        lines.append("")

    # --- Section 2 ---
    lines.append("## 2. 动态间隔离散化结果摘要\n")
    if dv_report:
        lines.append("### 2.1 动作空间对比\n")
        lines.append(f"- 均匀离散动作空间: **{dv_report.get('uniform_action_space_size',0):,}**")
        lines.append(
            f"- 动态离散动作空间（紧缩后）: **{dv_report.get('dynamic_action_space_size_after_tightening',0):,}**"
        )
        lines.append(f"- 缩减比例: **{dv_report.get('action_space_reduction_percent',0):.2f}%**")
        lines.append("")
        lines.append("### 2.2 收敛效率对比\n")
        lines.append(
            f"- 达到目标奖励回合数: 均匀 {dv_report.get('uniform_episodes_to_target_reward','N/A')} vs 动态 {dv_report.get('dynamic_episodes_to_target_reward','N/A')}"
        )
        lines.append(
            f"- 达到目标速度回合数: 均匀 {dv_report.get('uniform_episodes_to_target_velocity','N/A')} vs 动态 {dv_report.get('dynamic_episodes_to_target_velocity','N/A')}"
        )
        lines.append("")
        lines.append("### 2.3 最优性能对比\n")
        lines.append("| 指标 | 均匀离散 | 动态离散 |")
        lines.append("|------|---------|---------|")
        lines.append(
            f"| 最优累计奖励 | {dv_report.get('uniform_best_total_reward',0):.4f} | {dv_report.get('dynamic_best_total_reward',0):.4f} |"
        )
        lines.append(
            f"| 最优前向速度 | {dv_report.get('uniform_best_forward_velocity',0):.4f} | {dv_report.get('dynamic_best_forward_velocity',0):.4f} |"
        )
        lines.append(
            f"| 最优横向偏移 | {dv_report.get('uniform_best_lateral_offset',0):.4f} | {dv_report.get('dynamic_best_lateral_offset',0):.4f} |"
        )
        lines.append(
            f"| 最长存活时间 | {dv_report.get('uniform_best_time_to_fall','N/A') or 'N/A'} | {dv_report.get('dynamic_best_time_to_fall','N/A') or 'N/A'} |"
        )
        lines.append("")
        lines.append("### 2.4 动态离散化配置\n")
        if dd_summary:
            for pname, pconf in dd_summary.items():
                lines.append(
                    f"- **{pname}**: 离散值数量={pconf.get('num_discrete_values')}, 重点区间={pconf.get('focus_zones')}"
                )
        lines.append("")

    # --- Section 3 ---
    lines.append("## 3. 原始姿态库生成结果摘要\n")
    if orig_summary:
        lines.append(f"- **参数组数量**: {orig_summary.get('num_parameter_groups')}")
        lines.append(f"- **核心状态数**: {orig_summary.get('num_core_states')}")
        lines.append(f"- **每状态相位数**: {orig_summary.get('phases_per_state')}")
        lines.append(f"- **原始姿态总数**: **{orig_summary.get('total_poses')}** (5×2×16)")
        lines.append("")
        lines.append("### 各参数组概览\n")
        lines.append("| 组别 | Rank | HL | Ls | Lswb | Lforward | Reward | 姿态数 |")
        lines.append("|------|------|-----|-----|------|----------|--------|--------|")
        for g in orig_summary.get("parameter_groups", []):
            lines.append(
                f"| {g['param_group_idx']} | {g.get('rank','')} | {g.get('HL','')} | {g.get('Ls','')} "
                f"| {g.get('Lswb','')} | {g.get('Lforward','')} | {g.get('total_reward',0):.4f} | {g.get('num_poses','')} |"
            )
        lines.append("")

    # --- Section 4 ---
    lines.append("## 4. 人类步态扩充姿态库结果摘要\n")
    if exp_summary:
        lines.append(f"- **Stage1 姿态数**: {exp_summary.get('stage1_count')}")
        lines.append(f"- **人类候选输入**: {exp_summary.get('human_candidate_input')}")
        lines.append(f"- **最终人类姿态数**: {exp_summary.get('human_final_count')}")
        lines.append(f"- **最终姿态总数**: **{exp_summary.get('final_total')}**")
        lines.append(
            f"- **来源占比**: Stage1 {exp_summary.get('source_ratio',{}).get('stage1_percent',0)}%, Human {exp_summary.get('source_ratio',{}).get('human_cmu_percent',0)}%"
        )
        lines.append(
            f"- **Soft target**: {exp_summary.get('soft_target')} (达成: {exp_summary.get('soft_target_reached')})"
        )
        lines.append(f"- **Phase 全覆盖**: {exp_summary.get('phase_full_coverage')}")
        lines.append("")
        funnel = exp_summary.get("filtering_funnel", {})
        lines.append("### 筛选漏斗\n")
        lines.append("| 阶段 | 数量 |")
        lines.append("|------|------|")
        lines.append(f"| CMU 候选输入 | {funnel.get('human_input',0)} |")
        lines.append(f"| 有效性过滤后 | {funnel.get('validity_passed',0)} |")
        lines.append(f"| 形态学过滤后 | {funnel.get('final_kept_after_filter',0)} |")
        lines.append(f"| Stage1 去重后 | {funnel.get('after_stage1_dedup',0)} |")
        lines.append(f"| Human 去重后 | {funnel.get('after_human_dedup',0)} |")
        lines.append(f"| 预算控制后 | {funnel.get('budget_final',0)} |")
        lines.append("")
        score_stats = exp_summary.get("score_stats", {})
        if score_stats:
            lines.append("### 质量评分分布\n")
            lines.append("| 评分指标 | 均值 | 最小值 | 最大值 |")
            lines.append("|---------|------|--------|--------|")
            for key in ("quality_score", "novelty_score", "joint_margin_penalty", "final_score"):
                s = score_stats.get(key, {})
                lines.append(f"| {key} | {s.get('mean',0):.4f} | {s.get('min',0):.4f} | {s.get('max',0):.4f} |")
            lines.append("")
        dist_stats = exp_summary.get("nearest_stage1_distance_stats", {})
        if dist_stats:
            lines.append("### Human-Stage1 距离统计\n")
            lines.append(f"- 均值: {dist_stats.get('mean',0):.4f}")
            lines.append(f"- 最小值: {dist_stats.get('min',0):.4f}")
            lines.append(f"- 最大值: {dist_stats.get('max',0):.4f}")
            lines.append(f"- 标准差: {dist_stats.get('std',0):.4f}")
            lines.append(f"- 重叠评估: **{exp_summary.get('overlap_assessment','')}**")
            lines.append("")

    # --- Section 5 ---
    lines.append("## 5. 最适合写进论文正文的关键数字清单\n")
    lines.append("以下数据可直接引用至论文正文：\n")
    if params:
        b = params[0]
        lines.append("### 参数优化")
        lines.append(
            f"- 最优参数组: HL={b.get('HL')}, Ls={b.get('Ls')}, Lswb={b.get('Lswb')}, Lforward={b.get('Lforward')}"
        )
        lines.append(f"- 最优累计奖励: {b.get('total_reward',0):.4f}")
        lines.append(f"- 最优前向速度: {b.get('avg_forward_velocity',0):.4f} m/s")
    if dv_report:
        lines.append("")
        lines.append("### 动态离散化")
        lines.append(
            f"- 动作空间: {dv_report.get('uniform_action_space_size',0):,} → {dv_report.get('dynamic_action_space_size_after_tightening',0):,} (缩减 {dv_report.get('action_space_reduction_percent',0):.2f}%)"
        )
        lines.append(
            f"- 达到目标奖励: 均匀 {dv_report.get('uniform_episodes_to_target_reward','N/A')} 回合 vs 动态 {dv_report.get('dynamic_episodes_to_target_reward','N/A')} 回合"
        )
        lines.append(
            f"- 最优奖励: 均匀 {dv_report.get('uniform_best_total_reward',0):.4f} vs 动态 {dv_report.get('dynamic_best_total_reward',0):.4f}"
        )
    if orig_summary:
        lines.append("")
        lines.append("### 原始姿态库")
        lines.append(f"- 原始姿态库: {orig_summary.get('total_poses')} 姿态 (5组×2状态×16相位)")
    if exp_summary:
        lines.append("")
        lines.append("### 扩充姿态库")
        lines.append(f"- 扩充后姿态总数: {exp_summary.get('final_total')} (160+{exp_summary.get('human_final_count')})")
        lines.append(f"- Soft target 达成: {exp_summary.get('soft_target_reached')}")
        lines.append(f"- Phase 全覆盖: {exp_summary.get('phase_full_coverage')}")
        funnel = exp_summary.get("filtering_funnel", {})
        lines.append(
            f"- 筛选漏斗: {funnel.get('human_input',0)} → {funnel.get('budget_final',0)} (保留率 {100*funnel.get('budget_final',0)/max(funnel.get('human_input',0),1):.1f}%)"
        )
        lines.append(f"- 重叠评估: {exp_summary.get('overlap_assessment','')}")
    lines.append("")

    # --- Section 6 ---
    lines.append("## 6. 论文插图推荐对应关系\n")
    lines.append("| 图号 | 标题 | 推荐论文章节 |")
    lines.append("|------|------|-------------|")
    for fig_id, title, section in FIGURE_SECTION_MAP:
        lines.append(f"| {fig_id} | {title} | {section} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 1 analysis report")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    args = parser.parse_args()

    global OUTPUT_BASE, ANALYSIS_DIR, FIGURES_DIR
    root = Path(args.project_root)
    OUTPUT_BASE = root / "outputs"
    ANALYSIS_DIR = OUTPUT_BASE / "analysis_stage1"
    FIGURES_DIR = ANALYSIS_DIR / "figures"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    font = _setup_chinese_font()
    print(f"[INFO] Chinese font: {font}")

    print("\n" + "=" * 60)
    print("  MSLPO Stage 1 Analysis Report Generator")
    print("=" * 60)

    # --- Part 1: Parameter Optimization ---
    print("\n[Part 1] 参数优化结果 ...")
    params = _load_top_parameters()
    if params:
        _export_top_parameter_sets(params)
        _export_parameter_distribution_stats(params)
        _plot_top5_parameter_comparison(params)
        _plot_top5_performance_comparison(params)
    else:
        print("  (skipped: no top parameter data)")

    # --- Part 2: Dynamic Discretization ---
    print("\n[Part 2] 动态间隔离散化分析 ...")
    dv_report = _export_dynamic_vs_uniform()
    dd_summary = _export_dynamic_discretization_summary()
    if dv_report:
        _plot_action_space_comparison(dv_report)
        _plot_episodes_to_target_reward(dv_report)
        _plot_episodes_to_target_velocity(dv_report)
        _plot_best_performance_comparison(dv_report)
    else:
        print("  (skipped: no dynamic vs uniform report)")

    # --- Part 3: Sensitivity Scan ---
    print("\n[Part 3] 敏感性扫描曲线 ...")
    scan_data = _load_sensitivity_scan()
    param_scan_fig_nums = {"HL": 7, "Ls": 8, "Lswb": 9, "Lforward": 10}
    for pname, fig_num in param_scan_fig_nums.items():
        if pname in scan_data:
            zones = _get_focus_zones(pname, dd_summary)
            _plot_sensitivity_curve(pname, scan_data[pname], zones, fig_num)
        else:
            print(f"  (skipped {pname}: no scan data)")

    # --- Part 4: Pose Library ---
    print("\n[Part 4] 姿态库分析 ...")
    orig_summary = _export_original_pose_library_summary()
    exp_summary = _export_expanded_pose_library_summary()
    if orig_summary:
        pass
    if exp_summary:
        _plot_library_size_comparison(exp_summary)
        _plot_source_proportion(exp_summary)
        _plot_phase_coverage(orig_summary, exp_summary)
        _plot_human_filtering_funnel(exp_summary)
        _plot_key_joint_saturation(exp_summary)
    else:
        print("  (skipped: no expanded pose library data)")

    # --- Part 5: Pipeline Overview ---
    print("\n[Part 5] 流程总览图 ...")
    if params and dv_report and exp_summary:
        _plot_pipeline_overview(params, dv_report, exp_summary)
    else:
        print("  (skipped: missing data for pipeline overview)")

    # --- Part 6: Markdown Report ---
    print("\n[Part 6] 生成总报告 ...")
    md = _build_markdown_report(params, dv_report, dd_summary, orig_summary, exp_summary, scan_data)
    report_path = ANALYSIS_DIR / "stage1_results_report.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  -> {report_path}")

    print("\n" + "=" * 60)
    print("  Done! All outputs in:", ANALYSIS_DIR)
    print("  Figures in:", FIGURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
