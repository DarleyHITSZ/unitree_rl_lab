"""Pose library fusion: expand stage1 poses with human mocap candidates.

Fuses the stage1 SIMBICON pose library (160 poses, 5 groups x 2 states x 16 phases)
with filtered and deduplicated human CMU Mocap candidate poses (up to 240 input,
targeting 48-64 accepted).  Produces an expanded pose library ready for Stage 2 PPO.

This module works purely offline -- no Isaac Sim or GPU required.

Usage::

    from unitree_rl_lab.controllers.simbicon.pose_library_fusion import (
        build_expanded_pose_library,
        FusionConfig,
    )

    poses, meta, report = build_expanded_pose_library(
        stage1_dir="outputs/pose_library",
        human_dir="outputs/pose_library",
        output_dir="outputs/pose_library",
    )
"""

from __future__ import annotations

import json
import numpy as np
import os
from dataclasses import dataclass
from typing import Any

from .g1_joint_map import DEFAULT_JOINT_POSITIONS, G1_JOINT_NAMES, NUM_TOTAL_JOINTS

# ---------------------------------------------------------------------------
# Joint limits (conservative, matching human_to_g1_mapping_config.py)
# ---------------------------------------------------------------------------

G1_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-1.57, 1.57),
    "left_hip_roll_joint": (-0.6, 0.6),
    "left_hip_yaw_joint": (-0.52, 0.52),
    "left_knee_joint": (0.0, 2.09),
    "left_ankle_pitch_joint": (-1.0, 1.0),
    "left_ankle_roll_joint": (-0.52, 0.52),
    "right_hip_pitch_joint": (-1.57, 1.57),
    "right_hip_roll_joint": (-0.6, 0.6),
    "right_hip_yaw_joint": (-0.52, 0.52),
    "right_knee_joint": (0.0, 2.09),
    "right_ankle_pitch_joint": (-1.0, 1.0),
    "right_ankle_roll_joint": (-0.52, 0.52),
    "waist_yaw_joint": (-1.57, 1.57),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.14, 3.14),
    "left_shoulder_roll_joint": (-3.14, 3.14),
    "left_shoulder_yaw_joint": (-3.14, 3.14),
    "left_elbow_joint": (-3.14, 3.14),
    "left_wrist_roll_joint": (-3.14, 3.14),
    "left_wrist_pitch_joint": (-3.14, 3.14),
    "left_wrist_yaw_joint": (-3.14, 3.14),
    "right_shoulder_pitch_joint": (-3.14, 3.14),
    "right_shoulder_roll_joint": (-3.14, 3.14),
    "right_shoulder_yaw_joint": (-3.14, 3.14),
    "right_elbow_joint": (-3.14, 3.14),
    "right_wrist_roll_joint": (-3.14, 3.14),
    "right_wrist_pitch_joint": (-3.14, 3.14),
    "right_wrist_yaw_joint": (-3.14, 3.14),
}

_JOINT_LIMITS_LO = np.array([G1_JOINT_LIMITS[n][0] for n in G1_JOINT_NAMES], dtype=np.float64)
_JOINT_LIMITS_HI = np.array([G1_JOINT_LIMITS[n][1] for n in G1_JOINT_NAMES], dtype=np.float64)

# ---------------------------------------------------------------------------
# Key joints for deduplication subspace
# ---------------------------------------------------------------------------

KEY_JOINT_INDICES: list[int] = [0, 1, 3, 4, 6, 7, 9, 10]

DEFAULT_KEY_JOINT_WEIGHTS: np.ndarray = np.array(
    [1.5, 1.0, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5],
    dtype=np.float64,
)

MAPPED_JOINT_INDICES: list[int] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

KEY_JOINT_NAMES_SHORT: list[str] = [
    "L_hip_pitch",
    "L_hip_roll",
    "L_knee",
    "L_ankle_pitch",
    "R_hip_pitch",
    "R_hip_roll",
    "R_knee",
    "R_ankle_pitch",
]

_LEFT_SAGITTAL_INDICES = [0, 3, 4]
_RIGHT_SAGITTAL_INDICES = [6, 9, 10]


def _build_default_full_pos() -> np.ndarray:
    defaults = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float64)
    for name, val in DEFAULT_JOINT_POSITIONS.items():
        if name in G1_JOINT_NAMES:
            idx = G1_JOINT_NAMES.index(name)
            defaults[idx] = val
    return defaults


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    stage1_vs_human_dedup_threshold: float = 0.40
    human_vs_human_dedup_threshold: float = 0.25
    cross_phase_dedup_multiplier: float = 0.0
    max_human_poses_per_phase: int = 3
    boosted_max_human_poses_per_phase: int = 4
    boost_phase_candidate_threshold: int = 4
    soft_target_human_total: int = 56
    hard_max_human_total: int = 80
    clip_ratio_max: float = 0.3
    near_default_threshold: float = 0.08
    knee_min: float = 0.05
    knee_max: float = 1.8
    use_half_cycle_asymmetry: bool = True
    half_cycle_asymmetry_threshold: float = 1.0
    asymmetry_threshold: float = 1.5
    smoothness_max_jump: float = 1.0
    key_joint_margin_min: float = 0.01
    w_quality: float = 0.5
    w_novelty: float = 0.3
    w_limit_penalty: float = 0.2
    novelty_normalization: str | float = "auto"
    novelty_auto_lower_percentile: float = 10.0
    novelty_auto_upper_percentile: float = 90.0
    joint_margin_penalty_scale: float = 0.3


# ---------------------------------------------------------------------------
# PoseLibraryLoader
# ---------------------------------------------------------------------------


class PoseLibraryLoader:
    @staticmethod
    def load_stage1(
        directory: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        npy_path = os.path.join(directory, "pose_library.npy")
        meta_path = os.path.join(directory, "pose_library_meta.json")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Stage1 pose library not found: {npy_path}")
        poses = np.load(npy_path).astype(np.float64)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return poses, meta

    @staticmethod
    def load_human_candidates(
        directory: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        npy_path = os.path.join(directory, "human_g1_candidate_poses.npy")
        meta_path = os.path.join(directory, "human_g1_candidate_poses_meta.json")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Human candidate poses not found: {npy_path}")
        poses = np.load(npy_path).astype(np.float64)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return poses, meta


# ---------------------------------------------------------------------------
# HumanPoseFilter
# ---------------------------------------------------------------------------


def _compute_half_cycle_asymmetry(
    poses: np.ndarray,
    meta: list[dict[str, Any]],
    raw_asymmetry: np.ndarray,
) -> np.ndarray:
    result = raw_asymmetry.copy() * 0.3

    cycle_groups: dict[tuple[str, int], dict[int, int]] = {}
    for i, m in enumerate(meta):
        key = (m.get("source_npz", ""), m.get("cycle_id", -1))
        pi = m.get("phase_index", 0)
        cycle_groups.setdefault(key, {})[pi] = i

    for members in cycle_groups.values():
        for p, idx in members.items():
            mirror_p = (p + 8) % 16
            if mirror_p in members:
                mirror_idx = members[mirror_p]
                left_key = poses[idx, _LEFT_SAGITTAL_INDICES]
                right_key_mirror = poses[mirror_idx, _RIGHT_SAGITTAL_INDICES]
                result[idx] = float(np.linalg.norm(left_key - right_key_mirror))
            else:
                result[idx] = raw_asymmetry[idx] * 0.3

    return result


class HumanPoseFilter:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def filter_all(
        self,
        poses: np.ndarray,
        meta: list[dict[str, Any]],
        default_pose: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[dict[str, Any]],
        dict[str, np.ndarray],
        dict[str, Any],
    ]:
        features = self._compute_features(poses, meta, default_pose)
        v_mask, v_reasons = self._filter_validity(poses, features)
        m_mask, m_reasons = self._filter_morphology(features, v_mask)
        combined = v_mask & m_mask

        filtered_poses = poses[combined]
        filtered_meta = [meta[i] for i in range(len(meta)) if combined[i]]

        filtered_features: dict[str, np.ndarray] = {}
        for k, v in features.items():
            if isinstance(v, np.ndarray) and v.shape[0] == len(poses):
                filtered_features[k] = v[combined]

        old_asym = features["asymmetry_raw"][v_mask]
        new_asym = features["asymmetry_half_cycle"][v_mask]

        report: dict[str, Any] = {
            "total_input": len(poses),
            "validity_passed": int(v_mask.sum()),
            "validity_rejected": int((~v_mask).sum()),
            "validity_reasons": v_reasons,
            "morphology_passed": int((v_mask & m_mask).sum()),
            "morphology_rejected": int((v_mask & ~m_mask).sum()),
            "morphology_reasons": m_reasons,
            "final_kept": int(combined.sum()),
            "old_asymmetry_stat": {
                "mean": round(float(old_asym.mean()), 6),
                "max": round(float(old_asym.max()), 6),
                "rejected_by_old": int((old_asym > self.config.asymmetry_threshold).sum()),
            },
            "new_asymmetry_stat": {
                "mean": round(float(new_asym.mean()), 6),
                "max": round(float(new_asym.max()), 6),
                "rejected_by_new": int((new_asym > self.config.half_cycle_asymmetry_threshold).sum()),
            },
        }

        per_joint_margins = features.get("key_joint_margins_per_joint")
        if per_joint_margins is not None:
            per_joint_sat: dict[str, dict[str, Any]] = {}
            for ki in range(len(KEY_JOINT_INDICES)):
                name = KEY_JOINT_NAMES_SHORT[ki]
                margins = per_joint_margins[:, ki]
                per_joint_sat[name] = {
                    "count_saturated": int((margins < self.config.key_joint_margin_min).sum()),
                    "mean_margin": round(float(margins.mean()), 6),
                    "min_margin": round(float(margins.min()), 6),
                    "p10_margin": round(float(np.percentile(margins, 10)), 6),
                    "p50_margin": round(float(np.percentile(margins, 50)), 6),
                }
            report["key_joint_saturated_by_joint"] = per_joint_sat

        return filtered_poses, filtered_meta, filtered_features, report

    def _compute_features(
        self,
        poses: np.ndarray,
        meta: list[dict[str, Any]],
        default_pose: np.ndarray,
    ) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        features["clip_ratio"] = np.array([m.get("clip_ratio", 0.0) for m in meta])

        key_diff = poses[:, KEY_JOINT_INDICES] - default_pose[np.newaxis, KEY_JOINT_INDICES]
        features["key_joint_diff_norm"] = np.linalg.norm(key_diff, axis=1)
        features["left_knee"] = poses[:, 3].copy()
        features["right_knee"] = poses[:, 9].copy()

        left_key = poses[:, [0, 1, 3, 4]]
        right_key = poses[:, [6, 7, 9, 10]]
        features["asymmetry_raw"] = np.linalg.norm(left_key - right_key, axis=1)

        key_poses = poses[:, KEY_JOINT_INDICES]
        key_lo = _JOINT_LIMITS_LO[KEY_JOINT_INDICES]
        key_hi = _JOINT_LIMITS_HI[KEY_JOINT_INDICES]
        dist_to_limit = np.minimum(key_poses - key_lo, key_hi - key_poses)
        features["key_joint_min_margin"] = dist_to_limit.min(axis=1)
        features["key_joint_margins_per_joint"] = dist_to_limit

        features["smoothness"] = self._compute_cycle_smoothness(poses, meta)

        features["asymmetry_half_cycle"] = _compute_half_cycle_asymmetry(poses, meta, features["asymmetry_raw"])
        return features

    def _compute_cycle_smoothness(
        self,
        poses: np.ndarray,
        meta: list[dict[str, Any]],
    ) -> np.ndarray:
        n = len(poses)
        smoothness = np.ones(n, dtype=np.float64)
        cycle_groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for i, m in enumerate(meta):
            key = (m.get("source_npz", ""), m.get("cycle_id", -1))
            cycle_groups.setdefault(key, []).append((m.get("phase_index", 0), i))

        for members in cycle_groups.values():
            if len(members) < 2:
                continue
            members.sort()
            for j in range(len(members)):
                _, idx = members[j]
                max_jump = 0.0
                if j > 0:
                    _, prev_idx = members[j - 1]
                    d = np.linalg.norm(poses[idx, MAPPED_JOINT_INDICES] - poses[prev_idx, MAPPED_JOINT_INDICES])
                    max_jump = max(max_jump, d)
                if j < len(members) - 1:
                    _, next_idx = members[j + 1]
                    d = np.linalg.norm(poses[idx, MAPPED_JOINT_INDICES] - poses[next_idx, MAPPED_JOINT_INDICES])
                    max_jump = max(max_jump, d)
                smoothness[idx] = 1.0 / (1.0 + max_jump)
        return smoothness

    def _filter_validity(
        self,
        poses: np.ndarray,
        features: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, int]]:
        mask = np.ones(len(poses), dtype=bool)
        reasons: dict[str, int] = {}

        nan_inf = np.any(np.isnan(poses), axis=1) | np.any(np.isinf(poses), axis=1)
        reasons["nan_or_inf"] = int(nan_inf.sum())
        mask &= ~nan_inf

        high_clip = features["clip_ratio"] > self.config.clip_ratio_max
        reasons["clip_ratio_exceeded"] = int(high_clip.sum())
        mask &= ~high_clip

        saturated = features["key_joint_min_margin"] < self.config.key_joint_margin_min
        reasons["key_joint_saturated"] = int(saturated.sum())
        mask &= ~saturated

        near_default = features["key_joint_diff_norm"] < self.config.near_default_threshold
        reasons["near_default_pose"] = int(near_default.sum())
        mask &= ~near_default

        reasons["total_rejected"] = int((~mask).sum())
        return mask, reasons

    def _filter_morphology(
        self,
        features: dict[str, np.ndarray],
        validity_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int]]:
        mask = np.ones(features["left_knee"].shape[0], dtype=bool)
        reasons: dict[str, int] = {}
        check = validity_mask

        lk = features["left_knee"]
        rk = features["right_knee"]
        bad_knee = check & (
            (lk < self.config.knee_min)
            | (lk > self.config.knee_max)
            | (rk < self.config.knee_min)
            | (rk > self.config.knee_max)
        )
        reasons["knee_out_of_range"] = int(bad_knee.sum())
        mask &= ~bad_knee

        if self.config.use_half_cycle_asymmetry:
            asym = features["asymmetry_half_cycle"]
            threshold = self.config.half_cycle_asymmetry_threshold
            reasons["asymmetry_mode"] = "half_cycle_mirror"  # type: ignore[assignment]
        else:
            asym = features["asymmetry_raw"]
            threshold = self.config.asymmetry_threshold
            reasons["asymmetry_mode"] = "raw_left_right"  # type: ignore[assignment]
        asymmetrical = check & (asym > threshold)
        reasons["excessive_asymmetry"] = int(asymmetrical.sum())
        mask &= ~asymmetrical

        smooth_threshold = 1.0 / (1.0 + self.config.smoothness_max_jump)
        not_smooth = check & (features["smoothness"] < smooth_threshold)
        reasons["not_smooth"] = int(not_smooth.sum())
        mask &= ~not_smooth

        reasons["total_rejected"] = int((validity_mask & ~mask).sum())
        return mask, reasons


# ---------------------------------------------------------------------------
# PoseDistanceMetric
# ---------------------------------------------------------------------------


class PoseDistanceMetric:
    def __init__(
        self,
        key_joint_indices: list[int] | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self.key_joints = key_joint_indices or KEY_JOINT_INDICES
        self.weights = (weights if weights is not None else DEFAULT_KEY_JOINT_WEIGHTS).astype(np.float64)

    def distance(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        diff = pose_a[self.key_joints] - pose_b[self.key_joints]
        return float(np.sqrt(np.sum(self.weights * diff**2)))

    def min_distance_to_set(self, pose: np.ndarray, reference_poses: np.ndarray) -> float:
        if len(reference_poses) == 0:
            return float("inf")
        diffs = reference_poses[:, self.key_joints] - pose[self.key_joints]
        weighted_sq = np.sum(self.weights * diffs**2, axis=1)
        return float(np.sqrt(np.min(weighted_sq)))

    def pairwise_distances(self, poses_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
        a = poses_a[:, self.key_joints]
        b = poses_b[:, self.key_joints]
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        weighted_sq = np.sum(self.weights * diff**2, axis=2)
        return np.sqrt(weighted_sq)


# ---------------------------------------------------------------------------
# Stage1 distance diagnostics
# ---------------------------------------------------------------------------


def diagnose_stage1_distances(
    human_poses: np.ndarray,
    human_meta: list[dict[str, Any]],
    stage1_poses: np.ndarray,
    stage1_meta: list[dict[str, Any]],
    metric: PoseDistanceMetric,
    output_dir: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    dists = metric.pairwise_distances(human_poses, stage1_poses)
    min_dists = dists.min(axis=1)
    nearest_indices = dists.argmin(axis=1)

    percentiles = {f"p{int(p)}": round(float(np.percentile(min_dists, p)), 6) for p in [5, 10, 25, 50, 75, 90, 95]}
    stats: dict[str, Any] = {
        "mean": round(float(min_dists.mean()), 6),
        "min": round(float(min_dists.min()), 6),
        "max": round(float(min_dists.max()), 6),
        "std": round(float(min_dists.std()), 6),
        "percentiles": percentiles,
    }

    n_samples = min(10, len(human_poses))
    sample_indices = np.argsort(min_dists)[:n_samples]
    samples: list[dict[str, Any]] = []
    for si in sample_indices:
        ni = int(nearest_indices[si])
        joint_comparison: dict[str, dict[str, float]] = {}
        key_joint_abs_diff: dict[str, float] = {}
        for ki, jidx in enumerate(KEY_JOINT_INDICES):
            name = KEY_JOINT_NAMES_SHORT[ki]
            abs_d = abs(float(human_poses[si, jidx] - stage1_poses[ni, jidx]))
            joint_comparison[name] = {
                "human": round(float(human_poses[si, jidx]), 4),
                "stage1": round(float(stage1_poses[ni, jidx]), 4),
                "diff": round(float(human_poses[si, jidx] - stage1_poses[ni, jidx]), 4),
            }
            key_joint_abs_diff[name] = round(abs_d, 4)
        samples.append(
            {
                "human_idx": int(si),
                "human_meta": {
                    "source_npz": human_meta[si].get("source_npz", ""),
                    "phase_index": human_meta[si].get("phase_index", -1),
                    "cycle_id": human_meta[si].get("cycle_id", -1),
                    "support_foot": human_meta[si].get("support_foot", ""),
                },
                "nearest_stage1_idx": ni,
                "stage1_meta": {
                    "param_group_idx": stage1_meta[ni].get("param_group_idx", -1),
                    "fsm_state": stage1_meta[ni].get("fsm_state", ""),
                    "phase_index": stage1_meta[ni].get("phase_index", -1),
                    "support_foot": stage1_meta[ni].get("support_foot", ""),
                },
                "distance": round(float(min_dists[si]), 6),
                "key_joint_comparison": joint_comparison,
                "key_joint_abs_diff": key_joint_abs_diff,
            }
        )

    diag_report: dict[str, Any] = {
        "stats": stats,
        "sample_comparisons": samples,
    }

    os.makedirs(output_dir, exist_ok=True)
    dist_path = os.path.join(output_dir, "nearest_stage1_distances.json")
    with open(dist_path, "w") as f:
        json.dump(diag_report, f, indent=2)
    print(f"[FUSION] Stage1 distance diagnostics saved to {dist_path}")

    print(
        f"[FUSION] Human→Stage1 min distances: mean={stats['mean']:.4f}, "
        f"p10={percentiles['p10']:.4f}, p50={percentiles['p50']:.4f}, "
        f"p90={percentiles['p90']:.4f}, max={stats['max']:.4f}"
    )

    return min_dists, diag_report


# ---------------------------------------------------------------------------
# score_human_pose
# ---------------------------------------------------------------------------


def score_human_pose(
    pose: np.ndarray,
    stage1_poses: np.ndarray,
    clip_ratio: float,
    smoothness: float,
    left_knee: float,
    right_knee: float,
    asymmetry: float,
    key_joint_min_margin: float,
    metric: PoseDistanceMetric,
    config: FusionConfig,
    novelty_lower: float = 0.0,
    novelty_upper: float = 1.0,
) -> dict[str, float]:
    knee_score = 1.0
    for kv in (left_knee, right_knee):
        if kv < config.knee_min:
            knee_score *= max(0.0, 1.0 - (config.knee_min - kv) / max(config.knee_min, 1e-8))
        elif kv > config.knee_max:
            knee_score *= max(
                0.0,
                1.0 - (kv - config.knee_max) / max(2.09 - config.knee_max, 1e-8),
            )
    knee_score = float(np.clip(knee_score, 0.0, 1.0))

    symmetry_score = float(np.clip(1.0 - asymmetry / config.half_cycle_asymmetry_threshold, 0.0, 1.0))
    smooth_score = float(np.clip(smoothness, 0.0, 1.0))

    quality = 0.3 * knee_score + 0.3 * symmetry_score + 0.4 * smooth_score

    min_dist = metric.min_distance_to_set(pose, stage1_poses)
    novelty_range = novelty_upper - novelty_lower
    if novelty_range < 1e-6:
        novelty_range = 1.0
    novelty = float(np.clip((min_dist - novelty_lower) / novelty_range, 0.0, 1.0))

    margin_penalty = float(np.clip(1.0 - key_joint_min_margin / config.joint_margin_penalty_scale, 0.0, 1.0))
    limit_penalty = (float(clip_ratio) + margin_penalty) / 2.0

    final = config.w_quality * quality + config.w_novelty * novelty - config.w_limit_penalty * limit_penalty

    return {
        "quality_score": round(float(quality), 6),
        "novelty_score": round(novelty, 6),
        "clip_penalty": round(float(clip_ratio), 6),
        "joint_margin_penalty": round(margin_penalty, 6),
        "limit_penalty": round(limit_penalty, 6),
        "final_score": round(float(final), 6),
    }


# ---------------------------------------------------------------------------
# PoseLibraryFusion
# ---------------------------------------------------------------------------


class PoseLibraryFusion:
    def __init__(
        self,
        config: FusionConfig | None = None,
        metric: PoseDistanceMetric | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.metric = metric or PoseDistanceMetric()

    def dedup_human_vs_stage1(
        self,
        human_poses: np.ndarray,
        human_meta: list[dict[str, Any]],
        stage1_poses: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dists = self.metric.pairwise_distances(human_poses, stage1_poses)
        min_dists = dists.min(axis=1)
        min_indices = dists.argmin(axis=1)
        keep = min_dists >= self.config.stage1_vs_human_dedup_threshold

        absorbed: list[dict[str, Any]] = []
        for i in range(len(human_poses)):
            if not keep[i]:
                absorbed.append(
                    {
                        "human_idx": i,
                        "nearest_stage1_idx": int(min_indices[i]),
                        "distance": round(float(min_dists[i]), 6),
                        "source_npz": human_meta[i].get("source_npz", ""),
                        "phase_index": human_meta[i].get("phase_index", -1),
                    }
                )

        report: dict[str, Any] = {
            "total_human": len(human_poses),
            "kept": int(keep.sum()),
            "absorbed": int((~keep).sum()),
            "absorbed_details": absorbed,
            "distance_stats": {
                "mean": round(float(min_dists.mean()), 6),
                "min": round(float(min_dists.min()), 6),
                "max": round(float(min_dists.max()), 6),
                "threshold": self.config.stage1_vs_human_dedup_threshold,
            },
        }
        return keep, report

    def dedup_human_vs_human(
        self,
        human_poses: np.ndarray,
        human_meta: list[dict[str, Any]],
        human_scores: list[float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n = len(human_poses)
        if n <= 1:
            return np.ones(n, dtype=bool), {"total": n, "kept": n, "removed": 0}

        dists = self.metric.pairwise_distances(human_poses, human_poses)
        np.fill_diagonal(dists, float("inf"))

        keep = np.ones(n, dtype=bool)
        removal_log: list[dict[str, Any]] = []
        order = np.argsort([-s for s in human_scores])

        phase_by_idx = [human_meta[i].get("phase_index", -1) for i in range(n)]
        npz_by_idx = [human_meta[i].get("source_npz", "") for i in range(n)]
        cycle_by_idx = [human_meta[i].get("cycle_id", -1) for i in range(n)]

        for idx in order:
            if not keep[idx]:
                continue
            for jdx in order:
                if jdx == idx or not keep[jdx]:
                    continue
                same_phase = phase_by_idx[idx] == phase_by_idx[jdx]
                if not same_phase and self.config.cross_phase_dedup_multiplier <= 0:
                    continue
                threshold = (
                    self.config.human_vs_human_dedup_threshold
                    if same_phase
                    else self.config.human_vs_human_dedup_threshold * self.config.cross_phase_dedup_multiplier
                )
                if dists[idx, jdx] < threshold:
                    keep[jdx] = False
                    removal_log.append(
                        {
                            "removed_idx": int(jdx),
                            "kept_idx": int(idx),
                            "distance": round(float(dists[idx, jdx]), 6),
                            "removed_score": round(human_scores[jdx], 6),
                            "kept_score": round(human_scores[idx], 6),
                            "removed_phase": phase_by_idx[jdx],
                            "kept_phase": phase_by_idx[idx],
                            "removed_source_npz": npz_by_idx[jdx],
                            "kept_source_npz": npz_by_idx[idx],
                            "removed_cycle_id": cycle_by_idx[jdx],
                            "kept_cycle_id": cycle_by_idx[idx],
                            "dedup_pass": "within_phase" if same_phase else "cross_phase",
                        }
                    )

        within_removed = sum(1 for r in removal_log if r["dedup_pass"] == "within_phase")
        cross_removed = sum(1 for r in removal_log if r["dedup_pass"] == "cross_phase")

        report: dict[str, Any] = {
            "total": n,
            "kept": int(keep.sum()),
            "removed": int((~keep).sum()),
            "within_phase_removed": within_removed,
            "cross_phase_removed": cross_removed,
            "within_phase_threshold": self.config.human_vs_human_dedup_threshold,
            "cross_phase_threshold": round(
                self.config.human_vs_human_dedup_threshold * self.config.cross_phase_dedup_multiplier,
                6,
            ),
            "removal_log": removal_log,
        }
        return keep, report

    def enforce_three_stage_budget(
        self,
        human_meta: list[dict[str, Any]],
        human_scores: list[float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n = len(human_meta)
        if n == 0:
            report: dict[str, Any] = {
                "input_count": 0,
                "final_count": 0,
                "floor_count": 0,
                "fill_count": 0,
                "soft_target_reached": False,
                "hard_max_applied": False,
                "phase_candidate_counts": {},
                "phase_kept_counts": {},
                "phases_with_no_candidates": list(range(16)),
                "phases_emptied_by_dedup": [],
                "phase_min_coverage_satisfied": False,
            }
            return np.zeros(0, dtype=bool), report

        phase_candidates: dict[int, list[int]] = {}
        for i in range(n):
            p = human_meta[i].get("phase_index", 0)
            phase_candidates.setdefault(p, []).append(i)

        for p in phase_candidates:
            phase_candidates[p].sort(key=lambda i: -human_scores[i])

        keep = np.zeros(n, dtype=bool)
        kept_phases: dict[int, int] = {}

        floor_count = 0
        for p in range(16):
            candidates = phase_candidates.get(p, [])
            if candidates:
                best = candidates[0]
                keep[best] = True
                kept_phases[p] = kept_phases.get(p, 0) + 1
                floor_count += 1

        fill_count = 0
        remaining_indices = []
        for i in range(n):
            if not keep[i]:
                remaining_indices.append(i)
        remaining_indices.sort(key=lambda i: -human_scores[i])

        phase_caps: dict[int, int] = {}
        for p in range(16):
            cand_count = len(phase_candidates.get(p, []))
            if cand_count >= self.config.boost_phase_candidate_threshold:
                phase_caps[p] = self.config.boosted_max_human_poses_per_phase
            else:
                phase_caps[p] = self.config.max_human_poses_per_phase

        for idx in remaining_indices:
            if int(keep.sum()) >= self.config.soft_target_human_total:
                break
            p = human_meta[idx].get("phase_index", 0)
            current_count = kept_phases.get(p, 0)
            if current_count < phase_caps.get(p, self.config.max_human_poses_per_phase):
                keep[idx] = True
                kept_phases[p] = current_count + 1
                fill_count += 1

        hard_max_applied = False
        if int(keep.sum()) > self.config.hard_max_human_total:
            hard_max_applied = True
            kept_indices = np.where(keep)[0]
            sorted_kept = sorted(kept_indices, key=lambda i: -human_scores[i])
            keep[:] = False
            for idx in sorted_kept[: self.config.hard_max_human_total]:
                keep[idx] = True
            kept_phases = {}
            for i in range(n):
                if keep[i]:
                    p = human_meta[i].get("phase_index", 0)
                    kept_phases[p] = kept_phases.get(p, 0) + 1

        phase_candidate_counts = {p: len(phase_candidates.get(p, [])) for p in range(16)}
        phase_kept_counts = {p: kept_phases.get(p, 0) for p in range(16)}
        phases_with_no_candidates = [p for p in range(16) if phase_candidate_counts[p] == 0]
        phases_emptied_by_dedup = [p for p in range(16) if phase_candidate_counts[p] > 0 and phase_kept_counts[p] == 0]
        phases_covered = sum(1 for p in range(16) if phase_kept_counts[p] > 0)
        phase_min_coverage_satisfied = phases_covered >= 14

        effective_cap = sum(
            min(phase_candidate_counts[p], phase_caps.get(p, self.config.max_human_poses_per_phase)) for p in range(16)
        )

        report = {
            "input_count": n,
            "final_count": int(keep.sum()),
            "floor_count": floor_count,
            "fill_count": fill_count,
            "soft_target": self.config.soft_target_human_total,
            "soft_target_reached": int(keep.sum()) >= self.config.soft_target_human_total,
            "hard_max_applied": hard_max_applied,
            "effective_human_cap_given_phase_budget": effective_cap,
            "soft_target_feasible": effective_cap >= self.config.soft_target_human_total,
            "phase_caps_used": {str(k): v for k, v in sorted(phase_caps.items())},
            "phase_candidate_counts": {str(k): v for k, v in sorted(phase_candidate_counts.items())},
            "phase_kept_counts": {str(k): v for k, v in sorted(phase_kept_counts.items())},
            "phases_with_no_candidates": phases_with_no_candidates,
            "phases_emptied_by_dedup": phases_emptied_by_dedup,
            "phase_min_coverage_satisfied": phase_min_coverage_satisfied,
            "phases_covered": phases_covered,
        }
        return keep, report

    def merge(
        self,
        stage1_poses: np.ndarray,
        stage1_meta: list[dict[str, Any]],
        human_poses: np.ndarray,
        human_meta: list[dict[str, Any]],
        human_scores: list[dict[str, float]],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        total = len(stage1_poses) + len(human_poses)
        merged_poses = np.zeros((total, NUM_TOTAL_JOINTS), dtype=np.float32)
        merged_meta: list[dict[str, Any]] = []

        idx = 0
        for i in range(len(stage1_poses)):
            merged_poses[idx] = stage1_poses[i].astype(np.float32)
            entry = dict(stage1_meta[i])
            entry["action_idx"] = idx
            merged_meta.append(entry)
            idx += 1

        for i in range(len(human_poses)):
            merged_poses[idx] = human_poses[i].astype(np.float32)
            entry = dict(human_meta[i])
            entry["action_idx"] = idx
            if "pose_id" in entry:
                del entry["pose_id"]
            if i < len(human_scores):
                entry.update(human_scores[i])
            merged_meta.append(entry)
            idx += 1

        return merged_poses, merged_meta

    def build(
        self,
        stage1_dir: str,
        human_dir: str,
        output_dir: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
        default_pose = _build_default_full_pos()

        s1_poses, s1_meta = PoseLibraryLoader.load_stage1(stage1_dir)

        try:
            h_poses, h_meta = PoseLibraryLoader.load_human_candidates(human_dir)
        except FileNotFoundError:
            print("[FUSION] No human candidates found, exporting stage1 only")
            merged_poses = s1_poses.astype(np.float32)
            merged_meta = [dict(m, action_idx=i) for i, m in enumerate(s1_meta)]
            self._export(merged_poses, merged_meta, output_dir)
            report = self._make_report(
                len(s1_poses),
                0,
                0,
                {},
                {},
                {},
                {},
                [],
                {},
                {},
            )
            return merged_poses, merged_meta, report

        print(f"[FUSION] Loaded stage1: {s1_poses.shape}, human: {h_poses.shape}")

        pose_filter = HumanPoseFilter(self.config)
        h_poses_f, h_meta_f, h_feat_f, filter_report = pose_filter.filter_all(h_poses, h_meta, default_pose)
        print(f"[FUSION] After filtering: {len(h_poses_f)} / {len(h_poses)} human poses kept")

        if len(h_poses_f) == 0:
            merged_poses, merged_meta = self.merge(s1_poses, s1_meta, np.zeros((0, NUM_TOTAL_JOINTS)), [], [])
            self._export(merged_poses, merged_meta, output_dir)
            report = self._make_report(
                len(s1_poses),
                len(h_poses),
                0,
                filter_report,
                {},
                {},
                {},
                [],
                {},
                {},
            )
            return merged_poses, merged_meta, report

        min_dists, diag_info = diagnose_stage1_distances(
            h_poses_f,
            h_meta_f,
            s1_poses,
            s1_meta,
            self.metric,
            output_dir,
        )

        if self.config.novelty_normalization == "auto":
            novelty_lower = float(np.percentile(min_dists, self.config.novelty_auto_lower_percentile))
            novelty_upper = float(np.percentile(min_dists, self.config.novelty_auto_upper_percentile))
            if novelty_upper - novelty_lower < 1e-6:
                novelty_upper = novelty_lower + 1.0
            novelty_norm_info: dict[str, Any] = {
                "mode": "auto",
                "method": (
                    f"percentile_mapping_p{int(self.config.novelty_auto_lower_percentile)}_p{int(self.config.novelty_auto_upper_percentile)}"
                ),
                "lower": round(novelty_lower, 6),
                "upper": round(novelty_upper, 6),
            }
        else:
            novelty_lower = 0.0
            novelty_upper = float(self.config.novelty_normalization)
            novelty_norm_info = {
                "mode": "fixed",
                "method": "fixed_range",
                "lower": 0.0,
                "upper": novelty_upper,
            }
        print(
            f"[FUSION] Novelty norm: [{novelty_lower:.4f}, {novelty_upper:.4f}] " f"(mode={novelty_norm_info['mode']})"
        )

        scores: list[dict[str, float]] = []
        for i in range(len(h_poses_f)):
            s = score_human_pose(
                pose=h_poses_f[i],
                stage1_poses=s1_poses,
                clip_ratio=float(h_feat_f["clip_ratio"][i]),
                smoothness=float(h_feat_f["smoothness"][i]),
                left_knee=float(h_feat_f["left_knee"][i]),
                right_knee=float(h_feat_f["right_knee"][i]),
                asymmetry=float(h_feat_f["asymmetry_half_cycle"][i]),
                key_joint_min_margin=float(h_feat_f["key_joint_min_margin"][i]),
                metric=self.metric,
                config=self.config,
                novelty_lower=novelty_lower,
                novelty_upper=novelty_upper,
            )
            scores.append(s)
        final_scores = [s["final_score"] for s in scores]

        keep_s1, s1_dedup_report = self.dedup_human_vs_stage1(h_poses_f, h_meta_f, s1_poses)
        print(f"[FUSION] Stage1 dedup: {keep_s1.sum()} / {len(h_poses_f)} kept")

        surv = np.where(keep_s1)[0]
        keep_hh, hh_dedup_report = self.dedup_human_vs_human(
            h_poses_f[surv],
            [h_meta_f[i] for i in surv],
            [final_scores[i] for i in surv],
        )
        surv = surv[keep_hh]
        print(f"[FUSION] Human dedup: {len(surv)} kept")

        budget_meta = [h_meta_f[i] for i in surv]
        budget_scores = [final_scores[i] for i in surv]
        keep_budget, budget_report = self.enforce_three_stage_budget(budget_meta, budget_scores)
        surv = surv[keep_budget]
        print(
            f"[FUSION] Budget: {len(surv)} kept "
            f"(floor={budget_report['floor_count']}, "
            f"fill={budget_report['fill_count']}, "
            f"soft_target_reached={budget_report['soft_target_reached']})"
        )

        h_poses_final = h_poses_f[surv]
        h_meta_final = [h_meta_f[i] for i in surv]
        h_scores_final = [scores[i] for i in surv]

        merged_poses, merged_meta = self.merge(s1_poses, s1_meta, h_poses_final, h_meta_final, h_scores_final)
        self._export(merged_poses, merged_meta, output_dir)

        report = self._make_report(
            s1_count=len(s1_poses),
            h_input=len(h_poses),
            h_final=len(surv),
            filter_report=filter_report,
            s1_dedup=s1_dedup_report,
            hh_dedup=hh_dedup_report,
            budget_report=budget_report,
            h_scores=h_scores_final,
            novelty_norm_info=novelty_norm_info,
            diag_info=diag_info,
        )
        return merged_poses, merged_meta, report

    def _assess_stage1_overlap(self, diag_info: dict[str, Any]) -> str:
        stats = diag_info.get("stats", {})
        percentiles = stats.get("percentiles", {})
        p10 = float(percentiles.get("p10", 1.0))
        mean = float(stats.get("mean", 1.0))
        threshold = self.config.stage1_vs_human_dedup_threshold
        if p10 > threshold * 2 and mean > threshold * 2:
            return "well_separated"
        if p10 > threshold:
            return "partially_overlapping"
        return "possible_mapping_misalignment"

    def _export(
        self,
        poses: np.ndarray,
        meta: list[dict[str, Any]],
        output_dir: str,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, "pose_library_expanded.npy")
        np.save(npy_path, poses)
        meta_path = os.path.join(output_dir, "pose_library_expanded_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[FUSION] Exported: {npy_path} (shape={poses.shape})")
        print(f"[FUSION] Metadata: {meta_path} ({len(meta)} entries)")

    def _make_report(
        self,
        s1_count: int,
        h_input: int,
        h_final: int,
        filter_report: dict[str, Any],
        s1_dedup: dict[str, Any],
        hh_dedup: dict[str, Any],
        budget_report: dict[str, Any],
        h_scores: list[dict[str, float]],
        novelty_norm_info: dict[str, Any],
        diag_info: dict[str, Any],
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "stage1_count": s1_count,
            "human_input": h_input,
            "human_final": h_final,
            "final_total": s1_count + h_final,
            "soft_target_human_total": self.config.soft_target_human_total,
            "soft_target_reached": h_final >= self.config.soft_target_human_total,
            "filter_report": filter_report,
            "stage1_dedup_report": s1_dedup,
            "human_dedup_report": hh_dedup,
            "budget_report": budget_report,
            "novelty_normalization_info": novelty_norm_info,
            "nearest_stage1_distance_stats": diag_info.get("stats", {}),
            "nearest_stage1_distance_percentiles": diag_info.get("stats", {}).get("percentiles", {}),
            "nearest_stage1_examples": diag_info.get("sample_comparisons", []),
            "stage1_human_overlap_assessment": self._assess_stage1_overlap(diag_info),
            "metric_config": {
                "distance_type": "weighted L2 over key joints",
                "formula": "sqrt(sum(w_i * (a_i - b_i)^2))",
                "key_joints": KEY_JOINT_NAMES_SHORT,
                "key_joint_indices": KEY_JOINT_INDICES,
                "weights": self.metric.weights.tolist(),
                "stage1_vs_human_threshold": self.config.stage1_vs_human_dedup_threshold,
                "human_vs_human_within_phase_threshold": self.config.human_vs_human_dedup_threshold,
                "human_vs_human_cross_phase_threshold": round(
                    self.config.human_vs_human_dedup_threshold * self.config.cross_phase_dedup_multiplier,
                    6,
                ),
            },
        }
        if h_scores:
            report["score_stats"] = {
                key: {
                    "mean": round(float(np.mean([s[key] for s in h_scores])), 6),
                    "min": round(float(np.min([s[key] for s in h_scores])), 6),
                    "max": round(float(np.max([s[key] for s in h_scores])), 6),
                }
                for key in (
                    "quality_score",
                    "novelty_score",
                    "joint_margin_penalty",
                    "limit_penalty",
                    "final_score",
                )
            }
        return report


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_expanded_pose_library(
    stage1_dir: str = "outputs/pose_library",
    human_dir: str = "outputs/pose_library",
    output_dir: str = "outputs/pose_library",
    config: FusionConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    fusion = PoseLibraryFusion(config)
    return fusion.build(stage1_dir, human_dir, output_dir)


def save_fusion_report(report: dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pose_library_expanded_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path
