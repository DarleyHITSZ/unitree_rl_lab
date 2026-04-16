"""Map processed CMU Mocap gait data to G1-29dof candidate poses.

This module implements the core pipeline for converting human walking gait
data from the CMU Mocap Database (already converted to .npz) into a
candidate pose dataset that aligns with the stage1 pose library format.

Pipeline overview::

    1. ProcessedCMUGaitLoader        -- load .npz files
    2. CMULowerBodySelector          -- extract 14 lower-body channels
    3. analyze_cmu_axis_importance()  -- verify axis mapping
    4. GaitCycleSegmenter            -- split into gait cycles
    5. resample_cycle_to_16_phases() -- resample to 16 phases
    6. HumanToG1PoseMapper           -- map to (29,) G1 pose vectors
    7. build_human_g1_candidate_dataset() -- orchestrate everything

Usage::

    from human_gait_to_g1 import build_human_g1_candidate_dataset

    poses, meta = build_human_g1_candidate_dataset(
        input_dir="data/human_gait/processed/cmu_subject_07",
        output_dir="outputs/pose_library",
    )
"""

from __future__ import annotations

import json
import logging
import numpy as np
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_to_g1_mapping_config import (
    CMU_LOWER_BODY_CHANNELS,
    G1_JOINT_LIMITS,
    G1_JOINT_NAMES,
    HUMAN_TO_G1_JOINT_MAPPING,
    NUM_PHASES,
    NUM_TOTAL_JOINTS,
    build_default_g1_pose,
    resolve_knee_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ProcessedCMUGaitLoader
# ---------------------------------------------------------------------------


class ProcessedCMUGaitLoader:
    """Load processed CMU .npz motion files.

    Each .npz is expected to contain:
        angles (T, 56), joint_names (56,), num_frames, fps,
        root_translation, root_rotation, source_amc, motion_name, ...
    """

    def load_directory(self, input_dir: str) -> list[dict[str, Any]]:
        """Load all .npz files from a directory.

        Args:
            input_dir: Path to directory containing .npz files.

        Returns:
            List of motion dictionaries, sorted by filename.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no .npz files are found.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        npz_files = sorted(input_path.glob("*.npz"))
        if not npz_files:
            raise ValueError(f"No .npz files found in {input_dir}")

        motions: list[dict[str, Any]] = []
        for npz_path in npz_files:
            try:
                motion = self.load_single(str(npz_path))
                motions.append(motion)
            except Exception as exc:
                log.warning("Failed to load %s: %s", npz_path.name, exc)
        log.info("Loaded %d/%d motions from %s", len(motions), len(npz_files), input_dir)
        return motions

    def load_single(self, npz_path: str) -> dict[str, Any]:
        """Load a single .npz motion file.

        Args:
            npz_path: Path to the .npz file.

        Returns:
            Dictionary with motion data.
        """
        data = np.load(npz_path, allow_pickle=False)
        motion_name = str(data.get("motion_name", Path(npz_path).stem))
        source_amc = str(data.get("source_amc", ""))
        fps = float(data.get("fps", 120.0))
        num_frames = int(data.get("num_frames", data["angles"].shape[0]))

        motion = {
            "angles": data["angles"].astype(np.float64),
            "joint_names": list(data["joint_names"]),
            "num_frames": num_frames,
            "fps": fps,
            "root_translation": data["root_translation"].astype(np.float64) if "root_translation" in data else None,
            "root_rotation": data["root_rotation"].astype(np.float64) if "root_rotation" in data else None,
            "source_amc": source_amc,
            "motion_name": motion_name,
            "source_npz": os.path.basename(npz_path),
        }
        log.debug("Loaded %s: %d frames, %d DOFs", motion_name, num_frames, motion["angles"].shape[1])
        return motion


# ---------------------------------------------------------------------------
# 2. CMULowerBodySelector
# ---------------------------------------------------------------------------


class CMULowerBodySelector:
    """Extract lower-body channels from 56-DOF CMU motion data."""

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels if channels is not None else CMU_LOWER_BODY_CHANNELS

    def select(self, motion: dict[str, Any]) -> dict[str, Any]:
        """Extract configured channels from a motion.

        Args:
            motion: Motion dictionary from ProcessedCMUGaitLoader.

        Returns:
            Dictionary with per-channel arrays and metadata.

        Raises:
            ValueError: If a requested channel is not found.
        """
        joint_names = motion["joint_names"]
        angles = motion["angles"]

        channel_data: dict[str, np.ndarray] = {}
        missing = []
        for ch in self.channels:
            if ch in joint_names:
                idx = joint_names.index(ch)
                channel_data[ch] = angles[:, idx].copy()
            else:
                missing.append(ch)

        if missing:
            raise ValueError(f"Channels not found in motion: {missing}")

        return {
            "channels": channel_data,
            "num_frames": motion["num_frames"],
            "fps": motion["fps"],
            "motion_name": motion["motion_name"],
            "source_npz": motion["source_npz"],
        }

    def get_channel(self, lower_body: dict[str, Any], channel_name: str) -> np.ndarray:
        """Get a single channel array from selected lower-body data.

        Args:
            lower_body: Output of ``select()``.
            channel_name: Name of the channel to retrieve.

        Returns:
            1-D array of shape (T,).
        """
        return lower_body["channels"][channel_name]

    def print_channel_stats(self, lower_body: dict[str, Any]) -> None:
        """Print range statistics for all channels."""
        print(f"  Channel statistics for {lower_body['source_npz']}:")
        for ch, arr in lower_body["channels"].items():
            print(f"    {ch:16s}: min={arr.min():+7.3f}  max={arr.max():+7.3f}  range={arr.max()-arr.min():6.3f}")


# ---------------------------------------------------------------------------
# 3. Axis analysis
# ---------------------------------------------------------------------------


def analyze_cmu_axis_importance(
    motions: list[dict[str, Any]],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Analyze lower-body DOF characteristics across all motions.

    For each channel, computes:
    - Range of motion (max - min) across all motions
    - Standard deviation
    - Periodicity score (autocorrelation at ~0.5 s lag)
    - Cross-correlation of hip-knee at zero lag

    Args:
        motions: List of motion dictionaries.
        output_path: Optional path to save JSON report.

    Returns:
        Analysis dictionary with per-channel stats and recommendations.
    """
    selector = CMULowerBodySelector()
    all_lower: list[dict[str, Any]] = []
    for m in motions:
        all_lower.append(selector.select(m))

    bone_groups = {
        "lfemur": ["lfemur_rx", "lfemur_ry", "lfemur_rz"],
        "rfemur": ["rfemur_rx", "rfemur_ry", "rfemur_rz"],
        "ltibia": ["ltibia_rx"],
        "rtibia": ["rtibia_rx"],
        "lfoot": ["lfoot_rx", "lfoot_rz"],
        "rfoot": ["rfoot_rx", "rfoot_rz"],
    }

    per_channel_stats: dict[str, dict[str, float]] = {}
    for ch in CMU_LOWER_BODY_CHANNELS:
        all_vals = np.concatenate([lb["channels"][ch] for lb in all_lower])
        global_min = float(all_vals.min())
        global_max = float(all_vals.max())
        global_range = global_max - global_min
        global_std = float(all_vals.std())

        periodicity_scores = []
        for lb in all_lower:
            arr = lb["channels"][ch]
            if len(arr) < 120:
                continue
            arr_c = arr - arr.mean()
            norm = np.sum(arr_c[:60] ** 2) * np.sum(arr_c[60:120] ** 2)
            if norm < 1e-10:
                continue
            score = float(np.sum(arr_c[:60] * arr_c[60:120]) / np.sqrt(norm))
            periodicity_scores.append(score)

        avg_periodicity = float(np.mean(periodicity_scores)) if periodicity_scores else 0.0

        per_channel_stats[ch] = {
            "global_min": global_min,
            "global_max": global_max,
            "global_range": global_range,
            "global_std": global_std,
            "avg_periodicity": avg_periodicity,
        }

    recommended_sagittal: dict[str, str] = {}
    for bone, dofs in bone_groups.items():
        if len(dofs) <= 1:
            if dofs:
                recommended_sagittal[bone] = dofs[0]
            continue
        best_dof = max(dofs, key=lambda d: per_channel_stats[d]["global_range"])
        recommended_sagittal[bone] = best_dof

    hip_knee_corr: dict[str, float] = {}
    for side, hip_ch, knee_ch in [("left", "lfemur_rx", "ltibia_rx"), ("right", "rfemur_rx", "rtibia_rx")]:
        all_hip = np.concatenate([lb["channels"][hip_ch] for lb in all_lower])
        all_knee = np.concatenate([lb["channels"][knee_ch] for lb in all_lower])
        if len(all_hip) > 2 and len(all_knee) > 2:
            min_len = min(len(all_hip), len(all_knee))
            corr = float(np.corrcoef(all_hip[:min_len], all_knee[:min_len])[0, 1])
        else:
            corr = 0.0
        hip_knee_corr[f"{side}_hip_knee_lag0"] = corr

    analysis = {
        "per_channel_stats": per_channel_stats,
        "recommended_sagittal": recommended_sagittal,
        "hip_knee_correlation": hip_knee_corr,
        "current_mapping": {k: v["source"] for k, v in HUMAN_TO_G1_JOINT_MAPPING.items()},
    }

    print("\n" + "=" * 60)
    print("  CMU Axis Importance Analysis")
    print("=" * 60)
    print(f"\n  {'Channel':16s}  {'Range':>7s}  {'Std':>7s}  {'Period':>7s}  {'Rec Sagittal':>14s}")
    print(f"  {'-' * 60}")
    for ch in CMU_LOWER_BODY_CHANNELS:
        s = per_channel_stats[ch]
        rec = ""
        for bone, dof in recommended_sagittal.items():
            if ch == dof:
                rec = f"<-- {bone}"
        print(f"  {ch:16s}  {s['global_range']:7.3f}  {s['global_std']:7.3f}  {s['avg_periodicity']:+7.3f}  {rec}")

    print("\n  Hip-Knee correlations (lag=0):")
    for pair, corr in hip_knee_corr.items():
        print(f"    {pair}: {corr:+.4f}")

    print(f"\n  Recommended sagittal DOFs: {recommended_sagittal}")
    print(f"  Current mapping sources:   {analysis['current_mapping']}")

    if output_path:
        serializable = json.dumps(analysis, indent=2, default=str)
        with open(output_path, "w") as f:
            f.write(serializable)
        log.info("Analysis report saved to %s", output_path)

    return analysis


# ---------------------------------------------------------------------------
# 4. GaitCycleSegmenter
# ---------------------------------------------------------------------------


class GaitCycleSegmenter:
    """Segment a lower-body motion sequence into gait cycles.

    Primary strategy: detect troughs (max flexion) in lfemur_rx using
    scipy ``find_peaks``.  Each pair of consecutive troughs defines one
    gait cycle (left leg swing-to-swing).

    Fallback: if fewer than 2 troughs are found, the entire sequence is
    treated as a single cycle.
    """

    def __init__(
        self,
        fps: float = 120.0,
        min_cycle_frames: int = 30,
        peak_distance: int = 30,
        peak_prominence: float = 0.15,
        channel: str = "lfemur_rx",
    ) -> None:
        self.fps = fps
        self.min_cycle_frames = min_cycle_frames
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.channel = channel

    def segment(self, lower_body: dict[str, Any]) -> list[dict[str, Any]]:
        """Segment a lower-body sequence into gait cycles.

        Args:
            lower_body: Output of ``CMULowerBodySelector.select()``.

        Returns:
            List of cycle dictionaries, each with:
                start_frame, end_frame, duration_s, num_frames, method.
        """
        channels = lower_body["channels"]
        if self.channel not in channels:
            log.warning("Channel %s not found, falling back to single cycle", self.channel)
            return self._make_single_cycle(lower_body["num_frames"])

        signal = channels[self.channel]
        cycles = self._detect_cycles_by_peaks(signal)

        if not cycles:
            log.info(
                "No peaks detected in %s for %s, using single-cycle fallback", self.channel, lower_body["source_npz"]
            )
            return self._make_single_cycle(lower_body["num_frames"])

        return [
            {
                "start_frame": int(s),
                "end_frame": int(e),
                "duration_s": (e - s) / self.fps,
                "num_frames": int(e - s),
                "method": "peak_detection",
            }
            for s, e in cycles
        ]

    def _detect_cycles_by_peaks(self, signal: np.ndarray) -> list[tuple[int, int]]:
        """Detect gait cycles via trough detection in the signal.

        Troughs in lfemur_rx correspond to maximum hip flexion (leg
        forward), which marks mid-swing.  Each pair of consecutive
        troughs spans one full gait cycle.

        Args:
            signal: 1-D array of joint angle values.

        Returns:
            List of (start_frame, end_frame) tuples.
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            log.warning("scipy not available, cannot detect peaks")
            return []

        neg_signal = -signal
        troughs, properties = find_peaks(
            neg_signal,
            distance=self.peak_distance,
            prominence=self.peak_prominence,
        )

        cycles: list[tuple[int, int]] = []
        for i in range(len(troughs) - 1):
            s = int(troughs[i])
            e = int(troughs[i + 1])
            if e - s >= self.min_cycle_frames:
                cycles.append((s, e))

        return cycles

    def _make_single_cycle(self, num_frames: int) -> list[dict[str, Any]]:
        """Create a single cycle spanning the entire sequence.

        Args:
            num_frames: Total number of frames in the sequence.

        Returns:
            List with one cycle dictionary.
        """
        return [
            {
                "start_frame": 0,
                "end_frame": num_frames - 1,
                "duration_s": (num_frames - 1) / self.fps,
                "num_frames": num_frames - 1,
                "method": "single_cycle_fallback",
            }
        ]


# ---------------------------------------------------------------------------
# 5. Resampling
# ---------------------------------------------------------------------------


def resample_cycle_to_16_phases(
    lower_body: dict[str, Any],
    cycle: dict[str, Any],
    num_phases: int = NUM_PHASES,
) -> tuple[list[dict[str, float]], list[int], list[float]]:
    """Resample a gait cycle to a fixed number of phases.

    Uses linear interpolation via ``np.interp`` to resample each channel
    from its original frame count to exactly ``num_phases`` frames.

    Args:
        lower_body: Output of ``CMULowerBodySelector.select()``.
        cycle: Cycle dictionary from ``GaitCycleSegmenter``.
        num_phases: Number of output phases (default 16).

    Returns:
        Tuple of:
            frames -- list of num_phases dicts, each {channel: float}
            phase_indices -- [0, 1, ..., num_phases - 1]
            phase_values -- [0.0, 1/(n-1), ..., 1.0]
    """
    start = cycle["start_frame"]
    end = cycle["end_frame"]
    original_frames = np.arange(start, end + 1, dtype=np.float64)
    target_frames = np.linspace(start, end, num=num_phases)

    phase_indices = list(range(num_phases))
    phase_values = [round(i / max(num_phases - 1, 1), 6) for i in range(num_phases)]

    frames: list[dict[str, float]] = []
    for t in target_frames:
        frame: dict[str, float] = {}
        for ch, arr in lower_body["channels"].items():
            frame[ch] = float(np.interp(t, original_frames, arr[start : end + 1]))
        frames.append(frame)

    return frames, phase_indices, phase_values


# ---------------------------------------------------------------------------
# 6. HumanToG1PoseMapper
# ---------------------------------------------------------------------------


class HumanToG1PoseMapper:
    """Map human (CMU) lower-body channels to a (29,) G1-29dof pose vector.

    Uses the ``HUMAN_TO_G1_JOINT_MAPPING`` configuration to transform each
    CMU channel value into the corresponding G1 joint angle.  Unmapped
    joints retain their default standing/crouching positions.

    Knee joints use a robust affine mapping: human tibia_rx is remapped
    so that the [p5, p95] distribution lands in a safe target range
    away from the G1 joint limits.  Call :meth:`compute_knee_affine_params`
    **before** any calls to :meth:`map_frame_to_g1_pose`.
    """

    _KNEE_ABSTRACT_NAMES = ("left_knee", "right_knee")
    _KNEE_CHANNELS = ("ltibia_rx", "rtibia_rx")
    _KNEE_G1_INDICES = {
        "left_knee": G1_JOINT_NAMES.index("left_knee_joint"),
        "right_knee": G1_JOINT_NAMES.index("right_knee_joint"),
    }

    def __init__(
        self,
        mapping: dict[str, dict[str, float | str]] | None = None,
        default_pose: np.ndarray | None = None,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.mapping = mapping if mapping is not None else HUMAN_TO_G1_JOINT_MAPPING
        self.default_pose = default_pose if default_pose is not None else build_default_g1_pose()
        self.joint_limits = joint_limits if joint_limits is not None else G1_JOINT_LIMITS
        self.mapped_joints: set[str] = set()
        for abstract_name in self.mapping:
            g1_name = abstract_name + "_joint"
            self.mapped_joints.add(g1_name)

        self._knee_affine: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Knee affine parameter computation
    # ------------------------------------------------------------------

    def compute_knee_affine_params(self, all_frames: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        """Compute per-side affine parameters from all raw human frames.

        Must be called **before** :meth:`map_frame_to_g1_pose`.

        Args:
            all_frames: List of dicts mapping CMU channel names to values,
                as produced by :func:`resample_cycle_to_16_phases`.

        Returns:
            Dict ``{side: {p_lo, p_hi, scale, offset, target_low, target_high}}``
            for inspection / logging.
        """
        left_vals = np.array([f.get("ltibia_rx", 0.0) for f in all_frames], dtype=np.float64)
        right_vals = np.array([f.get("rtibia_rx", 0.0) for f in all_frames], dtype=np.float64)

        result: dict[str, dict[str, float]] = {}
        for side, raw in (("left", left_vals), ("right", right_vals)):
            cfg = resolve_knee_config(side)
            p_lo_pct = cfg.get("human_percentile_low", 5)
            p_hi_pct = cfg.get("human_percentile_high", 95)
            target_lo = float(cfg.get("target_low", 0.08))
            target_hi = float(cfg.get("target_high", 1.65))

            p_lo = float(np.percentile(raw, p_lo_pct))
            p_hi = float(np.percentile(raw, p_hi_pct))
            span = p_hi - p_lo
            if span < 1e-6:
                span = 1.0
            scale = (target_hi - target_lo) / span
            offset = target_lo - scale * p_lo

            params = {
                "p_lo": p_lo,
                "p_hi": p_hi,
                "scale": scale,
                "offset": offset,
                "target_low": target_lo,
                "target_high": target_hi,
            }
            self._knee_affine[side] = params
            result[side] = params

        return result

    # ------------------------------------------------------------------
    # Per-joint knee mapping
    # ------------------------------------------------------------------

    def _map_knee(self, side: str, raw_val: float) -> float:
        """Map a single raw human knee value to the G1 knee joint position.

        Args:
            side: ``"left"`` or ``"right"``.
            raw_val: Raw CMU tibia_rx value (rad, >= 0).

        Returns:
            Mapped G1 knee joint position (rad).
        """
        p = self._knee_affine[side]
        return p["scale"] * raw_val + p["offset"]

    # ------------------------------------------------------------------
    # Main mapping method
    # ------------------------------------------------------------------

    def map_frame_to_g1_pose(self, frame: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        """Map one resampled human frame to a (29,) G1 pose vector.

        Args:
            frame: Dict mapping CMU channel names to float values.

        Returns:
            Tuple of:
                pose -- (29,) float32 array
                info -- dict with clip_ratio, clipped_joints, etc.
        """
        pose = self.default_pose.copy()

        mapped_sources: dict[str, str] = {}
        for abstract_name, config in self.mapping.items():
            source_channel = str(config["source"])
            sign = float(config["sign"])
            scale = float(config["scale"])

            channel_val = frame.get(source_channel, 0.0)

            if abstract_name in self._KNEE_ABSTRACT_NAMES:
                side = "left" if "left" in abstract_name else "right"
                g1_val = self._map_knee(side, channel_val)
            else:
                g1_val = sign * scale * channel_val

            g1_name = abstract_name + "_joint"
            if g1_name in G1_JOINT_NAMES:
                idx = G1_JOINT_NAMES.index(g1_name)
                pose[idx] = g1_val
                mapped_sources[abstract_name] = source_channel

        all_default_names = {G1_JOINT_NAMES[i] for i in range(NUM_TOTAL_JOINTS)}
        missing_joints = sorted(all_default_names - self.mapped_joints)

        pose_clipped, clip_ratio, clipped_joints = self._clip_to_limits(pose)

        info: dict[str, Any] = {
            "clip_ratio": clip_ratio,
            "clipped_joints": clipped_joints,
            "mapped_joint_sources": mapped_sources,
            "missing_joints_filled_with_default": missing_joints,
            "upper_body_filled_with_default": True,
        }

        return pose_clipped.astype(np.float32), info

    def _clip_to_limits(self, pose: np.ndarray) -> tuple[np.ndarray, float, list[str]]:
        """Clip pose values to per-joint limits.

        Args:
            pose: (29,) joint angle array.

        Returns:
            Tuple of (clipped_pose, clip_ratio, clipped_joint_names).
        """
        clipped = pose.copy()
        clipped_count = 0
        clipped_names: list[str] = []

        for joint_name, (lo, hi) in self.joint_limits.items():
            if joint_name in G1_JOINT_NAMES:
                idx = G1_JOINT_NAMES.index(joint_name)
                orig = clipped[idx]
                clipped[idx] = np.clip(clipped[idx], lo, hi)
                if not np.isclose(orig, clipped[idx]):
                    clipped_count += 1
                    clipped_names.append(joint_name)

        clip_ratio = clipped_count / NUM_TOTAL_JOINTS
        return clipped, clip_ratio, clipped_names


# ---------------------------------------------------------------------------
# Support foot inference
# ---------------------------------------------------------------------------


def infer_support_foot(phase_index: int, num_phases: int = NUM_PHASES) -> tuple[str, str, bool]:
    """Infer support/swing foot from phase position within gait cycle.

    Heuristic based on cycle segmentation:
    - Cycles are segmented at left hip max flexion troughs (left mid-swing)
    - Phase 0 = start of cycle = left leg at max forward position (swing)
    - Phase ~halfway = left leg now in stance, right leg swings

    Phase heuristic:
    - Phase 0 to (num_phases/2 - 1): left leg still swinging / transitioning
      -> support = right, swing = left
    - Phase num_phases/2 to end: left leg in stance, right leg swings
      -> support = left, swing = right

    Args:
        phase_index: Phase index 0..num_phases-1.
        num_phases: Total number of phases (default 16).

    Returns:
        Tuple of (support_foot, swing_foot, is_inferred).
    """
    half = num_phases // 2
    if phase_index < half:
        return "right", "left", True
    else:
        return "left", "right", True


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------


def build_human_g1_candidate_dataset(
    input_dir: str,
    output_dir: str,
    num_phases: int = NUM_PHASES,
    run_analysis: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """End-to-end pipeline: CMU npz -> G1-29dof candidate poses.

    Args:
        input_dir: Directory with processed CMU .npz files.
        output_dir: Directory for output files.
        num_phases: Phases per gait cycle (default 16, aligned with stage1).
        run_analysis: Whether to run axis analysis before building.

    Returns:
        Tuple of (poses array shape (N, 29), metadata list).
    """
    print("\n" + "=" * 60)
    print("  Human Gait -> G1 Candidate Pose Pipeline")
    print("=" * 60)
    print(f"  Input dir:   {input_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Num phases:  {num_phases}")

    loader = ProcessedCMUGaitLoader()
    motions = loader.load_directory(input_dir)
    print(f"  Loaded motions: {len(motions)}")

    if run_analysis:
        analysis_path = os.path.join(output_dir, "cmu_axis_analysis.json")
        os.makedirs(output_dir, exist_ok=True)
        analyze_cmu_axis_importance(motions, output_path=analysis_path)

    selector = CMULowerBodySelector()
    segmenter = GaitCycleSegmenter()
    mapper = HumanToG1PoseMapper()

    # ------------------------------------------------------------------
    # Pass 1: collect all raw frames for knee affine parameter computation
    # ------------------------------------------------------------------
    all_raw_frames: list[dict[str, float]] = []
    per_motion_data: list[tuple[Any, list[tuple[int, dict[str, Any]]]]] = []

    for motion in motions:
        lower_body = selector.select(motion)
        cycles = segmenter.segment(lower_body)

        source_amc_name = os.path.basename(motion.get("source_amc", ""))
        cycle_data: list[tuple[int, dict[str, Any]]] = []

        for cycle_id, cycle in enumerate(cycles):
            frames, phase_indices, phase_values = resample_cycle_to_16_phases(lower_body, cycle, num_phases)
            all_raw_frames.extend(frames)
            cycle_data.append(
                (
                    cycle_id,
                    {
                        "cycle": cycle,
                        "frames": frames,
                        "phase_indices": phase_indices,
                        "phase_values": phase_values,
                        "source_npz": motion["source_npz"],
                        "source_amc": source_amc_name,
                        "motion_name": motion["motion_name"],
                    },
                )
            )

        per_motion_data.append((motion, cycle_data))

    knee_params = mapper.compute_knee_affine_params(all_raw_frames)
    print("  Knee mapping: robust_affine")
    for side, p in knee_params.items():
        print(
            f"    {side}: p_lo={p['p_lo']:.4f} p_hi={p['p_hi']:.4f} "
            f"scale={p['scale']:.4f} offset={p['offset']:.4f} "
            f"target=[{p['target_low']:.2f}, {p['target_high']:.2f}]"
        )

    # ------------------------------------------------------------------
    # Pass 2: map all frames to G1 poses using computed knee params
    # ------------------------------------------------------------------
    all_poses: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    pose_id = 0

    total_cycles = 0
    total_clipped = 0
    total_clip_ratio_sum = 0.0

    for motion, cycle_data in per_motion_data:
        total_cycles += len(cycle_data)

        log.info(
            "%s: %d frames -> %d cycles (methods: %s)",
            motion["source_npz"],
            motion["num_frames"],
            len(cycle_data),
            [cd[1]["cycle"]["method"] for cd in cycle_data],
        )

        for cycle_id, cd in cycle_data:
            frames = cd["frames"]
            phase_indices = cd["phase_indices"]
            phase_values = cd["phase_values"]
            cycle = cd["cycle"]

            for phase_idx, phase_val, frame in zip(phase_indices, phase_values, frames):
                pose, info = mapper.map_frame_to_g1_pose(frame)
                support_foot, swing_foot, is_inferred = infer_support_foot(phase_idx, num_phases)

                meta_entry: dict[str, Any] = {
                    "pose_id": pose_id,
                    "source": "human_cmu",
                    "source_npz": cd["source_npz"],
                    "source_amc": cd["source_amc"],
                    "cycle_id": cycle_id,
                    "phase_index": phase_idx,
                    "phase_value": phase_val,
                    "clip_ratio": round(info["clip_ratio"], 6),
                    "mapped_joint_sources": info["mapped_joint_sources"],
                    "missing_joints_filled_with_default": info["missing_joints_filled_with_default"],
                    "upper_body_filled_with_default": info["upper_body_filled_with_default"],
                    "support_foot": support_foot,
                    "swing_foot": swing_foot,
                    "support_foot_inferred": is_inferred,
                    "inference_method": "phase_heuristic",
                    "cycle_method": cycle["method"],
                    "cycle_duration_s": round(cycle["duration_s"], 4),
                    "notes": f"CMU {cd['motion_name']} walking gait, cycle {cycle_id}",
                }

                all_poses.append(pose)
                all_meta.append(meta_entry)

                if info["clip_ratio"] > 0:
                    total_clipped += 1
                total_clip_ratio_sum += info["clip_ratio"]
                pose_id += 1

    poses_array = np.array(all_poses, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "human_g1_candidate_poses.npy")
    meta_path = os.path.join(output_dir, "human_g1_candidate_poses_meta.json")

    np.save(npy_path, poses_array)
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  Pipeline Results")
    print("=" * 60)
    print(f"  Total motions:     {len(motions)}")
    print(f"  Total cycles:      {total_cycles}")
    print(f"  Total poses:       {len(all_poses)}")
    print(f"  Pose array shape:  {poses_array.shape}")
    print(f"  Joint range:       [{poses_array.min():.4f}, {poses_array.max():.4f}] rad")
    print(f"  NaN count:         {int(np.isnan(poses_array).sum())}")
    print(f"  Clipped poses:     {total_clipped}/{len(all_poses)} ({100*total_clipped/max(len(all_poses),1):.1f}%)")
    print(f"  Mean clip ratio:   {total_clip_ratio_sum/max(len(all_poses),1):.4f}")
    print(f"  Saved: {npy_path}")
    print(f"  Saved: {meta_path}")

    return poses_array, all_meta
