"""Human (CMU Mocap) to G1-29dof joint mapping configuration.

This module defines the standalone mapping between CMU Mocap lower-body DOFs
and the G1-29dof robot's 29-DOF joint space.  It is used by the
``human_gait_to_g1.py`` pipeline and has **no** dependency on IsaacLab or
Isaac Sim.

Axis convention analysis (verified on subject_07 walking data):

    - femur_rx : sagittal-plane flexion/extension (range ~1.5 rad, periodicity ~0.93)
                 negative = flexion (leg forward), positive = extension (leg backward)
    - femur_rz : frontal-plane abduction/adduction   (range ~0.5-0.7 rad, periodicity ~0.85)
                 left: mostly negative; right: mostly positive
    - femur_ry : transverse-plane rotation            (range ~0.4 rad, periodicity ~0.2)
    - tibia_rx : knee flexion                         (range ~1.3 rad, always >= 0)
                 0 = full extension, ~1.3 = deep flexion
    - foot_rx  : dorsiflexion / plantarflexion        (range ~0.8 rad)
    - foot_rz  : inversion / eversion                 (range ~0.6-0.8 rad)

G1-29dof convention (matching SIMBICON controller):

    - hip_pitch : negative = leg tilts forward, positive = backward
    - knee      : positive = flexion (bent), 0 = straight
    - ankle_pitch : negative = plantarflexion (toes down), positive = dorsiflexion

Sign derivation:
    CMU femur_rx and G1 hip_pitch share the same sign convention (negative = forward).
    Therefore the mapping uses sign = +1.0 (no flip) for hip_pitch.

    CMU tibia_rx is always non-negative (flexion), matching G1 knee convention.
    Therefore the mapping uses sign = +1.0 for knee.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# G1-29dof joint names (SDK order, indices 0-28)
# ---------------------------------------------------------------------------

G1_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NUM_TOTAL_JOINTS: int = 29

# ---------------------------------------------------------------------------
# Default joint positions (from UNITREE_G1_29DOF_CFG init_state)
# Joints not listed default to 0.0.
# ---------------------------------------------------------------------------

DEFAULT_JOINT_POSITIONS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}


def build_default_g1_pose() -> np.ndarray:
    """Build the full (29,) default joint position vector.

    Returns:
        Float64 array of shape (29,) with default joint positions.
    """
    defaults = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float64)
    for name, val in DEFAULT_JOINT_POSITIONS.items():
        if name in G1_JOINT_NAMES:
            idx = G1_JOINT_NAMES.index(name)
            defaults[idx] = val
    return defaults


# ---------------------------------------------------------------------------
# Per-joint limits (conservative first version)
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

# ---------------------------------------------------------------------------
# Human (CMU) -> G1 joint mapping
#
# Each entry maps a G1 abstract joint name (without "_joint" suffix) to a
# CMU channel.  ``sign`` and ``scale`` are applied as:
#     g1_value = sign * scale * cmu_channel_value
# ---------------------------------------------------------------------------

HUMAN_TO_G1_JOINT_MAPPING: dict[str, dict[str, float | str]] = {
    # Primary sagittal chain (rx axes)
    "left_hip_pitch": {"source": "lfemur_rx", "sign": 1.0, "scale": 1.0},
    "left_knee": {"source": "ltibia_rx", "sign": 1.0, "scale": 1.0},
    "left_ankle_pitch": {"source": "lfoot_rx", "sign": 1.0, "scale": 1.0},
    "right_hip_pitch": {"source": "rfemur_rx", "sign": 1.0, "scale": 1.0},
    "right_knee": {"source": "rtibia_rx", "sign": 1.0, "scale": 1.0},
    "right_ankle_pitch": {"source": "rfoot_rx", "sign": 1.0, "scale": 1.0},
    # Secondary frontal chain (rz axes)
    "left_hip_roll": {"source": "lfemur_rz", "sign": 1.0, "scale": 1.0},
    "right_hip_roll": {"source": "rfemur_rz", "sign": 1.0, "scale": 1.0},
    # Tertiary transverse chain (ry axes)
    "left_hip_yaw": {"source": "lfemur_ry", "sign": 1.0, "scale": 1.0},
    "right_hip_yaw": {"source": "rfemur_ry", "sign": 1.0, "scale": 1.0},
}

# ---------------------------------------------------------------------------
# Knee mapping -- robust affine remap
#
# Maps human tibia_rx to G1 knee position via affine transform:
#     g1_knee = scale * (tibia_rx - p_lo) + target_low
# where scale = (target_high - target_low) / (p_hi - p_lo)
# and p_lo/p_hi are robust percentiles of tibia_rx across all frames.
#
# Per-side overrides: set left_specific / right_specific to a dict with
# any subset of the same keys to override for one side only.
# ---------------------------------------------------------------------------

from typing import Any

KNEE_MAPPING_CONFIG: dict[str, Any] = {
    "target_low": 0.08,
    "target_high": 1.65,
    "human_percentile_low": 5,
    "human_percentile_high": 95,
    "left_specific": None,
    "right_specific": None,
}


def resolve_knee_config(side: str) -> dict[str, Any]:
    """Return per-side effective config by merging base + side-specific override."""
    base = dict(KNEE_MAPPING_CONFIG)
    override = base.pop(f"{side}_specific", None)
    if override is not None and isinstance(override, dict):
        base.update(override)
    return base


# ---------------------------------------------------------------------------
# CMU lower-body channel names used for extraction
# ---------------------------------------------------------------------------

CMU_LOWER_BODY_CHANNELS: list[str] = [
    "lfemur_rx",
    "lfemur_ry",
    "lfemur_rz",
    "ltibia_rx",
    "lfoot_rx",
    "lfoot_rz",
    "ltoes_rx",
    "rfemur_rx",
    "rfemur_ry",
    "rfemur_rz",
    "rtibia_rx",
    "rfoot_rx",
    "rfoot_rz",
    "rtoes_rx",
]

# ---------------------------------------------------------------------------
# Phase discretization (aligned with stage1 pose library)
# ---------------------------------------------------------------------------

NUM_PHASES: int = 16
