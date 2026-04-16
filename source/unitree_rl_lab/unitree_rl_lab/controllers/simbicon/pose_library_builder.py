"""Pose library builder for MSLPO Stage 1 → Stage 2 pipeline.

Converts the top-5 gait parameter sets from Q-learning (Stage 1) into a
discrete pose library of 160 poses (5 groups × 2 states × 16 phases) for
use as the action space in PPO (Stage 2).

This module works purely offline — no Isaac Sim or GPU required. It
reuses the SimbiconFSM pose definitions and interpolation logic to
generate target joint poses without simulation rollouts.

Usage::

    from unitree_rl_lab.controllers.simbicon.pose_library_builder import (
        Stage1PoseLibraryBuilder,
    )

    builder = Stage1PoseLibraryBuilder(num_phases=16)
    poses, meta = builder.build_from_top5(
        top5_path="outputs/qlearn_search/top5_pose_params.json",
        output_dir="outputs",
    )
"""

from __future__ import annotations

import json
import numpy as np
import os
from pathlib import Path
from typing import Any

from .g1_joint_map import ABSTRACT_JOINT_NAMES, DEFAULT_JOINT_POSITIONS, G1_JOINT_NAMES, NUM_TOTAL_JOINTS
from .simbicon_cfg import SimbiconCfg
from .simbicon_fsm import STATE_NAMES, SimbiconFSM, SimbiconState

CORE_STATES: list[str] = [
    "STEP_RIGHT_WITH_LEFT_FRONT",
    "STEP_LEFT_WITH_RIGHT_FRONT",
]

STATE_SUPPORT_FOOT: dict[str, str] = {
    "STANCE_CROUCH": "both",
    "START_STEP_LEFT": "right",
    "STEP_RIGHT_WITH_LEFT_FRONT": "left",
    "STEP_LEFT_WITH_RIGHT_FRONT": "right",
    "RECOVER_CROUCH": "both",
}

STATE_SWING_FOOT: dict[str, str] = {
    "STANCE_CROUCH": "none",
    "START_STEP_LEFT": "left",
    "STEP_RIGHT_WITH_LEFT_FRONT": "right",
    "STEP_LEFT_WITH_RIGHT_FRONT": "left",
    "RECOVER_CROUCH": "none",
}


def load_top5_params(top5_path: str) -> list[dict[str, Any]]:
    """Load and validate the top-5 gait parameter sets from Stage 1.

    Args:
        top5_path: Path to ``top5_pose_params.json``.

    Returns:
        List of 5 parameter dictionaries, each containing at least
        ``HL``, ``Ls``, ``Lswb``, ``Lforward``.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the file does not contain exactly 5 entries or
            required parameter keys are missing.
    """
    path = Path(top5_path)
    if not path.exists():
        raise FileNotFoundError(f"Top-5 params file not found: {top5_path}")

    with open(path, "r") as f:
        params_list = json.load(f)

    if not isinstance(params_list, list) or len(params_list) != 5:
        raise ValueError(f"Expected a list of exactly 5 parameter sets, got {type(params_list)}")

    required_keys = {"HL", "Ls", "Lswb", "Lforward"}
    for i, params in enumerate(params_list):
        missing = required_keys - set(params.keys())
        if missing:
            raise ValueError(f"Param group {i} missing keys: {missing}")

    return params_list


def build_default_full_pos() -> np.ndarray:
    """Build the full 29-DOF default joint position vector.

    Uses ``DEFAULT_JOINT_POSITIONS`` from the G1 joint map. Joints not
    explicitly listed default to 0.0.

    Returns:
        NumPy array of shape (29,) with default joint positions.
    """
    defaults = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
    for name, val in DEFAULT_JOINT_POSITIONS.items():
        if name in G1_JOINT_NAMES:
            idx = G1_JOINT_NAMES.index(name)
            defaults[idx] = val
    return defaults


def sample_state_poses(
    pose_def_start: dict[str, float],
    pose_def_end: dict[str, float],
    num_phases: int,
    default_full_pos: np.ndarray,
) -> np.ndarray:
    """Sample discrete poses from a single FSM state at fixed phase intervals.

    Linearly interpolates between the state's start and end poses at
    ``num_phases`` equally-spaced phase values in [0, 1], then expands
    the 12 controllable DOFs to the full 29-DOF vector.

    Args:
        pose_def_start: Start pose dictionary (12 abstract joint names).
        pose_def_end: End pose dictionary (12 abstract joint names).
        num_phases: Number of discrete phase samples.
        default_full_pos: Default 29-DOF joint positions, shape (29,).

    Returns:
        NumPy array of shape (num_phases, 29) with sampled poses.
    """
    poses = np.zeros((num_phases, NUM_TOTAL_JOINTS), dtype=np.float32)
    for i in range(num_phases):
        phase = i / max(num_phases - 1, 1)
        controllable = np.zeros(len(ABSTRACT_JOINT_NAMES), dtype=np.float32)
        for j, abstract_name in enumerate(ABSTRACT_JOINT_NAMES):
            start_val = pose_def_start.get(abstract_name, 0.0)
            end_val = pose_def_end.get(abstract_name, 0.0)
            controllable[j] = start_val + phase * (end_val - start_val)
        full = default_full_pos.copy()
        for j, abstract_name in enumerate(ABSTRACT_JOINT_NAMES):
            g1_name = abstract_name + "_joint"
            if g1_name in G1_JOINT_NAMES:
                idx = G1_JOINT_NAMES.index(g1_name)
                full[idx] = controllable[j]
        poses[i] = full
    return poses


def encode_pose_action_index(num_groups: int, num_states: int, num_phases: int) -> list[dict[str, int]]:
    """Generate action index mapping metadata.

    Args:
        num_groups: Number of parameter groups (5).
        num_states: Number of core FSM states (2).
        num_phases: Number of phase samples per state (16).

    Returns:
        List of dictionaries with index layout information.
    """
    mapping = []
    idx = 0
    for g in range(num_groups):
        for s in range(num_states):
            for p in range(num_phases):
                mapping.append(
                    {
                        "action_idx": idx,
                        "param_group_idx": g,
                        "state_idx": s,
                        "phase_index": p,
                    }
                )
                idx += 1
    return mapping


def validate_pose_library(poses: np.ndarray, meta: list[dict[str, Any]]) -> list[str]:
    """Validate the pose library integrity.

    Args:
        poses: Pose array, expected shape (160, 29).
        meta: Metadata list, expected length 160.

    Returns:
        List of error/warning strings. Empty list means all checks pass.
    """
    issues: list[str] = []

    if poses.shape != (160, 29):
        issues.append(f"Expected shape (160, 29), got {poses.shape}")

    if len(meta) != 160:
        issues.append(f"Expected 160 meta entries, got {len(meta)}")

    if np.any(np.isnan(poses)):
        nan_count = int(np.sum(np.isnan(poses)))
        issues.append(f"Found {nan_count} NaN values in pose array")

    if np.any(np.isinf(poses)):
        inf_count = int(np.sum(np.isinf(poses)))
        issues.append(f"Found {inf_count} Inf values in pose array")

    joint_range = np.abs(poses).max()
    if joint_range > 2.0:
        issues.append(f"Max absolute joint value {joint_range:.3f} exceeds 2.0 rad sanity threshold")

    group_counts: dict[int, int] = {}
    state_counts: dict[str, int] = {}
    action_indices = set()
    for entry in meta:
        gid = entry.get("param_group_idx", -1)
        group_counts[gid] = group_counts.get(gid, 0) + 1
        sname = entry.get("fsm_state", "")
        state_counts[sname] = state_counts.get(sname, 0) + 1
        action_indices.add(entry.get("action_idx", -1))

    for g in range(5):
        if group_counts.get(g, 0) != 32:
            issues.append(f"Param group {g} has {group_counts.get(g, 0)} poses, expected 32")

    for sname in CORE_STATES:
        if state_counts.get(sname, 0) != 80:
            issues.append(f"State '{sname}' has {state_counts.get(sname, 0)} poses, expected 80")

    expected_indices = set(range(160))
    if action_indices != expected_indices:
        missing = expected_indices - action_indices
        extra = action_indices - expected_indices
        if missing:
            issues.append(f"Missing action indices: {sorted(missing)}")
        if extra:
            issues.append(f"Extra action indices: {sorted(extra)}")

    for entry in meta:
        pv = entry.get("phase_value", -1)
        if pv < 0.0 or pv > 1.0 + 1e-6:
            issues.append(f"action_idx={entry.get('action_idx')} has invalid phase_value={pv}")

    return issues


def export_pose_library(
    poses: np.ndarray,
    meta: list[dict[str, Any]],
    output_dir: str,
) -> tuple[str, str]:
    """Save the pose library and metadata to disk.

    Args:
        poses: Pose array, shape (N, 29).
        meta: Metadata list.
        output_dir: Directory to write files into.

    Returns:
        Tuple of (npy_path, meta_path) for the saved files.
    """
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, "pose_library.npy")
    np.save(npy_path, poses.astype(np.float32))

    meta_path = os.path.join(output_dir, "pose_library_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return npy_path, meta_path


class Stage1PoseLibraryBuilder:
    """Builds a discrete pose library from Stage 1 top-5 gait parameters.

    For each of the 5 parameter groups, this builder:
    1. Injects the parameters into a SimbiconFSM (offline, no simulation).
    2. Samples poses from the 2 core stepping states at 16 fixed phases.
    3. Expands the 12 controllable DOFs to the full 29-DOF vector.

    The result is a (160, 29) pose array and associated metadata, ready
    for use as a discrete action space in Stage 2 PPO.

    Args:
        num_phases: Number of discrete phase samples per state (default 16).
        core_states: FSM state names to include (default: the two stepping
            states).
    """

    def __init__(
        self,
        num_phases: int = 16,
        core_states: list[str] | None = None,
    ) -> None:
        self.num_phases = num_phases
        self.core_states = core_states or CORE_STATES
        self._default_full_pos = build_default_full_pos()

    def build_from_top5(
        self,
        top5_path: str,
        output_dir: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Build the pose library from top-5 parameter sets.

        Args:
            top5_path: Path to ``top5_pose_params.json``.
            output_dir: Directory to save ``pose_library.npy`` and
                ``pose_library_meta.json``.

        Returns:
            Tuple of (poses array shape (160, 29), metadata list of 160 dicts).
        """
        params_list = load_top5_params(top5_path)
        return self.build(params_list, output_dir)

    def build(
        self,
        params_list: list[dict[str, Any]],
        output_dir: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Build the pose library from a list of parameter sets.

        Args:
            params_list: List of parameter dictionaries (at least 1, up to 5).
            output_dir: Directory to save outputs.

        Returns:
            Tuple of (poses, metadata).
        """
        num_groups = len(params_list)
        num_states = len(self.core_states)
        total_poses = num_groups * num_states * self.num_phases

        all_poses = np.zeros((total_poses, NUM_TOTAL_JOINTS), dtype=np.float32)
        all_meta: list[dict[str, Any]] = []

        cfg = SimbiconCfg(continuous_walking=True, max_steps=100)
        fsm = SimbiconFSM(cfg=cfg)

        idx = 0
        for g, params in enumerate(params_list):
            hl = params["HL"]
            ls = params["Ls"]
            lswb = params["Lswb"]
            lforward = params["Lforward"]

            fsm.update_pose_from_params(hl=hl, ls=ls, lswb=lswb, lforward=lforward)

            for s, state_name in enumerate(self.core_states):
                state_enum = STATE_NAMES_inv(state_name)
                pose_def = fsm.pose_defs[state_enum]

                state_poses = sample_state_poses(
                    pose_def_start=pose_def.start_pose,
                    pose_def_end=pose_def.end_pose,
                    num_phases=self.num_phases,
                    default_full_pos=self._default_full_pos,
                )

                for p in range(self.num_phases):
                    phase_value = p / max(self.num_phases - 1, 1)
                    all_poses[idx] = state_poses[p]
                    all_meta.append(
                        {
                            "action_idx": idx,
                            "param_group_idx": g,
                            "source": "stage1",
                            "fsm_state": state_name,
                            "phase_index": p,
                            "phase_value": round(phase_value, 6),
                            "HL": hl,
                            "Ls": ls,
                            "Lswb": lswb,
                            "Lforward": lforward,
                            "support_foot": STATE_SUPPORT_FOOT.get(state_name, "both"),
                            "swing_foot": STATE_SWING_FOOT.get(state_name, "none"),
                            "rank": params.get("rank", g + 1),
                            "total_reward": params.get("total_reward", 0.0),
                            "notes": f"Top-{params.get('rank', g+1)} params from Q-learning",
                        }
                    )
                    idx += 1

        issues = validate_pose_library(all_poses, all_meta)
        if issues:
            print("[WARN] Pose library validation issues:")
            for issue in issues:
                print(f"  - {issue}")

        npy_path, meta_path = export_pose_library(all_poses, all_meta, output_dir)
        print(f"[INFO] Pose library saved: {npy_path} (shape={all_poses.shape})")
        print(f"[INFO] Metadata saved:     {meta_path} ({len(all_meta)} entries)")

        return all_poses, all_meta


def STATE_NAMES_inv(state_name: str) -> SimbiconState:
    """Reverse lookup: state name string → SimbiconState enum.

    Args:
        state_name: FSM state name string.

    Returns:
        Corresponding SimbiconState enum value.

    Raises:
        ValueError: If the state name is not recognized.
    """
    for state, name in STATE_NAMES.items():
        if name == state_name:
            return state
    valid = list(STATE_NAMES.values())
    raise ValueError(f"Unknown FSM state '{state_name}'. Valid: {valid}")
