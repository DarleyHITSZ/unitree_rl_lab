"""Discrete pose library loader for MSLPO Stage 2 PPO.

Loads the expanded pose library (216 discrete poses, each 29-DOF) and provides
lookup by action index. Designed to be used inside Isaac Sim scripts that need
to map PPO's categorical output to joint position targets.

Usage::

    from unitree_rl_lab.controllers.simbicon.discrete_pose_library import DiscretePoseLibrary

    lib = DiscretePoseLibrary(
        library_path="outputs/pose_library/pose_library_expanded.npy",
        meta_path="outputs/pose_library/pose_library_expanded_meta.json",
        device="cuda:0",
    )
    pose = lib.get_pose(42)          # (29,) tensor
    poses = lib.get_pose(indices)    # (num_envs, 29) tensor
"""

from __future__ import annotations

import json
import numpy as np
import os
import torch
from typing import Any


class DiscretePoseLibrary:
    """Expanded pose library with discrete action indexing.

    Attributes:
        poses: All poses as a (num_actions, 29) tensor on device.
        meta: Per-pose metadata list loaded from JSON.
        num_actions: Number of discrete actions (216 for the expanded library).
        num_joints: Number of joints per pose (29 for G1-29dof).
    """

    def __init__(
        self,
        library_path: str,
        meta_path: str,
        device: str = "cuda:0",
    ) -> None:
        if not os.path.isfile(library_path):
            raise FileNotFoundError(f"Pose library not found: {library_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Pose library metadata not found: {meta_path}")

        raw = np.load(library_path).astype(np.float32)
        self.poses: torch.Tensor = torch.from_numpy(raw).to(device)
        self.poses.requires_grad_(False)

        with open(meta_path, "r") as f:
            self.meta: list[dict[str, Any]] = json.load(f)

        self._device = device
        self.num_actions: int = self.poses.shape[0]
        self.num_joints: int = self.poses.shape[1]

        if len(self.meta) != self.num_actions:
            raise ValueError(
                f"Pose library has {self.num_actions} poses but metadata has " f"{len(self.meta)} entries."
            )

    def get_pose(self, indices: int | torch.Tensor) -> torch.Tensor:
        """Look up pose(s) by action index.

        Args:
            indices: A single integer or a (num_envs,) integer tensor.

        Returns:
            Pose tensor of shape (29,) for scalar input or (num_envs, 29) for
            tensor input.
        """
        return self.poses[indices]

    def get_meta(self, idx: int) -> dict[str, Any]:
        """Return metadata dict for a single action index."""
        return self.meta[idx]

    def get_all_poses(self) -> torch.Tensor:
        """Return the full pose matrix (num_actions, 29)."""
        return self.poses

    @property
    def device(self) -> str:
        return self._device

    def __len__(self) -> int:
        return self.num_actions

    def __repr__(self) -> str:
        return (
            f"DiscretePoseLibrary(num_actions={self.num_actions}, "
            f"num_joints={self.num_joints}, device={self._device})"
        )
