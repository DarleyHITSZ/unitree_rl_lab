"""COM-based balance feedback controller for SIMBICON.

Implements the core SIMBICON balance feedback:

    theta_d = theta_d0 + c_d * d + c_v * v

where:
    theta_d0 : default swing hip target angle (from FSM pose)
    d        : horizontal COM displacement relative to the support ankle
    v        : horizontal COM velocity
    c_d, c_v : feedback gains

This feedback corrects the swing hip pitch angle to maintain balance
by compensating for COM deviations. It operates independently on each
environment in the batch.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .g1_joint_map import G1JointIndices
    from .simbicon_cfg import BalanceFeedbackCfg


@dataclass
class BalanceFeedbackState:
    """Mutable state for the balance feedback controller.

    Attributes:
        com_velocity_filtered: Exponentially filtered forward COM velocity.
        com_vel_lateral_filtered: Exponentially filtered lateral COM velocity.
        prev_com_pos: Previous COM position for velocity estimation.
    """

    com_velocity_filtered: torch.Tensor | None = None
    com_vel_lateral_filtered: torch.Tensor | None = None
    prev_com_pos: torch.Tensor | None = None
    last_lateral_correction: torch.Tensor | None = None


class BalanceFeedbackController:
    """COM-based balance feedback for the SIMBICON controller.

    Corrects the swing hip pitch angle based on COM position and velocity
    relative to the support ankle. This is the key balancing mechanism in
    SIMBICON that prevents the robot from falling over.

    Args:
        cfg: Balance feedback configuration.
        joint_indices: Resolved G1 joint indices.
    """

    def __init__(
        self,
        cfg: BalanceFeedbackCfg,
        joint_indices: G1JointIndices,
    ) -> None:
        self.cfg = cfg
        self.joint_indices = joint_indices
        self._state = BalanceFeedbackState()
        self._last_correction: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None, num_envs: int = 0, device: str = "cpu") -> None:
        """Reset internal state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
            num_envs: Total number of environments (for full reset).
            device: Torch device string.
        """
        if env_ids is None:
            self._state.com_velocity_filtered = None
            self._state.com_vel_lateral_filtered = None
            self._state.prev_com_pos = None
            self._last_correction = None
            self._state.last_lateral_correction = None
        else:
            if self._state.com_velocity_filtered is not None:
                data = self._state.com_velocity_filtered.clone()
                data[env_ids] = 0.0
                self._state.com_velocity_filtered = data
            if self._state.com_vel_lateral_filtered is not None:
                data = self._state.com_vel_lateral_filtered.clone()
                data[env_ids] = 0.0
                self._state.com_vel_lateral_filtered = data
            if self._state.prev_com_pos is not None:
                data = self._state.prev_com_pos.clone()
                data[env_ids] = 0.0
                self._state.prev_com_pos = data
            if self._last_correction is not None:
                data = self._last_correction.clone()
                data[env_ids] = 0.0
                self._last_correction = data
            if self._state.last_lateral_correction is not None:
                data = self._state.last_lateral_correction.clone()
                data[env_ids] = 0.0
                self._state.last_lateral_correction = data

    def compute_feedback(
        self,
        com_pos: torch.Tensor,
        com_vel: torch.Tensor,
        support_ankle_pos: torch.Tensor,
        swing_foot: torch.Tensor,
        default_swing_hip_pitch: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute balance-corrected swing hip pitch and roll angles.

        Applies the SIMBICON feedback equation to the swing hip:
        - Pitch: theta_d = theta_d0 + cd * dx + cv * vx  (forward balance)
        - Roll:  phi_d = phi_d0 + cd_lat * dy + cv_lat * vy  (lateral balance)

        Args:
            com_pos: COM position in world frame, shape (num_envs, 3).
            com_vel: COM velocity in world frame, shape (num_envs, 3).
            support_ankle_pos: Support ankle position in world frame,
                shape (num_envs, 3).
            swing_foot: Swing foot indicator, shape (num_envs,):
                0=left, 1=right, 2=none.
            default_swing_hip_pitch: Default swing hip pitch from FSM pose
                interpolation, shape (num_envs,).
            dt: Time step for velocity filtering.

        Returns:
            Tuple of:
            - corrected_hip_pitch: Feedback-corrected swing hip pitch,
              shape (num_envs,).
            - com_displacement: Forward COM displacement from support ankle,
              shape (num_envs,).
            - lateral_correction: COM lateral displacement feedback for hip_roll,
              shape (num_envs,).
        """
        num_envs = com_pos.shape[0]
        device = str(com_pos.device)

        com_displacement = self._compute_com_displacement(com_pos, support_ankle_pos)
        com_vel_filtered = self._filter_com_velocity(com_vel, com_pos, dt, num_envs, device)

        lateral_displacement = self._compute_lateral_displacement(com_pos, support_ankle_pos)
        lateral_vel_filtered = self._filter_lateral_velocity(com_vel, dt, num_envs, device)

        pitch_correction = self.cfg.cd * com_displacement + self.cfg.cv * com_vel_filtered
        self._last_correction = pitch_correction.clone()

        lateral_correction = self.cfg.cd * lateral_displacement + self.cfg.cv * lateral_vel_filtered
        self._state.last_lateral_correction = lateral_correction.clone()

        no_swing = swing_foot == 2
        pitch_correction = torch.where(no_swing, torch.zeros_like(pitch_correction), pitch_correction)
        lateral_correction = torch.where(no_swing, torch.zeros_like(lateral_correction), lateral_correction)

        corrected_hip_pitch = default_swing_hip_pitch + pitch_correction
        return corrected_hip_pitch, com_displacement, lateral_correction

    def get_debug_info(self) -> dict[str, torch.Tensor | None]:
        """Return debug information about the last feedback computation.

        Returns:
            Dictionary with 'correction' and 'com_velocity_filtered'.
        """
        return {
            "correction": self._last_correction,
            "lateral_correction": self._state.last_lateral_correction,
            "com_velocity_filtered": self._state.com_velocity_filtered,
        }

    def _compute_com_displacement(
        self,
        com_pos: torch.Tensor,
        support_ankle_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward (sagittal) COM displacement relative to support ankle.

        Uses only the x-component (forward direction) as in the standard
        SIMBICON formulation: d = com_x - stance_ankle_x.

        Args:
            com_pos: COM position, shape (num_envs, 3).
            support_ankle_pos: Support ankle position, shape (num_envs, 3).

        Returns:
            Forward displacement, shape (num_envs,).
        """
        return com_pos[:, 0] - support_ankle_pos[:, 0]

    def _compute_lateral_displacement(
        self,
        com_pos: torch.Tensor,
        support_ankle_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lateral COM displacement relative to support ankle.

        Uses the y-component: d_lat = com_y - stance_ankle_y.

        Args:
            com_pos: COM position, shape (num_envs, 3).
            support_ankle_pos: Support ankle position, shape (num_envs, 3).

        Returns:
            Lateral displacement, shape (num_envs,).
        """
        return com_pos[:, 1] - support_ankle_pos[:, 1]

    def _filter_com_velocity(
        self,
        com_vel: torch.Tensor,
        com_pos: torch.Tensor,
        dt: float,
        num_envs: int,
        device: str,
    ) -> torch.Tensor:
        """Apply exponential moving average filter to COM velocity.

        Args:
            com_vel: Raw COM velocity, shape (num_envs, 3).
            com_pos: Current COM position, shape (num_envs, 3).
            dt: Time step.
            num_envs: Number of environments.
            device: Torch device string.

        Returns:
            Filtered horizontal COM velocity magnitude, shape (num_envs,).
        """
        if self._state.com_velocity_filtered is None or self._state.com_velocity_filtered.shape[0] != num_envs:
            self._state.com_velocity_filtered = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self._state.prev_com_pos = com_pos.clone()

        raw_vel_x = com_vel[:, 0]

        alpha = self.cfg.com_filter_alpha
        assert self._state.com_velocity_filtered is not None
        filtered = alpha * raw_vel_x + (1.0 - alpha) * self._state.com_velocity_filtered
        self._state.com_velocity_filtered = filtered
        self._state.prev_com_pos = com_pos.clone()

        return filtered

    def _filter_lateral_velocity(
        self,
        com_vel: torch.Tensor,
        dt: float,
        num_envs: int,
        device: str,
    ) -> torch.Tensor:
        """Apply exponential moving average filter to lateral COM velocity.

        Args:
            com_vel: Raw COM velocity, shape (num_envs, 3).
            dt: Time step.
            num_envs: Number of environments.
            device: Torch device string.

        Returns:
            Filtered lateral COM velocity, shape (num_envs,).
        """
        if (
            self._state.com_vel_lateral_filtered is None
            or self._state.com_vel_lateral_filtered.shape[0] != num_envs
        ):
            self._state.com_vel_lateral_filtered = torch.zeros(num_envs, dtype=torch.float32, device=device)

        raw_vel_y = com_vel[:, 1]

        alpha = self.cfg.com_filter_alpha
        assert self._state.com_vel_lateral_filtered is not None
        filtered = alpha * raw_vel_y + (1.0 - alpha) * self._state.com_vel_lateral_filtered
        self._state.com_vel_lateral_filtered = filtered

        return filtered
