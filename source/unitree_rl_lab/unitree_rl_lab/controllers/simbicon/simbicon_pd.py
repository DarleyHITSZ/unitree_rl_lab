"""PD joint tracking controller for the SIMBICON gait controller.

Implements a configurable per-joint PD controller that computes joint
torques (or position targets) to track desired trajectories. Supports
different gain groups for different joint types.

The PD control law is:

    tau = Kp * (q_d - q) - Kd * (q_dot - q_dot_d)

where q_d is the desired position, q is the current position, q_dot is the
current velocity, and q_dot_d is the desired velocity (default 0).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simbicon_cfg import PDGainsCfg


@dataclass
class PDControllerState:
    """Mutable state for the PD controller across time steps.

    Attributes:
        prev_joint_vel: Previous joint velocities for acceleration estimation.
        prev_action: Previous target positions for action rate computation.
    """

    prev_joint_vel: torch.Tensor | None = None
    prev_action: torch.Tensor | None = None


class PDController:
    """Per-joint PD tracking controller with configurable gains.

    Supports computing both joint torques (for direct torque control) and
    position targets (for the environment's implicit PD actuators).

    Args:
        joint_names: List of abstract joint names (15 controllable joints).
        gains_map: Mapping from joint name to PD gains.
    """

    def __init__(
        self,
        joint_names: list[str],
        gains_map: dict[str, PDGainsCfg] | None = None,
    ) -> None:
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.gains_map: dict[str, PDGainsCfg] = gains_map or {}
        self._state = PDControllerState()

    def set_gains(self, joint_name: str, kp: float, kd: float) -> None:
        """Set PD gains for a specific joint.

        Args:
            joint_name: Abstract joint name.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        from .simbicon_cfg import PDGainsCfg

        self.gains_map[joint_name] = PDGainsCfg(kp=kp, kd=kd)

    def compute_torques(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
        q_desired: torch.Tensor,
        q_dot_desired: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques using the PD control law.

        Args:
            q: Current joint positions, shape (num_envs, num_joints).
            q_dot: Current joint velocities, shape (num_envs, num_joints).
            q_desired: Desired joint positions, shape (num_envs, num_joints).
            q_dot_desired: Desired joint velocities, shape (num_envs, num_joints).
                If None, defaults to zero.

        Returns:
            Joint torques, shape (num_envs, num_joints).
        """
        if q_dot_desired is None:
            q_dot_desired = torch.zeros_like(q)

        kp = self._get_kp_tensor(str(q.device), q.shape[0])
        kd = self._get_kd_tensor(str(q.device), q.shape[0])

        tau = kp * (q_desired - q) - kd * (q_dot - q_dot_desired)
        return tau

    def compute_position_targets(
        self,
        q_desired: torch.Tensor,
    ) -> torch.Tensor:
        """Pass through desired positions (for implicit actuator PD control).

        When the environment uses ImplicitActuatorCfg, the PD tracking is
        handled by the physics engine. This method simply returns the desired
        positions unchanged, allowing the caller to convert them to actions.

        Args:
            q_desired: Desired joint positions, shape (num_envs, num_joints).

        Returns:
            Same as q_desired, shape (num_envs, num_joints).
        """
        self._state.prev_action = q_desired.clone()
        return q_desired

    def get_action_rate(self, q_desired: torch.Tensor) -> torch.Tensor:
        """Compute the rate of change of desired positions.

        Args:
            q_desired: Current desired positions, shape (num_envs, num_joints).

        Returns:
            Action rate (change per step), shape (num_envs, num_joints).
        """
        if self._state.prev_action is None:
            self._state.prev_action = q_desired.clone()
            return torch.zeros_like(q_desired)
        rate = q_desired - self._state.prev_action
        self._state.prev_action = q_desired.clone()
        return rate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        self._state.prev_action = None
        self._state.prev_joint_vel = None

    def _get_kp_tensor(self, device: str, num_envs: int) -> torch.Tensor:
        """Build per-joint Kp tensor.

        Returns:
            Tensor of shape (1, num_joints) with per-joint proportional gains.
        """
        kp_values = []
        for name in self.joint_names:
            gains = self.gains_map.get(name)
            kp_values.append(gains.kp if gains else 100.0)
        return torch.tensor(kp_values, dtype=torch.float32, device=device).unsqueeze(0)

    def _get_kd_tensor(self, device: str, num_envs: int) -> torch.Tensor:
        """Build per-joint Kd tensor.

        Returns:
            Tensor of shape (1, num_joints) with per-joint derivative gains.
        """
        kd_values = []
        for name in self.joint_names:
            gains = self.gains_map.get(name)
            kd_values.append(gains.kd if gains else 5.0)
        return torch.tensor(kd_values, dtype=torch.float32, device=device).unsqueeze(0)
