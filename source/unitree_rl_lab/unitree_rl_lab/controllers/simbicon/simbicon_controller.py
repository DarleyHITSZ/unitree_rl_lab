"""Main SIMBICON-style bipedal gait controller.

Orchestrates the FSM, PD controller, and balance feedback modules to produce
joint position targets for the G1-29dof robot. Designed to be usable both
as a standalone controller and as a component in the MSLPO framework.

The controller is fully vectorized and supports batched simulation across
multiple parallel environments.

Usage example::

    from unitree_rl_lab.controllers import SimbiconController, SimbiconCfg

    cfg = SimbiconCfg()
    controller = SimbiconController(cfg)
    controller.initialize(robot, num_envs=64, device="cuda:0")

    # Each simulation step:
    target_positions = controller.step(state_data)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .g1_joint_map import ABSTRACT_JOINT_NAMES, NUM_CONTROLLABLE_JOINTS, G1JointIndices
from .simbicon_balance import BalanceFeedbackController
from .simbicon_cfg import SimbiconCfg
from .simbicon_fsm import STATE_NAMES, SimbiconFSM, SimbiconState
from .simbicon_pd import PDController

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@dataclass
class SimbiconStateData:
    """Input state data for the SIMBICON controller.

    Collects all information needed by the controller from the simulation
    environment. Designed to decouple the controller from any specific
    environment implementation.

    Attributes:
        joint_pos: Full joint positions, shape (num_envs, 29).
        joint_vel: Full joint velocities, shape (num_envs, 29).
        root_pos: Root body position (world frame), shape (num_envs, 3).
        root_quat: Root body orientation (world frame), shape (num_envs, 4).
        root_lin_vel: Root body linear velocity, shape (num_envs, 3).
        root_ang_vel: Root body angular velocity, shape (num_envs, 3).
        body_pos: All body positions, shape (num_envs, num_bodies, 3).
        left_foot_contact: Left foot contact flag, shape (num_envs,).
        right_foot_contact: Right foot contact flag, shape (num_envs,).
        left_foot_force: Left foot contact force magnitude, shape (num_envs,).
        right_foot_force: Right foot contact force magnitude, shape (num_envs,).
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    body_pos: torch.Tensor
    left_foot_contact: torch.Tensor
    right_foot_contact: torch.Tensor
    left_foot_force: torch.Tensor = torch.zeros(0)
    right_foot_force: torch.Tensor = torch.zeros(0)


class SimbiconController:
    """SIMBICON-style bipedal gait controller for G1-29dof.

    Combines a 5-state FSM, per-state target pose interpolation, PD joint
    tracking, and COM-based balance feedback to produce walking gaits.

    The controller outputs full 29-DOF joint position targets. Controllable
    leg joints (12 DOFs) are set by the SIMBICON algorithm; waist (3 DOFs)
    and arm joints (14 DOFs) are held at their default positions.

    Args:
        cfg: Controller configuration.
    """

    def __init__(self, cfg: SimbiconCfg | None = None) -> None:
        self.cfg = cfg or SimbiconCfg()
        self.joint_indices = G1JointIndices()
        self.fsm = SimbiconFSM(cfg=self.cfg)
        self.pd_controller = PDController(ABSTRACT_JOINT_NAMES, self.cfg.pd_gains)
        self.balance_controller: BalanceFeedbackController | None = None
        self._initialized = False
        self._device = "cpu"
        self._num_envs = 0
        self._default_full_pos: torch.Tensor | None = None

    def initialize(
        self,
        robot: Articulation,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the controller with a live robot articulation.

        Resolves joint and body indices from the articulation and allocates
        internal tensors.

        Args:
            robot: The G1-29dof articulation asset from IsaacLab.
            num_envs: Number of parallel environments.
            device: Torch device string (e.g. "cuda:0").
        """
        self._device = device
        self._num_envs = num_envs

        self.joint_indices.resolve_from_articulation(robot)
        self.fsm.initialize(num_envs, device, self.cfg)
        self.balance_controller = BalanceFeedbackController(self.cfg.balance, self.joint_indices)

        self._default_full_pos = robot.data.default_joint_pos.clone()
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Whether the controller has been initialized."""
        return self._initialized

    @property
    def num_envs(self) -> int:
        """Number of managed environments."""
        return self._num_envs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state for specified environments.

        Args:
            env_ids: 1D tensor of environment indices. If None, reset all.
        """
        if not self._initialized:
            return
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.fsm.reset_envs(env_ids)
        self.pd_controller.reset(env_ids)
        if self.balance_controller is not None:
            self.balance_controller.reset(env_ids, self._num_envs, self._device)

    def step(
        self,
        state: SimbiconStateData,
        dt: float,
    ) -> torch.Tensor:
        """Advance the controller by one time step.

        Reads robot state, updates FSM, computes balance feedback, generates
        target poses, and returns full 29-DOF joint position targets.

        Args:
            state: Current robot state data.
            dt: Simulation time step (seconds).

        Returns:
            Full 29-DOF target joint positions, shape (num_envs, 29).
        """
        assert self._initialized, "Controller not initialized. Call initialize() first."

        self.fsm.advance_timer(dt)

        left_contact = state.left_foot_contact.bool()
        right_contact = state.right_foot_contact.bool()
        should_transition = self.fsm.check_transitions(dt, left_contact, right_contact)
        env_ids = torch.where(should_transition)[0]
        self.fsm.transition(env_ids)

        controllable_targets = self.fsm.get_interpolated_pose(self.joint_indices)

        if self.balance_controller is not None:
            controllable_targets = self._apply_balance_feedback(state, controllable_targets, dt)

        full_targets = self._build_full_targets(controllable_targets)
        return full_targets

    def compute_actions(
        self,
        state: SimbiconStateData,
        dt: float,
    ) -> torch.Tensor:
        """Compute environment-compatible actions from controller output.

        Converts the SIMBICON target positions to the action space expected
        by the ManagerBasedRLEnv with JointPositionActionCfg:

            action = (target_pos - default_pos) / scale

        Args:
            state: Current robot state data.
            dt: Simulation time step (seconds).

        Returns:
            Action tensor, shape (num_envs, 29).
        """
        assert self._default_full_pos is not None
        targets = self.step(state, dt)
        actions = (targets - self._default_full_pos) / self.cfg.action_scale
        return actions

    def compute_actions_from_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert pre-computed targets to environment actions.

        Args:
            targets: Full 29-DOF target positions, shape (num_envs, 29).

        Returns:
            Action tensor, shape (num_envs, 29).
        """
        assert self._default_full_pos is not None
        return (targets - self._default_full_pos) / self.cfg.action_scale

    def get_debug_info(self) -> dict[str, Any]:
        """Collect debug information from all controller components.

        Returns:
            Dictionary with current FSM state, phase, balance feedback
            data, and other diagnostic information.
        """
        assert self._initialized
        info: dict[str, Any] = {
            "fsm_state": self.fsm.current_state.clone() if self.fsm.current_state is not None else None,
            "fsm_state_names": [
                STATE_NAMES[SimbiconState(s.item())]
                for s in (self.fsm.current_state if self.fsm.current_state is not None else [])
            ],
            "state_timer": self.fsm.state_timer.clone() if self.fsm.state_timer is not None else None,
            "step_count": self.fsm.step_count.clone() if self.fsm.step_count is not None else None,
            "support_foot": self.fsm.get_support_foot().clone() if self.fsm.current_state is not None else None,
            "swing_foot": self.fsm.get_swing_foot().clone() if self.fsm.current_state is not None else None,
            "phase": self.fsm.get_phase().clone() if self.fsm.current_state is not None else None,
        }
        if self.balance_controller is not None:
            info.update(self.balance_controller.get_debug_info())
        return info

    def _apply_balance_feedback(
        self,
        state: SimbiconStateData,
        targets: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply COM balance feedback to swing hip pitch.

        Args:
            state: Current robot state data.
            targets: Interpolated target poses, shape (num_envs, 15).
            dt: Time step.

        Returns:
            Modified target poses with balance feedback applied.
        """
        assert self.balance_controller is not None
        assert self.joint_indices.left_ankle_body_id >= 0
        assert self.joint_indices.right_ankle_body_id >= 0

        com_pos = state.root_pos
        com_vel = state.root_lin_vel

        swing_foot = self.fsm.get_swing_foot()

        support_foot = self.fsm.get_support_foot()
        left_is_support = support_foot == 0

        left_ankle_pos = state.body_pos[:, self.joint_indices.left_ankle_body_id, :]
        right_ankle_pos = state.body_pos[:, self.joint_indices.right_ankle_body_id, :]

        support_ankle_pos = torch.where(
            left_is_support.unsqueeze(1),
            left_ankle_pos,
            right_ankle_pos,
        )

        default_swing_hip_pitch = self._extract_swing_hip_pitch(targets, swing_foot)
        default_swing_hip_roll = self._extract_swing_hip_roll(targets, swing_foot)

        corrected_hip_pitch, com_displacement, lateral_correction = self.balance_controller.compute_feedback(
            com_pos=com_pos,
            com_vel=com_vel,
            support_ankle_pos=support_ankle_pos,
            swing_foot=swing_foot,
            default_swing_hip_pitch=default_swing_hip_pitch,
            dt=dt,
        )

        targets = self._set_swing_hip_pitch(targets, corrected_hip_pitch, swing_foot)
        targets = self._set_swing_hip_roll(targets, default_swing_hip_roll - lateral_correction, swing_foot)
        return targets

    def _extract_swing_hip_pitch(
        self,
        targets: torch.Tensor,
        swing_foot: torch.Tensor,
    ) -> torch.Tensor:
        """Extract swing hip pitch from target poses.

        Args:
            targets: Target poses, shape (num_envs, 15).
            swing_foot: Swing foot indicator, shape (num_envs,).

        Returns:
            Swing hip pitch angle per environment, shape (num_envs,).
        """
        left_hip_pitch = targets[:, 0]
        right_hip_pitch = targets[:, 6]
        swing_hip = torch.where(swing_foot == 0, left_hip_pitch, right_hip_pitch)
        swing_hip = torch.where(swing_foot == 2, torch.zeros_like(swing_hip), swing_hip)
        return swing_hip

    def _set_swing_hip_pitch(
        self,
        targets: torch.Tensor,
        corrected: torch.Tensor,
        swing_foot: torch.Tensor,
    ) -> torch.Tensor:
        """Set the swing hip pitch in target poses after balance correction.

        Args:
            targets: Target poses, shape (num_envs, 15).
            corrected: Corrected hip pitch angles, shape (num_envs,).
            swing_foot: Swing foot indicator, shape (num_envs,).

        Returns:
            Modified target poses, shape (num_envs, 15).
        """
        result = targets.clone()
        left_is_swing = swing_foot == 0
        right_is_swing = swing_foot == 1

        result[left_is_swing, 0] = corrected[left_is_swing]
        result[right_is_swing, 6] = corrected[right_is_swing]
        return result

    def _extract_swing_hip_roll(
        self,
        targets: torch.Tensor,
        swing_foot: torch.Tensor,
    ) -> torch.Tensor:
        """Extract swing hip roll from target poses.

        Args:
            targets: Target poses, shape (num_envs, 15).
            swing_foot: Swing foot indicator, shape (num_envs,).

        Returns:
            Swing hip roll angle per environment, shape (num_envs,).
        """
        left_hip_roll = targets[:, 1]
        right_hip_roll = targets[:, 7]
        swing_roll = torch.where(swing_foot == 0, left_hip_roll, right_hip_roll)
        swing_roll = torch.where(swing_foot == 2, torch.zeros_like(swing_roll), swing_roll)
        return swing_roll

    def _set_swing_hip_roll(
        self,
        targets: torch.Tensor,
        corrected: torch.Tensor,
        swing_foot: torch.Tensor,
    ) -> torch.Tensor:
        """Set the swing hip roll in target poses after lateral balance correction.

        Args:
            targets: Target poses, shape (num_envs, 15).
            corrected: Corrected hip roll angles, shape (num_envs,).
            swing_foot: Swing foot indicator, shape (num_envs,).

        Returns:
            Modified target poses, shape (num_envs, 15).
        """
        result = targets.clone()
        left_is_swing = swing_foot == 0
        right_is_swing = swing_foot == 1

        result[left_is_swing, 1] = corrected[left_is_swing]
        result[right_is_swing, 7] = corrected[right_is_swing]
        return result

    def set_gait_params(
        self,
        *,
        hl: float,
        ls: float,
        lswb: float,
        lforward: float,
    ) -> None:
        """Hot-update gait parameters for parameter search.

        Updates the FSM pose definitions and state durations. The new
        parameters take effect on the next FSM state entry (no need to
        rebuild the controller). Call before ``reset()`` and ``env.reset()``.

        Args:
            hl: Foot clearance parameter (integer in [20, 60]).
            ls: Step length parameter (integer in [40, 95]).
            lswb: Lateral sway parameter (integer in [15, 40]).
            lforward: Forward distance parameter (integer in [5, 40]).
        """
        self.fsm.update_pose_from_params(hl=hl, ls=ls, lswb=lswb, lforward=lforward)

    def sample_pose_from_state_phase(
        self,
        state_name: str,
        phase: float,
    ) -> torch.Tensor:
        """Return a (29,) target joint pose for the given FSM state and phase.

        Static pose generation — no simulation rollout required. Interpolates
        between the state's start/end poses at the given phase and fills
        non-controllable joints (waist, arms) with their default positions.

        Args:
            state_name: FSM state name (e.g. "STEP_RIGHT_WITH_LEFT_FRONT").
            phase: Interpolation phase in [0, 1]. 0=start pose, 1=end pose.

        Returns:
            Full 29-DOF target joint position vector, shape (29,).

        Raises:
            ValueError: If state_name is not a valid FSM state.
        """
        state_map = {name: s for s, name in STATE_NAMES.items()}
        if state_name not in state_map:
            raise ValueError(f"Unknown FSM state '{state_name}'. " f"Valid states: {list(state_map.keys())}")
        state_enum = state_map[state_name]
        pose_def = self.fsm.pose_defs[state_enum]

        controllable = torch.zeros(NUM_CONTROLLABLE_JOINTS, dtype=torch.float32)
        for j, abstract_name in enumerate(ABSTRACT_JOINT_NAMES):
            start_val = pose_def.start_pose.get(abstract_name, 0.0)
            end_val = pose_def.end_pose.get(abstract_name, 0.0)
            controllable[j] = start_val + phase * (end_val - start_val)

        controllable_2d = controllable.unsqueeze(0)
        full = self._build_full_targets(controllable_2d)
        return full.squeeze(0)

    def _build_full_targets(self, controllable_targets: torch.Tensor) -> torch.Tensor:
        """Expand 12-DOF controllable targets to full 29-DOF targets.

        Non-controllable joints (waist and arms) are set to their default
        positions from the robot configuration.

        Args:
            controllable_targets: Targets for 12 controllable leg joints,
                shape (num_envs, 12).

        Returns:
            Full 29-DOF targets, shape (num_envs, 29).
        """
        assert self._default_full_pos is not None
        full = self._default_full_pos.clone()

        for j, joint_id in enumerate(self.joint_indices.controllable_joint_ids):
            full[:, joint_id] = controllable_targets[:, j]

        return full
