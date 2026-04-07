"""SIMBICON finite state machine and target pose definitions.

Implements a 5-state FSM for bipedal walking:

    STANCE_CROUCH -> START_STEP_LEFT -> STEP_RIGHT_WITH_LEFT_FRONT
        <-> STEP_LEFT_WITH_RIGHT_FRONT -> RECOVER_CROUCH

Each state defines:
- Support and swing foot assignments
- A start pose and end pose (linearly interpolated during the state)
- Transition conditions (time-based and contact-based)

The pose definitions target the G1-29dof robot. For the MSLPO extension,
each state can be upgraded to hold 16 discrete poses per state.
"""

from __future__ import annotations

import enum
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .g1_joint_map import G1JointIndices
    from .simbicon_cfg import SimbiconCfg


class SimbiconState(enum.IntEnum):
    """SIMBICON FSM states."""

    STANCE_CROUCH = 0
    START_STEP_LEFT = 1
    STEP_RIGHT_WITH_LEFT_FRONT = 2
    STEP_LEFT_WITH_RIGHT_FRONT = 3
    RECOVER_CROUCH = 4


STATE_NAMES: dict[SimbiconState, str] = {
    SimbiconState.STANCE_CROUCH: "STANCE_CROUCH",
    SimbiconState.START_STEP_LEFT: "START_STEP_LEFT",
    SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT: "STEP_RIGHT_WITH_LEFT_FRONT",
    SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT: "STEP_LEFT_WITH_RIGHT_FRONT",
    SimbiconState.RECOVER_CROUCH: "RECOVER_CROUCH",
}

NUM_STATES: int = len(SimbiconState)


def _mirror_pose(pose: dict[str, float]) -> dict[str, float]:
    """Mirror a pose dictionary by swapping left/right.

    Hip_roll and ankle_roll use per-joint convention (positive = abduction/eversion
    for both sides), so they are NOT negated. Waist_roll uses a global convention
    (positive = tilt left), so it IS negated.

    Args:
        pose: Joint angle dictionary keyed by abstract joint name.

    Returns:
        Mirrored pose with left/right swapped.
    """
    mirrored: dict[str, float] = {}
    for key, val in pose.items():
        if key.startswith("left_"):
            new_key = key.replace("left_", "right_", 1)
        elif key.startswith("right_"):
            new_key = key.replace("right_", "left_", 1)
        else:
            new_key = key
        mirrored[new_key] = -val if key == "waist_roll" else val
    return mirrored


# ---------------------------------------------------------------------------
# Default pose: the crouched standing posture used as the base reference.
# ---------------------------------------------------------------------------
CROUCH_POSE: dict[str, float] = {
    "left_hip_pitch": -0.1,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.3,
    "left_ankle_pitch": -0.2,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.1,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.3,
    "right_ankle_pitch": -0.2,
    "right_ankle_roll": 0.0,
}

# ---------------------------------------------------------------------------
# START_STEP_LEFT: shift weight to right foot, swing left foot forward.
# ---------------------------------------------------------------------------
START_STEP_LEFT_START: dict[str, float] = CROUCH_POSE.copy()

START_STEP_LEFT_END: dict[str, float] = {
    "left_hip_pitch": -0.35,
    "left_hip_roll": -0.03,
    "left_hip_yaw": 0.0,
    "left_knee": 0.5,
    "left_ankle_pitch": 0.1,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.2,
    "right_hip_roll": 0.03,
    "right_hip_yaw": 0.0,
    "right_knee": 0.25,
    "right_ankle_pitch": -0.15,
    "right_ankle_roll": 0.0,
}

# ---------------------------------------------------------------------------
# STEP_RIGHT_WITH_LEFT_FRONT: left is stance (in front), right swings forward.
# ---------------------------------------------------------------------------
STEP_RIGHT_START: dict[str, float] = {
    "left_hip_pitch": -0.3,
    "left_hip_roll": 0.03,
    "left_hip_yaw": 0.0,
    "left_knee": 0.2,
    "left_ankle_pitch": -0.1,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": -0.03,
    "right_hip_yaw": 0.0,
    "right_knee": 0.55,
    "right_ankle_pitch": 0.15,
    "right_ankle_roll": 0.0,
}

STEP_RIGHT_END: dict[str, float] = {
    "left_hip_pitch": -0.1,
    "left_hip_roll": 0.03,
    "left_hip_yaw": 0.0,
    "left_knee": 0.2,
    "left_ankle_pitch": -0.1,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.45,
    "right_hip_roll": -0.03,
    "right_hip_yaw": 0.0,
    "right_knee": 0.2,
    "right_ankle_pitch": -0.05,
    "right_ankle_roll": 0.0,
}

# ---------------------------------------------------------------------------
# STEP_LEFT_WITH_RIGHT_FRONT: mirror of STEP_RIGHT_WITH_LEFT_FRONT.
# ---------------------------------------------------------------------------
STEP_LEFT_START: dict[str, float] = _mirror_pose(STEP_RIGHT_END)

STEP_LEFT_END: dict[str, float] = _mirror_pose(STEP_RIGHT_START)

# ---------------------------------------------------------------------------
# State transition map (default transitions).
# ---------------------------------------------------------------------------
DEFAULT_TRANSITIONS: dict[SimbiconState, SimbiconState] = {
    SimbiconState.STANCE_CROUCH: SimbiconState.START_STEP_LEFT,
    SimbiconState.START_STEP_LEFT: SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT,
    SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT: SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT,
    SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT: SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT,
    SimbiconState.RECOVER_CROUCH: SimbiconState.STANCE_CROUCH,
}

RECOVER_TRANSITIONS: dict[SimbiconState, SimbiconState] = {
    SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT: SimbiconState.RECOVER_CROUCH,
    SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT: SimbiconState.RECOVER_CROUCH,
}


@dataclass
class StatePoseDef:
    """Per-state pose definition with start and end poses.

    For the MSLPO extension, this can be upgraded to hold 16 discrete poses
    with a lookup method.

    Attributes:
        start_pose: Joint angles at the beginning of the state.
        end_pose: Joint angles at the end of the state.
    """

    start_pose: dict[str, float]
    end_pose: dict[str, float]


@dataclass
class SimbiconFSM:
    """Finite state machine for SIMBICON bipedal walking.

    Manages per-environment FSM state, timers, and step counters.
    Fully vectorized for batched simulation.

    Attributes:
        current_state: Current FSM state per environment, shape (num_envs,).
        state_timer: Elapsed time in current state per environment, shape (num_envs,).
        step_count: Number of completed step cycles per environment, shape (num_envs,).
        pose_defs: Per-state pose definitions.
        transitions: State transition map.
        cfg: Controller configuration reference.
    """

    pose_defs: dict[SimbiconState, StatePoseDef] = field(default_factory=dict)
    transitions: dict[SimbiconState, SimbiconState] = field(default_factory=lambda: DEFAULT_TRANSITIONS.copy())
    cfg: SimbiconCfg | None = None
    current_state: torch.Tensor | None = None
    state_timer: torch.Tensor | None = None
    step_count: torch.Tensor | None = None
    _device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.pose_defs:
            self.pose_defs = {
                SimbiconState.STANCE_CROUCH: StatePoseDef(CROUCH_POSE.copy(), CROUCH_POSE.copy()),
                SimbiconState.START_STEP_LEFT: StatePoseDef(START_STEP_LEFT_START.copy(), START_STEP_LEFT_END.copy()),
                SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT: StatePoseDef(STEP_RIGHT_START.copy(), STEP_RIGHT_END.copy()),
                SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT: StatePoseDef(STEP_LEFT_START.copy(), STEP_LEFT_END.copy()),
                SimbiconState.RECOVER_CROUCH: StatePoseDef(CROUCH_POSE.copy(), CROUCH_POSE.copy()),
            }

    def initialize(self, num_envs: int, device: str, cfg: SimbiconCfg | None = None) -> None:
        """Initialize FSM tensors for a batch of environments.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device string.
            cfg: Controller configuration (optional, used for max_steps).
        """
        self._device = device
        if cfg is not None:
            self.cfg = cfg
        self.current_state = torch.full((num_envs,), SimbiconState.STANCE_CROUCH, dtype=torch.long, device=device)
        self.state_timer = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset FSM state for specific environments (e.g. on episode reset).

        Args:
            env_ids: 1D tensor of environment indices to reset.
        """
        if self.current_state is None:
            return
        assert self.state_timer is not None
        assert self.step_count is not None
        cs = self.current_state.clone()
        st = self.state_timer.clone()
        sc = self.step_count.clone()
        cs[env_ids] = SimbiconState.STANCE_CROUCH
        st[env_ids] = 0.0
        sc[env_ids] = 0
        self.current_state = cs
        self.state_timer = st
        self.step_count = sc

    @property
    def num_envs(self) -> int:
        """Return the number of managed environments."""
        if self.current_state is None:
            return 0
        return self.current_state.shape[0]

    def get_state_durations(self) -> torch.Tensor:
        """Get the configured duration for each environment's current state.

        Returns:
            Tensor of shape (num_envs,) with state durations in seconds.
        """
        assert self.cfg is not None, "FSM not initialized with config."
        assert self.current_state is not None
        durations = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        for state in SimbiconState:
            mask = self.current_state == state
            if mask.any():
                state_name = STATE_NAMES[state]
                durations[mask] = self.cfg.fsm_states[state_name].duration
        return durations

    def get_phase(self) -> torch.Tensor:
        """Compute interpolation phase [0, 1] for each environment.

        Returns:
            Tensor of shape (num_envs,) clamped to [0, 1].
        """
        assert self.current_state is not None
        assert self.state_timer is not None
        durations = self.get_state_durations()
        phase = self.state_timer / durations.clamp(min=1e-6)
        return phase.clamp(0.0, 1.0)

    def get_support_foot(self) -> torch.Tensor:
        """Get support foot indicator per environment.

        Returns:
            Tensor of shape (num_envs,): 0=left, 1=right, 2=both.
        """
        assert self.cfg is not None
        assert self.current_state is not None
        foot_map = {"left": 0, "right": 1, "both": 2}
        result = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        for state in SimbiconState:
            mask = self.current_state == state
            if mask.any():
                state_name = STATE_NAMES[state]
                support = self.cfg.fsm_states[state_name].support_foot
                result[mask] = foot_map[support]
        return result

    def get_swing_foot(self) -> torch.Tensor:
        """Get swing foot indicator per environment.

        Returns:
            Tensor of shape (num_envs,): 0=left, 1=right, 2=none.
        """
        assert self.cfg is not None
        assert self.current_state is not None
        foot_map = {"left": 0, "right": 1, "none": 2}
        result = torch.full((self.num_envs,), 2, dtype=torch.long, device=self._device)
        for state in SimbiconState:
            mask = self.current_state == state
            if mask.any():
                state_name = STATE_NAMES[state]
                swing = self.cfg.fsm_states[state_name].swing_foot
                result[mask] = foot_map[swing]
        return result

    def check_transitions(
        self,
        dt: float,
        left_foot_contact: torch.Tensor,
        right_foot_contact: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate state transition conditions for all environments.

        Checks both time-based and contact-based triggers.

        Args:
            dt: Simulation time step (seconds).
            left_foot_contact: Boolean tensor (num_envs,) indicating left foot contact.
            right_foot_contact: Boolean tensor (num_envs,) indicating right foot contact.

        Returns:
            Boolean tensor (num_envs,) indicating which environments should transition.
        """
        assert self.current_state is not None
        assert self.state_timer is not None
        assert self.cfg is not None

        durations = self.get_state_durations()
        time_trigger = self.state_timer >= durations

        swing_foot = self.get_swing_foot()
        swing_contact = torch.where(swing_foot == 0, left_foot_contact, right_foot_contact)
        swing_contact = torch.where(swing_foot == 2, torch.zeros_like(swing_contact), swing_contact)

        contact_trigger = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        for state in SimbiconState:
            mask = (self.current_state == state) & (swing_contact)
            if mask.any():
                state_name = STATE_NAMES[state]
                state_cfg = self.cfg.fsm_states[state_name]
                if state_cfg.use_contact_trigger:
                    contact_trigger = contact_trigger | mask

        return time_trigger | contact_trigger

    def transition(self, env_ids: torch.Tensor) -> None:
        """Execute state transitions for specified environments.

        Args:
            env_ids: 1D tensor of environment indices that should transition.
        """
        assert self.current_state is not None
        assert self.state_timer is not None
        assert self.step_count is not None
        assert self.cfg is not None
        if len(env_ids) == 0:
            return

        for i in env_ids:
            current = SimbiconState(self.current_state[i].item())

            if self.cfg.continuous_walking:
                next_state = self.transitions.get(current, current)
            else:
                if current in RECOVER_TRANSITIONS and self.step_count[i] >= self.cfg.max_steps:
                    next_state = SimbiconState.RECOVER_CROUCH
                else:
                    next_state = self.transitions.get(current, current)

            self.current_state[i] = next_state
            self.state_timer[i] = 0.0

            if current in (
                SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT,
                SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT,
            ):
                self.step_count[i] += 1

    def advance_timer(self, dt: float) -> None:
        """Advance state timers by dt.

        Args:
            dt: Simulation time step (seconds).
        """
        assert self.state_timer is not None
        self.state_timer += dt

    def get_interpolated_pose(self, joint_indices: G1JointIndices) -> torch.Tensor:
        """Compute interpolated target poses for all environments.

        Linearly interpolates between start and end poses based on phase.

        Args:
            joint_indices: Resolved G1 joint indices for mapping abstract
                joint names to full articulation indices.

        Returns:
            Tensor of shape (num_envs, 12) with target angles for the
            12 controllable leg joints.
        """
        assert self.current_state is not None
        phase = self.get_phase()

        num_envs = self.num_envs
        target = torch.zeros(num_envs, 12, dtype=torch.float32, device=self._device)

        for state in SimbiconState:
            mask = self.current_state == state
            if not mask.any():
                continue

            pose_def = self.pose_defs[state]
            env_phase = phase[mask]

            for j, abstract_name in enumerate(
                [
                    "left_hip_pitch",
                    "left_hip_roll",
                    "left_hip_yaw",
                    "left_knee",
                    "left_ankle_pitch",
                    "left_ankle_roll",
                    "right_hip_pitch",
                    "right_hip_roll",
                    "right_hip_yaw",
                    "right_knee",
                    "right_ankle_pitch",
                    "right_ankle_roll",
                ]
            ):
                start_val = pose_def.start_pose.get(abstract_name, 0.0)
                end_val = pose_def.end_pose.get(abstract_name, 0.0)
                target[mask, j] = start_val + env_phase * (end_val - start_val)

        return target

    def update_pose_from_params(
        self,
        *,
        hl: float,
        ls: float,
        lswb: float,
        lforward: float,
    ) -> None:
        """Update pose defs and durations from gait parameters.

        Hot-updates the FSM pose definitions so that the next time the FSM
        enters a state, it uses the new poses. Does not require rebuilding
        the controller.

        Parameter mapping (all values are integer indices from the search space):

        * ``HL``  (foot clearance, 20-60): maps to swing knee bend amplitude.
          Higher HL -> more knee flexion -> higher foot lift.
        * ``Ls``  (step length, 40-95): maps to hip pitch angles in
          STEP_RIGHT_END / STEP_LEFT_END. Higher Ls -> larger hip pitch
          (more forward lean) -> longer step.
        * ``Lswb`` (lateral sway, 15-40): maps to swing knee bend during
          mid-step. Higher Lswb -> more knee flexion during swing.
        * ``Lforward`` (forward distance, 5-40): maps to FSM state durations.
          Higher Lforward -> longer duration per state -> slower stepping.

        Args:
            hl: Foot clearance parameter (integer in [20, 60]).
            ls: Step length parameter (integer in [40, 95]).
            lswb: Lateral sway parameter (integer in [15, 40]).
            lforward: Forward distance parameter (integer in [5, 40]).
        """
        hl_n = (hl - 20) / 40.0
        ls_n = (ls - 40) / 55.0
        lswb_n = (lswb - 15) / 25.0
        lforward_n = (lforward - 5) / 35.0

        swing_hip_pitch_start = 0.0
        swing_knee_lift = 0.4 + 0.3 * hl_n
        swing_knee_mid = 0.1 + 0.2 * lswb_n
        swing_ankle_lift = 0.05 + 0.2 * hl_n

        stance_hip_pitch_start = -0.3 - 0.2 * ls_n
        stance_hip_pitch_end = -0.1 - 0.15 * ls_n
        stance_knee = 0.2
        stance_ankle_pitch = -0.1

        swing_hip_pitch_end = -(0.3 + 0.3 * ls_n)

        hip_roll_swing = -0.03 - 0.03 * lswb_n
        hip_roll_stance = 0.03 + 0.03 * lswb_n

        step_right_start: dict[str, float] = {
            "left_hip_pitch": stance_hip_pitch_start,
            "left_hip_roll": hip_roll_stance,
            "left_hip_yaw": 0.0,
            "left_knee": stance_knee,
            "left_ankle_pitch": stance_ankle_pitch,
            "left_ankle_roll": 0.0,
            "right_hip_pitch": swing_hip_pitch_start,
            "right_hip_roll": hip_roll_swing,
            "right_hip_yaw": 0.0,
            "right_knee": swing_knee_lift,
            "right_ankle_pitch": swing_ankle_lift,
            "right_ankle_roll": 0.0,
        }

        step_right_end: dict[str, float] = {
            "left_hip_pitch": stance_hip_pitch_end,
            "left_hip_roll": hip_roll_stance,
            "left_hip_yaw": 0.0,
            "left_knee": stance_knee,
            "left_ankle_pitch": stance_ankle_pitch,
            "left_ankle_roll": 0.0,
            "right_hip_pitch": swing_hip_pitch_end,
            "right_hip_roll": hip_roll_swing,
            "right_hip_yaw": 0.0,
            "right_knee": swing_knee_mid,
            "right_ankle_pitch": -0.05,
            "right_ankle_roll": 0.0,
        }

        step_left_start: dict[str, float] = _mirror_pose(step_right_end)
        step_left_end: dict[str, float] = _mirror_pose(step_right_start)

        start_step_left_end: dict[str, float] = {
            "left_hip_pitch": -0.35 - 0.2 * ls_n,
            "left_hip_roll": -0.03 - 0.02 * lswb_n,
            "left_hip_yaw": 0.0,
            "left_knee": 0.5 + 0.3 * hl_n,
            "left_ankle_pitch": 0.1 + 0.2 * hl_n,
            "left_ankle_roll": 0.0,
            "right_hip_pitch": -0.2 - 0.2 * ls_n,
            "right_hip_roll": 0.03 + 0.02 * lswb_n,
            "right_hip_yaw": 0.0,
            "right_knee": 0.25,
            "right_ankle_pitch": -0.15,
            "right_ankle_roll": 0.0,
        }

        self.pose_defs[SimbiconState.STANCE_CROUCH] = StatePoseDef(CROUCH_POSE.copy(), CROUCH_POSE.copy())
        self.pose_defs[SimbiconState.START_STEP_LEFT] = StatePoseDef(
            CROUCH_POSE.copy(),
            start_step_left_end,
        )
        self.pose_defs[SimbiconState.STEP_RIGHT_WITH_LEFT_FRONT] = StatePoseDef(
            step_right_start,
            step_right_end,
        )
        self.pose_defs[SimbiconState.STEP_LEFT_WITH_RIGHT_FRONT] = StatePoseDef(
            step_left_start,
            step_left_end,
        )
        self.pose_defs[SimbiconState.RECOVER_CROUCH] = StatePoseDef(CROUCH_POSE.copy(), CROUCH_POSE.copy())

        if self.cfg is not None:
            base_duration = 0.4 + 0.8 * lforward_n
            self.cfg.fsm_states["STANCE_CROUCH"].duration = base_duration * 0.6
            self.cfg.fsm_states["START_STEP_LEFT"].duration = base_duration * 1.5
            self.cfg.fsm_states["STEP_RIGHT_WITH_LEFT_FRONT"].duration = base_duration
            self.cfg.fsm_states["STEP_LEFT_WITH_RIGHT_FRONT"].duration = base_duration
            self.cfg.fsm_states["RECOVER_CROUCH"].duration = base_duration * 0.8
