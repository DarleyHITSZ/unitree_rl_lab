"""Configuration classes for the SIMBICON-style bipedal gait controller.

Provides :class:`SimbiconCfg` which holds all tunable parameters for the
controller, including FSM timing, PD gains, balance feedback gains, and
gait parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PDGainsCfg:
    """PD controller gains for a group of joints.

    Attributes:
        kp: Proportional gain (Nm/rad).
        kd: Derivative gain (Nm*s/rad).
    """

    kp: float = 100.0
    kd: float = 5.0


@dataclass
class BalanceFeedbackCfg:
    """COM-based balance feedback configuration.

    Implements the SIMBICON core feedback:
        theta_d = theta_d0 + c_d * d + c_v * v

    where d is COM displacement relative to support ankle and v is COM
    horizontal velocity.

    Attributes:
        cd: Gain for COM displacement feedback.
        cv: Gain for COM velocity feedback.
        com_filter_alpha: Exponential moving average filter for COM velocity.
    """

    cd: float = 0.1
    cv: float = 0.02
    com_filter_alpha: float = 0.3


@dataclass
class FSMStateCfg:
    """Configuration for a single FSM state.

    Attributes:
        duration: Nominal duration of the state in seconds.
        support_foot: Which foot is the support foot ("left", "right", or "both").
        swing_foot: Which foot is the swing foot ("left", "right", or "none").
        use_contact_trigger: Whether to allow early transition on swing-foot contact.
        contact_force_threshold: Minimum contact force (N) to trigger contact-based transition.
    """

    duration: float = 0.8
    support_foot: str = "left"
    swing_foot: str = "right"
    use_contact_trigger: bool = True
    contact_force_threshold: float = 10.0


@dataclass
class SimbiconCfg:
    """Top-level configuration for the SIMBICON gait controller.

    Attributes:
        continuous_walking: If True, the controller cycles between step states
            indefinitely. If False, it walks a fixed number of steps then recovers.
        max_steps: Maximum number of step cycles before recovery (ignored if
            continuous_walking is True).
        fsm_states: Per-state FSM configurations, keyed by state name.
        pd_gains: PD gains keyed by joint group name.
        balance: COM balance feedback configuration.
        hip_pitch_gains: Separate PD gains for hip pitch joints (higher torque).
        knee_gains: Separate PD gains for knee joints.
        ankle_gains: Separate PD gains for ankle joints.
        action_scale: Scale factor to convert from the environment's action space
            to joint position targets. Matches JointPositionActionCfg.scale.
    """

    continuous_walking: bool = True
    max_steps: int = 10

    fsm_states: dict[str, FSMStateCfg] = field(default_factory=dict)
    pd_gains: dict[str, PDGainsCfg] = field(default_factory=dict)
    balance: BalanceFeedbackCfg = field(default_factory=BalanceFeedbackCfg)
    hip_pitch_gains: PDGainsCfg = field(default_factory=lambda: PDGainsCfg(kp=120.0, kd=6.0))
    knee_gains: PDGainsCfg = field(default_factory=lambda: PDGainsCfg(kp=150.0, kd=8.0))
    ankle_gains: PDGainsCfg = field(default_factory=lambda: PDGainsCfg(kp=40.0, kd=2.0))
    action_scale: float = 0.25

    def __post_init__(self) -> None:
        if not self.fsm_states:
            self.fsm_states = {
                "STANCE_CROUCH": FSMStateCfg(
                    duration=0.5,
                    support_foot="both",
                    swing_foot="none",
                    use_contact_trigger=False,
                ),
                "START_STEP_LEFT": FSMStateCfg(
                    duration=1.2,
                    support_foot="right",
                    swing_foot="left",
                    use_contact_trigger=True,
                    contact_force_threshold=15.0,
                ),
                "STEP_RIGHT_WITH_LEFT_FRONT": FSMStateCfg(
                    duration=0.8,
                    support_foot="left",
                    swing_foot="right",
                    use_contact_trigger=True,
                    contact_force_threshold=15.0,
                ),
                "STEP_LEFT_WITH_RIGHT_FRONT": FSMStateCfg(
                    duration=0.8,
                    support_foot="right",
                    swing_foot="left",
                    use_contact_trigger=True,
                    contact_force_threshold=15.0,
                ),
                "RECOVER_CROUCH": FSMStateCfg(
                    duration=0.8,
                    support_foot="both",
                    swing_foot="none",
                    use_contact_trigger=False,
                ),
            }
        if not self.pd_gains:
            self.pd_gains = {
                "left_hip_pitch": self.hip_pitch_gains,
                "right_hip_pitch": self.hip_pitch_gains,
                "left_hip_roll": PDGainsCfg(kp=100.0, kd=5.0),
                "right_hip_roll": PDGainsCfg(kp=100.0, kd=5.0),
                "left_hip_yaw": PDGainsCfg(kp=100.0, kd=5.0),
                "right_hip_yaw": PDGainsCfg(kp=100.0, kd=5.0),
                "left_knee": self.knee_gains,
                "right_knee": self.knee_gains,
                "left_ankle_pitch": self.ankle_gains,
                "right_ankle_pitch": self.ankle_gains,
                "left_ankle_roll": PDGainsCfg(kp=40.0, kd=2.0),
                "right_ankle_roll": PDGainsCfg(kp=40.0, kd=2.0),
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict for logging."""
        import dataclasses

        def _convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _convert(self)
