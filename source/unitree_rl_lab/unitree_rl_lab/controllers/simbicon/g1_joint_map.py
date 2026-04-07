"""G1-29dof joint and body name mappings for the SIMBICON controller.

This module defines the mapping between abstract SIMBICON gait joint names
and the actual G1-29dof joint names in the IsaacLab articulation. It also
provides body name constants for foot contact detection and ankle position
queries.

G1-29dof joint layout (SDK order):
    Indices 0-5:   Left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    Indices 6-11:  Right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    Indices 12-14: Waist     (yaw, roll, pitch)
    Indices 15-21: Left arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    Indices 22-28: Right arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

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
NUM_LEG_JOINTS: int = 12
NUM_WAIST_JOINTS: int = 3
NUM_ARM_JOINTS: int = 14

# ---------------------------------------------------------------------------
# Abstract SIMBICON joint names (12 controllable DOFs: legs only)
# Waist and arm joints are held at their default positions.
# ---------------------------------------------------------------------------
ABSTRACT_JOINT_NAMES: list[str] = [
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

NUM_CONTROLLABLE_JOINTS: int = len(ABSTRACT_JOINT_NAMES)

# ---------------------------------------------------------------------------
# Mapping: abstract name -> G1 SDK joint name
# ---------------------------------------------------------------------------
ABSTRACT_TO_G1_MAP: dict[str, str] = {
    "left_hip_pitch": "left_hip_pitch_joint",
    "left_hip_roll": "left_hip_roll_joint",
    "left_hip_yaw": "left_hip_yaw_joint",
    "left_knee": "left_knee_joint",
    "left_ankle_pitch": "left_ankle_pitch_joint",
    "left_ankle_roll": "left_ankle_roll_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_yaw": "right_hip_yaw_joint",
    "right_knee": "right_knee_joint",
    "right_ankle_pitch": "right_ankle_pitch_joint",
    "right_ankle_roll": "right_ankle_roll_joint",
}

# ---------------------------------------------------------------------------
# Body name patterns for foot contact detection and ankle position queries.
# In the G1 USD model, the foot bodies are named after the last joint in the
# leg kinematic chain (ankle_roll).
# ---------------------------------------------------------------------------
LEFT_FOOT_BODY_PATTERN: str = ".*left_ankle_roll.*"
RIGHT_FOOT_BODY_PATTERN: str = ".*right_ankle_roll.*"

# ---------------------------------------------------------------------------
# Default joint positions from UNITREE_G1_29DOF_CFG init_state.
# Used for non-controllable joints (arms) to maintain their default pose.
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


@dataclass
class G1JointIndices:
    """Resolved joint and body indices for the G1-29dof robot.

    Populated by calling :meth:`resolve_from_articulation` after the
    simulation environment is created.

    Attributes:
        controllable_joint_ids: Tensor indices of the 15 SIMBICON-controllable
            joints within the full 29-DOF articulation.
        left_foot_body_id: Body index for the left foot (ankle_roll body).
        right_foot_body_id: Body index for the right foot (ankle_roll body).
        left_ankle_body_id: Body index for the left ankle (same as foot).
        right_ankle_body_id: Body index for the right ankle (same as foot).
        left_hip_pitch_id: Joint index for left hip pitch.
        right_hip_pitch_id: Joint index for right hip pitch.
        left_foot_sensor_id: Body index within the contact sensor for left foot.
        right_foot_sensor_id: Body index within the contact sensor for right foot.
    """

    controllable_joint_ids: list[int] = field(default_factory=list)
    left_foot_body_id: int = -1
    right_foot_body_id: int = -1
    left_ankle_body_id: int = -1
    right_ankle_body_id: int = -1
    left_hip_pitch_id: int = -1
    right_hip_pitch_id: int = -1
    all_joint_names: list[str] = field(default_factory=list)
    left_foot_sensor_id: int = -1
    right_foot_sensor_id: int = -1

    def resolve_from_articulation(self, robot: Articulation) -> None:
        """Resolve joint and body indices from a live IsaacLab articulation.

        Args:
            robot: The G1-29dof articulation asset.
        """
        self.all_joint_names = list(robot.joint_names)

        for i, name in enumerate(ABSTRACT_JOINT_NAMES):
            g1_name = ABSTRACT_TO_G1_MAP[name]
            try:
                joint_id = robot.joint_names.index(g1_name)
                self.controllable_joint_ids.append(joint_id)
            except ValueError:
                raise RuntimeError(f"Joint '{g1_name}' not found in articulation joint names.")

        left_foot_ids, left_foot_names = robot.find_bodies(LEFT_FOOT_BODY_PATTERN)
        right_foot_ids, right_foot_names = robot.find_bodies(RIGHT_FOOT_BODY_PATTERN)

        if len(left_foot_ids) > 0:
            self.left_foot_body_id = left_foot_ids[0]
            self.left_ankle_body_id = left_foot_ids[0]
        else:
            raise RuntimeError("Could not find left foot body matching pattern: " + LEFT_FOOT_BODY_PATTERN)

        if len(right_foot_ids) > 0:
            self.right_foot_body_id = right_foot_ids[0]
            self.right_ankle_body_id = right_foot_ids[0]
        else:
            raise RuntimeError("Could not find right foot body matching pattern: " + RIGHT_FOOT_BODY_PATTERN)

        self.left_hip_pitch_id = robot.joint_names.index("left_hip_pitch_joint")
        self.right_hip_pitch_id = robot.joint_names.index("right_hip_pitch_joint")

    def resolve_from_contact_sensor(self, sensor_body_names: list[str]) -> None:
        """Resolve sensor-local body indices by matching body names.

        The contact sensor's body ordering may differ from the articulation's
        body ordering. This method maps foot body names to sensor indices.

        Args:
            sensor_body_names: List of body names from the contact sensor.
        """
        for i, name in enumerate(sensor_body_names):
            if "left_ankle_roll" in name:
                self.left_foot_sensor_id = i
            if "right_ankle_roll" in name:
                self.right_foot_sensor_id = i

    def get_default_joint_pos(self, device: str) -> list[float]:
        """Return the full 29-DOF default joint position vector.

        Args:
            device: Torch device string (unused, kept for API consistency).

        Returns:
            List of 29 default joint positions in SDK order.
        """
        defaults = [0.0] * NUM_TOTAL_JOINTS
        for name, val in DEFAULT_JOINT_POSITIONS.items():
            if name in self.all_joint_names:
                idx = self.all_joint_names.index(name)
                defaults[idx] = val
        return defaults
