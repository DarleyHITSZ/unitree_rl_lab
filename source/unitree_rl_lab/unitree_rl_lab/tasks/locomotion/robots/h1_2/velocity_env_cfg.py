"""Official mjlab 27-joint H1_2 flat training scheme on IsaacLab."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.h1_2 import (
    H1_2_ACTION_SCALE,
    H1_2_COLLISION_BODIES,
    H1_2_EXCLUDED_PAIRS,
    H1_2_JOINT_NAMES,
    UNITREE_H1_2_CFG,
)

from . import mdp as h1_mdp
from .env import (
    EncoderPositionActionCfg,
    H12VelocityCommandCfg,
    SharedFootFriction,
    randomize_encoder_bias,
)
from .terms import FilteredContactSensor, variable_posture

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
JOINTS = SceneEntityCfg("robot", joint_names=H1_2_JOINT_NAMES, preserve_order=True)
FEET = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
TORSO = SceneEntityCfg("robot", body_names="torso_link")


class RobotSceneCfg(InteractiveSceneCfg):
    """Official handless H1_2 model with separate ground and self contact reports."""

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        spawn=sim_utils.CuboidCfg(
            size=(2000.0, 2000.0, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )
    robot: ArticulationCfg = UNITREE_H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    left_foot_contact = ContactSensorCfg(
        class_type=FilteredContactSensor,
        track_friction_forces=True,
        max_contact_data_count_per_prim=32,
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/left_ankle_roll_link",
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
        force_threshold=0.0,
        history_length=4,
    )
    right_foot_contact = ContactSensorCfg(
        class_type=FilteredContactSensor,
        track_friction_forces=True,
        max_contact_data_count_per_prim=32,
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/right_ankle_roll_link",
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
        force_threshold=0.0,
        history_length=4,
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


for index, body in enumerate(H1_2_COLLISION_BODIES):
    targets = [
        f"{{ENV_REGEX_NS}}/Robot/pelvis/{other}"
        for other in H1_2_COLLISION_BODIES
        if other != body and frozenset((body, other)) not in H1_2_EXCLUDED_PAIRS
    ]
    setattr(
        RobotSceneCfg,
        f"self_contact_{index:02d}",
        ContactSensorCfg(
            class_type=FilteredContactSensor,
            track_friction_forces=True,
            max_contact_data_count_per_prim=32,
            prim_path=f"{{ENV_REGEX_NS}}/Robot/pelvis/{body}",
            filter_prim_paths_expr=targets,
            history_length=4,
        ),
    )
RobotSceneCfg = configclass(RobotSceneCfg)


@configclass
class PolicyCfg(ObsGroup):
    """Ninety two inputs in the official actor observation order."""

    base_ang_vel = ObsTerm(func=h1_mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=h1_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    phase = ObsTerm(func=h1_mdp.phase, params={"period": 0.6, "command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(
        func=h1_mdp.joint_pos_rel, params={"asset_cfg": JOINTS}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": JOINTS}, noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        """Use one frame and retain the official unscaled observation noise."""
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(PolicyCfg):
    """Append IMU linear velocity and privileged foot measurements."""

    base_lin_vel = ObsTerm(func=h1_mdp.imu_lin_vel)
    foot_height = ObsTerm(func=h1_mdp.foot_height, params={"asset_cfg": FEET})
    foot_air_time = ObsTerm(func=h1_mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"})
    foot_contact = ObsTerm(func=h1_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"})
    foot_contact_forces = ObsTerm(func=h1_mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"})

    def __post_init__(self):
        """Disable corruption in the privileged critic group."""
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    """Asymmetric actor and critic observations."""

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Per-actuator position scaling from effort limits and stiffness."""

    JointPositionAction = EncoderPositionActionCfg(
        asset_name="robot",
        joint_names=H1_2_JOINT_NAMES,
        preserve_order=True,
        scale=H1_2_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Official heading feedback, standing fraction and command timing."""

    base_velocity = H12VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=H12VelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


WALKING_STD = {
    ".*hip_yaw.*": 0.15,
    ".*hip_pitch.*": 0.5,
    ".*hip_roll.*": 0.15,
    ".*knee.*": 0.5,
    ".*ankle_pitch.*": 0.15,
    ".*ankle_roll.*": 0.1,
    ".*torso.*": 0.15,
    ".*shoulder_pitch.*": 0.15,
    ".*shoulder_roll.*": 0.1,
    ".*shoulder_yaw.*": 0.1,
    ".*elbow.*": 0.1,
    ".*wrist.*": 0.1,
}
RUNNING_STD = WALKING_STD | {
    ".*hip_yaw.*": 0.25,
    ".*hip_roll.*": 0.25,
    ".*ankle_pitch.*": 0.25,
    ".*torso.*": 0.25,
    ".*shoulder_pitch.*": 0.25,
}


@configclass
class RewardsCfg:
    """All sixteen reward terms from the official H1_2 flat task."""

    track_linear_velocity = RewTerm(
        func=h1_mdp.track_linear_velocity, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_angular_velocity = RewTerm(
        func=h1_mdp.track_angular_velocity,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    body_orientation_l2 = RewTerm(func=h1_mdp.body_orientation_l2, weight=-1.0, params={"asset_cfg": TORSO})
    pose = RewTerm(
        func=variable_posture,
        weight=1.0,
        params={
            "asset_cfg": JOINTS,
            "command_name": "base_velocity",
            "std_standing": {".*": 0.05},
            "std_walking": WALKING_STD,
            "std_running": RUNNING_STD,
            "walking_threshold": 0.1,
            "running_threshold": 1.5,
        },
    )
    body_ang_vel = RewTerm(func=h1_mdp.body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": TORSO})
    angular_momentum = RewTerm(
        func=h1_mdp.angular_momentum_penalty, weight=-0.025, params={"sensor_name": "robot/root_angmom"}
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    foot_gait = RewTerm(
        func=h1_mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "base_velocity",
            "sensor_name": "feet_ground_contact",
        },
    )
    foot_clearance = RewTerm(
        func=h1_mdp.feet_clearance,
        weight=-1.0,
        params={"target_height": 0.1, "command_name": "base_velocity", "command_threshold": 0.1, "asset_cfg": FEET},
    )
    foot_slip = RewTerm(
        func=h1_mdp.feet_slip,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": FEET,
        },
    )
    soft_landing = RewTerm(
        func=h1_mdp.soft_landing,
        weight=-1e-3,
        params={"sensor_name": "feet_ground_contact", "command_name": "base_velocity", "command_threshold": 0.1},
    )
    stand_still = RewTerm(
        func=h1_mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1, "asset_cfg": JOINTS},
    )
    self_collisions = RewTerm(
        func=h1_mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision", "force_threshold": 10.0}
    )


@configclass
class TerminationsCfg:
    """Official twenty second timeout and seventy degree tilt termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class EventCfg:
    """Official reset, pushes, foot friction, encoder bias and torso COM randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": JOINTS},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )
    foot_friction = EventTerm(func=SharedFootFriction, mode="startup", params={"asset_cfg": FEET, "ranges": (0.3, 1.6)})
    encoder_bias = EventTerm(func=randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": TORSO, "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )


@configclass
class CurriculumCfg:
    """Expand command velocity ranges after five thousand PPO rollouts."""

    command_vel = CurrTerm(
        func=h1_mdp.commands_vel,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                {"step": 120000, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
            ],
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Replace the existing H1_2 task with the official twenty millisecond scheme."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    def __post_init__(self):
        """Set official simulation timing and contact history sampling."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.spawn.physics_material
        for name in vars(self.scene):
            sensor = getattr(self.scene, name)
            if isinstance(sensor, ContactSensorCfg):
                sensor.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Official play overrides with unlimited duration and no pushes or observation noise."""

    def __post_init__(self):
        """Retain startup randomization and the official reduced play command ranges."""
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 1e9
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
