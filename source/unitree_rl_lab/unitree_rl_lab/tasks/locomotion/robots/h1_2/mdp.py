"""Tensor adaptations of the official mjlab H1_2 velocity task."""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SENSOR_NAMES = ("left_foot_contact", "right_foot_contact")


def _asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None):
    return env.scene["robot" if asset_cfg is None else asset_cfg.name]


def _body_ids(asset, asset_cfg, defaults):
    ids = getattr(asset_cfg, "body_ids", None)
    if ids is not None and not isinstance(ids, slice) and len(ids):
        return ids
    return [asset.body_names.index(name) for name in defaults]


def _joint_ids(asset_cfg):
    return slice(None) if asset_cfg is None else asset_cfg.joint_ids


def _quat_apply(quat, vector):
    xyz = quat[..., 1:]
    vector = torch.broadcast_to(vector, xyz.shape)
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (quat[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def _quat_inverse_apply(quat, vector):
    return _quat_apply(torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1), vector)


def command_speed(env, command_name):
    """Return planar command magnitude plus absolute yaw rate."""
    command = env.command_manager.get_command(command_name)
    return command[:, :2].norm(dim=1) + command[:, 2].abs()


def _active(env, command_name, threshold):
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    return (command_speed(env, command_name) > threshold).float()


def _foot_sensors(env, sensor_name="feet_ground_contact"):
    names = FOOT_SENSOR_NAMES if sensor_name == "feet_ground_contact" else (sensor_name,)
    return [env.scene[name] for name in names]


def _foot_times(env, sensor_name="feet_ground_contact", field="current_contact_time"):
    return torch.cat([getattr(sensor.data, field)[:, :1] for sensor in _foot_sensors(env, sensor_name)], dim=1)


def _foot_forces(env, sensor_name="feet_ground_contact"):
    return torch.stack(
        [
            getattr(sensor.data, "total_force_matrix_w", sensor.data.force_matrix_w)[:, 0, 0, :]
            for sensor in _foot_sensors(env, sensor_name)
        ],
        dim=1,
    )


def filtered_total_force(normal, friction):
    """Combine filtered normal and tangential forces, treating absent friction as zero."""
    return normal if friction is None else normal + torch.nan_to_num(friction)


def filtered_contact_times(forces, elapsed, current_air, current_contact, last_air, last_contact):
    """Advance contact timers using only the filtered contact pair forces."""
    contact = forces.norm(dim=-1).gt(0.0).any(dim=-1)
    elapsed = elapsed[:, None]
    new_last_air = torch.where((current_air > 0) & contact, current_air + elapsed, last_air)
    new_last_contact = torch.where((current_contact > 0) & ~contact, current_contact + elapsed, last_contact)
    new_current_air = torch.where(~contact, current_air + elapsed, 0.0)
    new_current_contact = torch.where(contact, current_contact + elapsed, 0.0)
    return new_current_air, new_current_contact, new_last_air, new_last_contact


def _first_contact(env, sensor_name="feet_ground_contact"):
    return torch.cat(
        [sensor.compute_first_contact(env.step_dt)[:, :1] for sensor in _foot_sensors(env, sensor_name)], dim=1
    )


def _log(env, name, value):
    env.extras.setdefault("log", {})[f"Metrics/{name}"] = value


def track_linear_velocity(env, std: float, command_name="base_velocity", asset_cfg=None):
    """Track planar commands while penalizing vertical velocity twice as much."""
    actual = _asset(env, asset_cfg).data.root_link_lin_vel_b
    command = env.command_manager.get_command(command_name)
    error = (command[:, :2] - actual[:, :2]).square().sum(dim=1) + 2 * actual[:, 2].square()
    return torch.exp(-error / std**2)


def track_angular_velocity(env, std: float, command_name="base_velocity", asset_cfg=None):
    """Track yaw rate with the official roll and pitch error factor."""
    actual = _asset(env, asset_cfg).data.root_link_ang_vel_b
    command = env.command_manager.get_command(command_name)
    error = (command[:, 2] - actual[:, 2]).square() + 0.05 * actual[:, :2].square().sum(dim=1)
    return torch.exp(-error / std**2)


def projected_gravity(env, asset_cfg=None):
    """Express unit world gravity in the pelvis link frame."""
    quat = _asset(env, asset_cfg).data.root_link_quat_w
    return _quat_inverse_apply(quat, quat.new_tensor([0.0, 0.0, -1.0]))


def body_orientation_l2(env, asset_cfg=None):
    """Measure torso projected gravity in the horizontal plane."""
    asset = _asset(env, asset_cfg)
    quat = asset.data.body_link_quat_w[:, _body_ids(asset, asset_cfg, ("torso_link",))[0]]
    gravity = _quat_inverse_apply(quat, quat.new_tensor([0.0, 0.0, -1.0]))
    return gravity[:, :2].square().sum(dim=1)


def body_angular_velocity_penalty(env, asset_cfg=None):
    """Penalize torso world roll and pitch angular velocity."""
    asset = _asset(env, asset_cfg)
    velocity = asset.data.body_link_ang_vel_w[:, _body_ids(asset, asset_cfg, ("torso_link",))[0]]
    return velocity[:, :2].square().sum(dim=1)


def imu_ang_vel(env, asset_cfg=None):
    """Reconstruct the gyro at the torso mounted IMU."""
    asset = _asset(env, asset_cfg)
    index = asset.body_names.index("torso_link")
    return _quat_inverse_apply(asset.data.body_link_quat_w[:, index], asset.data.body_link_ang_vel_w[:, index])


def imu_lin_vel(env, asset_cfg=None):
    """Reconstruct the torso local velocity at the offset IMU site."""
    asset = _asset(env, asset_cfg)
    index = asset.body_names.index("torso_link")
    quat = asset.data.body_link_quat_w[:, index]
    offset = _quat_apply(quat, quat.new_tensor([-0.04452, -0.01891, 0.27756]))
    velocity = asset.data.body_link_lin_vel_w[:, index] + torch.cross(
        asset.data.body_link_ang_vel_w[:, index], offset, dim=-1
    )
    return _quat_inverse_apply(quat, velocity)


def foot_site_state(env, asset_cfg=None):
    """Return foot site positions and velocities including rotational motion."""
    asset = _asset(env, asset_cfg)
    ids = _body_ids(asset, asset_cfg, FOOT_BODY_NAMES)
    quat = asset.data.body_link_quat_w[:, ids]
    offset = _quat_apply(quat, quat.new_tensor([0.04, 0.0, -0.04]))
    positions = asset.data.body_link_pos_w[:, ids] + offset
    velocities = asset.data.body_link_lin_vel_w[:, ids] + torch.cross(
        asset.data.body_link_ang_vel_w[:, ids], offset, dim=-1
    )
    return positions, velocities


def foot_height(env, asset_cfg=None):
    """Observe world heights of the left and right foot sites."""
    return foot_site_state(env, asset_cfg)[0][..., 2]


def foot_air_time(env, sensor_name="feet_ground_contact"):
    """Observe time since each foot left the ground."""
    return _foot_times(env, sensor_name, "current_air_time")


def foot_contact(env, sensor_name="feet_ground_contact"):
    """Observe ground contact flags independently for each foot."""
    return (_foot_times(env, sensor_name) > 0).float()


def foot_contact_forces(env, sensor_name="feet_ground_contact"):
    """Observe signed logarithms of the ground filtered foot forces."""
    forces = _foot_forces(env, sensor_name).flatten(start_dim=1)
    return torch.sign(forces) * torch.log1p(forces.abs())


def phase(env, period=0.6, command_name="base_velocity"):
    """Observe sine and cosine gait phase, zeroed for small three dimensional commands."""
    angle = ((env.episode_length_buf * env.step_dt) % period / period) * 2 * torch.pi
    values = torch.stack((angle.sin(), angle.cos()), dim=1)
    standing = env.command_manager.get_command(command_name).norm(dim=1) < 0.1
    return torch.where(standing[:, None], torch.zeros_like(values), values)


def feet_gait(
    env,
    period=0.6,
    offset=(0.0, 0.5),
    threshold=0.56,
    command_threshold=0.1,
    command_name="base_velocity",
    sensor_name="feet_ground_contact",
):
    """Reward the mean agreement between desired stance and ground contact."""
    global_phase = (env.episode_length_buf * env.step_dt / period)[:, None]
    offsets = torch.as_tensor(offset, device=global_phase.device, dtype=global_phase.dtype)[None]
    stance = (global_phase + offsets) % 1 < threshold
    contact = _foot_times(env, sensor_name) > 0
    return (stance == contact).float().mean(dim=1) * _active(env, command_name, command_threshold)


def feet_air_time(env, sensor_name="feet_ground_contact", threshold=0.4, command_name=None, command_threshold=0.1):
    """Reward single stance durations with the official triangular time reward."""
    contact_time = _foot_times(env, sensor_name)
    air_time = foot_air_time(env, sensor_name)
    contact = contact_time > 0
    mode_time = torch.where(contact, contact_time, air_time)
    single_stance = contact.float().mean(dim=1) == 0.5
    mode_time = torch.where(single_stance[:, None], mode_time, 0.0).min(dim=1).values
    return (threshold - (mode_time - threshold).abs()).clamp(min=0.0) * _active(env, command_name, command_threshold)


def feet_clearance(env, target_height, command_name=None, command_threshold=0.1, asset_cfg=None):
    """Penalize absolute site clearance error weighted by planar site speed."""
    positions, velocities = foot_site_state(env, asset_cfg)
    cost = ((positions[..., 2] - target_height).abs() * velocities[..., :2].norm(dim=-1)).sum(dim=1)
    return cost * _active(env, command_name, command_threshold)


def feet_slip(
    env, sensor_name="feet_ground_contact", command_name="base_velocity", command_threshold=0.01, asset_cfg=None
):
    """Penalize squared planar site speed during ground contact."""
    contact = foot_contact(env, sensor_name)
    speed = foot_site_state(env, asset_cfg)[1][..., :2].norm(dim=-1)
    _log(env, "slip_velocity_mean", (speed * contact).sum() / contact.sum().clamp(min=1))
    return (speed.square() * contact).sum(dim=1) * _active(env, command_name, command_threshold)


def soft_landing(env, sensor_name="feet_ground_contact", command_name=None, command_threshold=0.05):
    """Penalize the ground force magnitude on each first contact."""
    first_contact = _first_contact(env, sensor_name).float()
    impact = _foot_forces(env, sensor_name).norm(dim=-1) * first_contact
    _log(env, "landing_force_mean", impact.sum() / first_contact.sum().clamp(min=1))
    return impact.sum(dim=1) * _active(env, command_name, command_threshold)


def self_collision_cost(env, sensor_name="self_collision", force_threshold=10.0):
    """Count history substeps with any self collision above the force threshold."""
    hits = []
    for name in env.self_contact_sensor_names:
        data = env.scene[name].data
        history = getattr(data, "total_force_matrix_w_history", data.force_matrix_w_history)
        hit = history.norm(dim=-1).flatten(start_dim=2).gt(force_threshold).any(dim=2)
        hits.append(hit)
    if not hits:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.stack(hits).any(dim=0).sum(dim=1).float()


def torso_subtree_angular_momentum(env, asset_cfg=None):
    """Compute intrinsic and orbital momentum of the torso and both arm subtrees."""
    asset = _asset(env, asset_cfg)
    ids = [
        i
        for i, name in enumerate(asset.body_names)
        if name == "torso_link" or any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    data = asset.data
    position = data.body_com_pos_w[:, ids]
    velocity = data.body_com_lin_vel_w[:, ids]
    mass = data.default_mass[:, ids].to(position.device)
    center = (position * mass[..., None]).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    center_velocity = (velocity * mass[..., None]).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    orbital = torch.cross(position - center[:, None], mass[..., None] * (velocity - center_velocity[:, None]), dim=-1)
    quat = data.body_link_quat_w[:, ids]
    angular_velocity_local = _quat_inverse_apply(quat, data.body_ang_vel_w[:, ids])
    inertia = data.default_inertia[:, ids].to(position.device).reshape(*mass.shape, 3, 3)
    intrinsic_local = torch.matmul(inertia, angular_velocity_local.unsqueeze(-1)).squeeze(-1)
    intrinsic = _quat_apply(quat, intrinsic_local)
    return (intrinsic + orbital).sum(dim=1)


def angular_momentum_penalty(env, sensor_name="robot/root_angmom", asset_cfg=None):
    """Match the MJCF torso subtree angular momentum sensor squared magnitude."""
    momentum = torso_subtree_angular_momentum(env, asset_cfg)
    squared = momentum.square().sum(dim=1)
    _log(env, "angular_momentum_mean", squared.sqrt().mean())
    return squared


def joint_pos_rel(env, asset_cfg=None, biased=False):
    """Observe joint displacement, optionally including the shared encoder bias."""
    data = _asset(env, asset_cfg).data
    ids = _joint_ids(asset_cfg)
    result = data.joint_pos[:, ids] - data.default_joint_pos[:, ids]
    bias = getattr(env, "joint_encoder_bias", None)
    return result if bias is None or not biased else result + bias[:, ids]


def stand_still(env, command_name="base_velocity", command_threshold=0.1, asset_cfg=None):
    """Penalize joint displacement from default when the command is small."""
    data = _asset(env, asset_cfg).data
    ids = _joint_ids(asset_cfg)
    error = (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).square().sum(dim=1)
    return error * (command_speed(env, command_name) <= command_threshold).float()


def resolve_joint_stds(joint_names, mapping, device="cpu"):
    """Resolve regular expressions in articulation order without reordering joints."""
    result = []
    for name in joint_names:
        matches = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one posture standard deviation for {name}, got {len(matches)}")
        result.append(matches[0])
    return torch.tensor(result, device=device, dtype=torch.float32)


def posture_reward(
    position,
    default_position,
    speed,
    std_standing,
    std_walking,
    std_running,
    walking_threshold=0.5,
    running_threshold=1.5,
):
    """Evaluate the official speed dependent normalized posture exponential."""
    std = torch.where(
        (speed < walking_threshold)[:, None],
        std_standing,
        torch.where((speed < running_threshold)[:, None], std_walking, std_running),
    )
    return torch.exp(-((position - default_position).square() / std.square()).mean(dim=1))


def bad_orientation(env, limit_angle=math.radians(70), asset_cfg=None):
    """Terminate on root inclination exceeding the official angle limit."""
    gravity = projected_gravity(env, asset_cfg)
    return torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)) > limit_angle


def commands_vel(env, env_ids, command_name="base_velocity", velocity_stages=()):
    """Apply command stages strictly after their common control step boundary."""
    cfg = env.command_manager.get_term(command_name).cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if stage.get(key) is not None:
                    setattr(cfg.ranges, key, stage[key])
    return {}
