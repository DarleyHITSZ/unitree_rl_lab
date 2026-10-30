"""IsaacLab adapters for the official mjlab H1_2 task."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import (
    JointPositionAction,
    JointPositionActionCfg,
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass


class H12Env(ManagerBasedRLEnv):
    """Keep official signed rewards and initialize encoder calibration state."""

    def load_managers(self):
        """Create adapter buffers before managers inspect observation terms."""
        self.joint_encoder_bias = torch.zeros_like(self.scene["robot"].data.default_joint_pos)
        self.self_contact_sensor_names = [name for name in self.scene.sensors if name.startswith("self_contact_")]
        self._previous_policy_action = torch.zeros_like(self.joint_encoder_bias)
        self._older_policy_action = torch.zeros_like(self.joint_encoder_bias)
        super().load_managers()

    def step(self, action: torch.Tensor):
        """Log the upstream mean absolute second action difference."""
        previous = self._previous_policy_action
        action_acc = (action - 2 * previous + self._older_policy_action).abs().mean(dim=-1)
        observations, rewards, terminated, truncated, extras = super().step(action)
        self._older_policy_action = previous.clone()
        self._previous_policy_action = action.to(self.device).clone()
        reset = terminated | truncated
        self._older_policy_action[reset] = 0.0
        self._previous_policy_action[reset] = 0.0
        extras.setdefault("log", {})["Metrics/mean_action_acc"] = action_acc.mean()
        return observations, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset diagnostic action history with the environment managers."""
        super()._reset_idx(env_ids)
        for name in ("_previous_policy_action", "_older_policy_action"):
            value = getattr(self, name).clone()
            value[env_ids] = 0.0
            setattr(self, name, value)


class EncoderPositionAction(JointPositionAction):
    """Apply the encoder bias to targets using mjlab's subtraction convention."""

    def apply_actions(self):
        """Write calibrated targets while keeping policy actions unchanged."""
        target = self.processed_actions - self._env.joint_encoder_bias[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class EncoderPositionActionCfg(JointPositionActionCfg):
    """Position actions with the upstream startup encoder bias."""

    class_type = EncoderPositionAction


class H12VelocityCommand(UniformVelocityCommand):
    """Use the official three axis command deadband before heading feedback."""

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample commands and zero commands below the official 0.1 threshold."""
        super()._resample_command(env_ids)
        command = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = command * (command.norm(dim=-1) > 0.1).unsqueeze(-1)


@configclass
class H12VelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform commands matching the official H1_2 sampler."""

    class_type = H12VelocityCommand


def randomize_encoder_bias(env: H12Env, env_ids: torch.Tensor | None, bias_range: tuple[float, float]):
    """Sample one fixed encoder calibration error per joint and environment."""
    if env_ids is None:
        env.joint_encoder_bias.uniform_(*bias_range)
    else:
        env.joint_encoder_bias[env_ids] = torch.empty_like(env.joint_encoder_bias[env_ids]).uniform_(*bias_range)


class SharedFootFriction(ManagerTermBase):
    """Assign a single sampled friction to every foot collider in an environment."""

    def __init__(self, cfg: EventTermCfg, env: H12Env):
        """Resolve PhysX collider slots independently of body indices."""
        super().__init__(cfg, env)
        self.asset = env.scene["robot"]
        body_ids = cfg.params["asset_cfg"].body_ids
        self.shape_ids = []
        shape_offset = 0
        for body_id, path in enumerate(self.asset.root_physx_view.link_paths[0]):
            view = self.asset._physics_sim_view.create_rigid_body_view(path)
            count = view.max_shapes
            if body_id in body_ids:
                self.shape_ids.extend(range(shape_offset, shape_offset + count))
            shape_offset += count
        if shape_offset != self.asset.root_physx_view.max_shapes:
            raise RuntimeError("Unable to resolve H1_2 foot collider material indices")

    def __call__(
        self, env: H12Env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, ranges: tuple[float, float]
    ):
        """Share static and dynamic friction samples across both feet."""
        del asset_cfg
        ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
        properties = self.asset.root_physx_view.get_material_properties().clone()
        values = torch.empty(len(ids), device="cpu").uniform_(*ranges)
        properties[ids[:, None], self.shape_ids, :2] = values[:, None, None]
        properties[ids[:, None], self.shape_ids, 2] = 0.0
        self.asset.root_physx_view.set_material_properties(properties, ids)
