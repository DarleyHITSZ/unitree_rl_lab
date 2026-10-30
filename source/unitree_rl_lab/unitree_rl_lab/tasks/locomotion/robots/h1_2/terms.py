"""Manager adapters for the official H1_2 velocity scheme."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .mdp import command_speed, filtered_contact_times, filtered_total_force, posture_reward, resolve_joint_stds


class variable_posture(ManagerTermBase):
    """Reward the default posture with command dependent per joint tolerances."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve all joint tolerances in the selected articulation joint order."""
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self.joint_ids = asset_cfg.joint_ids
        if isinstance(self.joint_ids, slice):
            indices = list(range(len(asset.joint_names)))[self.joint_ids]
        else:
            indices = list(self.joint_ids)
        names = [asset.joint_names[index] for index in indices]
        self.std_standing = resolve_joint_stds(names, cfg.params["std_standing"], env.device)
        self.std_walking = resolve_joint_stds(names, cfg.params["std_walking"], env.device)
        self.std_running = resolve_joint_stds(names, cfg.params["std_running"], env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ):
        """Evaluate the normalized mean joint error in the current speed regime."""
        data = env.scene[asset_cfg.name].data
        return posture_reward(
            data.joint_pos[:, self.joint_ids],
            data.default_joint_pos[:, self.joint_ids],
            command_speed(env, command_name),
            self.std_standing,
            self.std_walking,
            self.std_running,
            walking_threshold,
            running_threshold,
        )


class FilteredContactSensor(ContactSensor):
    """Track filtered total forces and derive contact timers from filtered pairs."""

    def _update_buffers_impl(self, env_ids):
        """Replace unfiltered timers and retain normal plus friction force history."""
        data = self._data
        previous_times = None
        if self.cfg.track_air_time:
            previous_times = tuple(
                getattr(data, field)[env_ids].clone()
                for field in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time")
            )
            elapsed = (self._timestamp[env_ids] - self._timestamp_last_update[env_ids]).clone()
        super()._update_buffers_impl(env_ids)
        total = filtered_total_force(data.force_matrix_w, data.friction_forces_w)
        data.total_force_matrix_w = total.clone()
        history = getattr(data, "total_force_matrix_w_history", None)
        if history is None:
            history = total.new_zeros((total.shape[0], max(self.cfg.history_length, 1), *total.shape[1:]))
        else:
            history = history.clone()
        history[env_ids] = history[env_ids].roll(1, dims=1)
        history[env_ids, 0] = total[env_ids]
        data.total_force_matrix_w_history = history
        if previous_times is not None:
            values = filtered_contact_times(total[env_ids], elapsed, *previous_times)
            for field, value in zip(
                ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time"), values
            ):
                getattr(data, field)[env_ids] = value

    def reset(self, env_ids=None):
        """Clear selected total force buffers safely across inference contexts."""
        super().reset(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        for field in ("total_force_matrix_w", "total_force_matrix_w_history"):
            value = getattr(self._data, field, None)
            if value is not None:
                value = value.clone()
                value[selection] = 0.0
                setattr(self._data, field, value)
