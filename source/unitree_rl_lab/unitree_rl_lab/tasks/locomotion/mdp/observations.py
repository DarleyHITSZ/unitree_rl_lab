from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def detect_terrain_from_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    slope_threshold_deg: float = 5.0,
    min_valid_rays: int = 10,
) -> torch.Tensor:
    """Detect if terrain is sloped using height scanner data.

    Uses front-back height difference across 1.6m grid to detect slopes.
    Returns True (1.0) for slope >= threshold, False (0.0) otherwise.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the height scanner sensor.
        slope_threshold_deg: Minimum slope angle (degrees) to classify as slope.
        min_valid_rays: Minimum number of valid rays required.

    Returns:
        Boolean tensor indicating slope detection for each environment.
    """
    height_scanner = env.scene.sensors[sensor_cfg.name]
    heights = height_scanner.data.ray_hits_w[:, :, 2]

    valid_rays_mask = ~torch.isinf(heights)
    num_valid_rays = torch.sum(valid_rays_mask, dim=1)

    front_half = heights[:, :heights.shape[1] // 2]
    back_half = heights[:, heights.shape[1] // 2:]

    front_valid = ~torch.isinf(front_half)
    back_valid = ~torch.isinf(back_half)

    front_heights = torch.where(front_valid, front_half, torch.zeros_like(front_half))
    back_heights = torch.where(back_valid, back_half, torch.zeros_like(back_half))

    front_mean = torch.sum(front_heights, dim=1) / (torch.sum(front_valid, dim=1).clamp(min=1).float())
    back_mean = torch.sum(back_heights, dim=1) / (torch.sum(back_valid, dim=1).clamp(min=1).float())

    height_diff = torch.abs(front_mean - back_mean)
    distance = 1.6

    slope_angle = torch.atan(height_diff / distance)
    slope_deg = torch.rad2deg(slope_angle)

    is_slope = slope_deg >= slope_threshold_deg
    is_slope = is_slope & (num_valid_rays >= min_valid_rays)

    return is_slope.float()


def terrain_type_obs_onehot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    slope_threshold_deg: float = 5.0,
    min_valid_rays: int = 10,
) -> torch.Tensor:
    """Generate one-hot terrain type observation.

    Returns [flat_prob, slope_prob] for each environment.
    Classification: Flat if slope < 5°, Slope if slope >= 5°.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the height scanner sensor.
        slope_threshold_deg: Minimum slope angle (degrees) to classify as slope.
        min_valid_rays: Minimum number of valid rays required.

    Returns:
        One-hot tensor of shape (num_envs, 2) where first column is flat_prob,
        second column is slope_prob.
    """
    is_slope = detect_terrain_from_height_scan(env, sensor_cfg, slope_threshold_deg, min_valid_rays)
    
    flat_prob = 1.0 - is_slope
    slope_prob = is_slope
    
    return torch.stack([flat_prob, slope_prob], dim=1)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


__all__ = ["detect_terrain_from_height_scan", "terrain_type_obs_onehot", "gait_phase"]
