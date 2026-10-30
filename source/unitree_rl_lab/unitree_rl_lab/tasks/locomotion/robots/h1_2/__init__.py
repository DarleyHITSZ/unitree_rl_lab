"""Register H1_2 velocity tracking tasks."""

import gymnasium as gym

gym.register(
    id="Unitree-H1-2-Velocity",
    entry_point=f"{__name__}.env:H12Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:H12PPORunnerCfg",
    },
)
