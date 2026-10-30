"""Official mjlab H1_2 PPO parameters adapted to the installed RSL-RL API."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class H12PPORunnerCfg(BasePPORunnerCfg):
    """Train normalized asymmetric feedforward actor and critic networks."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10001
    save_interval = 100
    experiment_name = "h1_2"
    empirical_normalization = None
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
