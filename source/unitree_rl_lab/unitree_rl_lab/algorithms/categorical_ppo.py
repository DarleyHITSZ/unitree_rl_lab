"""Categorical PPO for discrete action spaces.

Implements a self-contained discrete-action PPO with three components:

- :class:`CategoricalRolloutStorage`: minimal rollout buffer (no mu/sigma).
- :class:`CategoricalActorCritic`: actor (MLP -> logits -> Categorical) and
  critic (MLP -> value) sharing the same observation input.
- :class:`CategoricalPPO`: on-policy PPO trainer with clipped surrogate,
  value loss, and entropy bonus.

Designed for MSLPO Stage 2 where the action space is a discrete set of 216
pose-library indices rather than continuous joint angles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Generator

# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str = "elu",
) -> nn.Sequential:
    """Build a simple fully-connected network.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        hidden_dims: List of hidden layer widths.
        activation: Activation function name (``"elu"``, ``"relu"``, ``"tanh"``).

    Returns:
        ``nn.Sequential`` module.
    """
    act_cls = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ELU)

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# CategoricalRolloutStorage
# ---------------------------------------------------------------------------


class CategoricalRolloutStorage:
    """Minimal rollout buffer for discrete-action PPO.

    Stores observations, discrete action indices, rewards, dones, values,
    log-probabilities, and (after computation) returns and advantages.
    Deliberately does **not** store ``mu`` / ``sigma`` — those are specific
    to Gaussian policies.
    """

    class Transition:
        """Single-step data holder before flushing to storage."""

        __slots__ = ("observations", "actions", "rewards", "dones", "values", "actions_log_prob")

        def __init__(self) -> None:
            self.observations: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_dim: int,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.step = 0

        N = num_transitions_per_env
        E = num_envs

        self.observations = torch.zeros(N, E, obs_dim, device=device)
        self.actions = torch.zeros(N, E, 1, dtype=torch.long, device=device)
        self.rewards = torch.zeros(N, E, 1, device=device)
        self.dones = torch.zeros(N, E, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(N, E, 1, device=device)
        self.actions_log_prob = torch.zeros(N, E, 1, device=device)
        self.returns = torch.zeros(N, E, 1, device=device)
        self.advantages = torch.zeros(N, E, 1, device=device)

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow — call clear() first.")
        assert transition.observations is not None
        assert transition.actions is not None
        assert transition.rewards is not None
        assert transition.dones is not None
        assert transition.values is not None
        assert transition.actions_log_prob is not None
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        """Compute GAE-lambda returns and advantages (standard PPO formula)."""
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + gamma * next_values * next_not_terminal - self.values[step]
            advantage = delta + gamma * lam * next_not_terminal * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        adv_flat = self.advantages.flatten()
        adv_std = adv_flat.std()
        if adv_std > 1e-8:
            self.advantages = (self.advantages - adv_flat.mean()) / adv_std

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator:
        """Yield minibatches for PPO updates."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        obs_all = self.observations.flatten(0, 1)
        actions_all = self.actions.flatten(0, 1)
        old_log_prob_all = self.actions_log_prob.flatten(0, 1)
        returns_all = self.returns.flatten(0, 1)
        advantages_all = self.advantages.flatten(0, 1)

        for _epoch in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)[: num_mini_batches * mini_batch_size]
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                idx = indices[start:stop]
                yield (
                    obs_all[idx],
                    actions_all[idx],
                    old_log_prob_all[idx],
                    returns_all[idx],
                    advantages_all[idx],
                )


# ---------------------------------------------------------------------------
# CategoricalActorCritic
# ---------------------------------------------------------------------------


class CategoricalActorCritic(nn.Module):
    """Actor-critic with a Categorical (discrete) action distribution.

    The actor outputs logits over ``num_actions`` discrete choices; the critic
    outputs a scalar state value.  Both share the same observation input.

    Attributes:
        is_recurrent: Always ``False`` for this module.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [256, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [256, 256, 128]

        self.actor = _build_mlp(num_obs, num_actions, actor_hidden_dims, activation)
        self.critic = _build_mlp(num_obs, 1, critic_hidden_dims, activation)
        self.distribution: Categorical | None = None
        self._entropy: torch.Tensor | None = None

    def _update_distribution(self, obs: torch.Tensor) -> None:
        logits = self.actor(obs)
        self.distribution = Categorical(logits=logits)
        self._entropy = self.distribution.entropy()

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample discrete actions and cache distribution.

        Args:
            obs: ``(batch, obs_dim)`` observation tensor.

        Returns:
            Sampled action indices, shape ``(batch, 1)``.
        """
        self._update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.sample().unsqueeze(-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy action selection (argmax) for evaluation.

        Args:
            obs: ``(batch, obs_dim)`` observation tensor.

        Returns:
            Greedy action indices, shape ``(batch, 1)``.
        """
        with torch.no_grad():
            logits = self.actor(obs)
            return logits.argmax(dim=-1, keepdim=True)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            obs: ``(batch, obs_dim)`` observation tensor.

        Returns:
            State values, shape ``(batch, 1)``.
        """
        return self.critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of *actions* under the current distribution.

        Args:
            actions: ``(batch, 1)`` long tensor of action indices.

        Returns:
            Log-probabilities, shape ``(batch, 1)``.
        """
        if self.distribution is None:
            raise RuntimeError("Call act() before get_actions_log_prob().")
        return self.distribution.log_prob(actions.squeeze(-1)).unsqueeze(-1)

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the current action distribution, shape ``(batch,)``."""
        if self._entropy is None:
            raise RuntimeError("Call act() before accessing entropy.")
        return self._entropy


# ---------------------------------------------------------------------------
# CategoricalPPO
# ---------------------------------------------------------------------------


class CategoricalPPO:
    """On-policy PPO trainer for discrete action spaces.

    Follows the standard PPO clipped-surrogate pipeline but uses
    :class:`Categorical` distributions instead of Gaussian.  No KL-based
    learning-rate adaptation (not applicable without a parameterised std).

    Typical usage::

        policy = CategoricalActorCritic(num_obs, num_actions)
        storage = CategoricalRolloutStorage(num_envs, N, obs_dim)
        ppo = CategoricalPPO(policy, storage, lr=3e-4, ...)

        for iteration in range(max_iterations):
            for _ in range(N):
                actions = ppo.act(obs)
                obs, rewards, dones, extras = env.step(mapped_actions)
                ppo.process_env_step(obs, rewards, dones, extras)
            ppo.compute_returns(obs)
            loss_dict = ppo.update()
    """

    def __init__(
        self,
        policy: CategoricalActorCritic,
        storage: CategoricalRolloutStorage,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        update_epochs: int = 5,
        num_mini_batches: int = 4,
        device: str = "cuda:0",
    ) -> None:
        self.policy = policy
        self.storage = storage
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_mini_batches = num_mini_batches
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.transition = CategoricalRolloutStorage.Transition()

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions, cache log-prob / value / obs.

        Args:
            obs: ``(num_envs, obs_dim)`` observation tensor.

        Returns:
            Discrete action indices ``(num_envs, 1)`` as long tensor.
        """
        self.transition.observations = obs
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        return self.transition.actions

    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Record transition data from an environment step.

        Handles bootstrap value addition for time-out (truncated) episodes.

        Args:
            obs: New observations ``(num_envs, obs_dim)``.
            rewards: Step rewards ``(num_envs,)``.
            dones: Combined terminated | truncated ``(num_envs,)``.
            extras: Info dict; may contain ``"time_outs"``.
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()

        if "time_outs" in extras:
            time_outs = extras["time_outs"].float()
            assert self.transition.values is not None
            self.transition.rewards += self.gamma * self.transition.values.squeeze(-1) * time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """Compute GAE returns using the last observation for bootstrapping.

        Args:
            last_obs: Final observation ``(num_envs, obs_dim)``.
        """
        last_values = self.policy.evaluate(last_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.gae_lambda)

    def update(self) -> dict[str, float]:
        """Run PPO update epochs over stored rollouts.

        Returns:
            Dict with mean ``value_loss``, ``surrogate_loss``, and ``entropy``.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.update_epochs)
        num_updates = 0

        for obs_batch, actions_batch, old_log_prob_batch, returns_batch, advantages_batch in generator:
            self.policy.act(obs_batch)
            new_log_prob = self.policy.get_actions_log_prob(actions_batch)
            values = self.policy.evaluate(obs_batch)
            entropy = self.policy.entropy

            ratio = torch.exp(new_log_prob - old_log_prob_batch)
            surr1 = advantages_batch * ratio
            surr2 = advantages_batch * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns_batch - values).pow(2).mean()

            loss = surrogate_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            num_updates += 1

        self.storage.clear()

        return {
            "value_loss": mean_value_loss / max(num_updates, 1),
            "surrogate_loss": mean_surrogate_loss / max(num_updates, 1),
            "entropy": mean_entropy / max(num_updates, 1),
        }
