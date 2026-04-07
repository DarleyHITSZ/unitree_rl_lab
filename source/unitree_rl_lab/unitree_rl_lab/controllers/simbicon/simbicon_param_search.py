"""MSLPO Phase 1: Q-learning parameter search for SIMBICON gait parameters.

Implements a sparse Q-learning based search over four gait parameters:
    HL      in [20, 60]   (41 values)
    Ls      in [40, 95]   (56 values)
    Lswb    in [15, 40]   (26 values)
    Lforward in [5, 40]   (36 values)

Total action space: 41 * 56 * 26 * 36 = 2,149,056 combinations.

State space: discretized forward velocity bins.

Reward:
    r^p = r_suc + r_v + r_y
    r_suc = +1 (truncated, no fall), -1 (terminated/fall), -2 (NaN/error)
    r_v   = k_v * max(forward_vel, 0)
    r_y   = -k_y * abs(lateral_offset)
"""

from __future__ import annotations

import heapq
import json
import os
import random
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .g1_joint_map import G1JointIndices
    from .simbicon_controller import SimbiconController, SimbiconStateData


@dataclass
class GaitParameterSpace:
    """Discrete parameter space for the 4 gait parameters.

    Attributes:
        hl_range: (min, max) for HL (foot clearance).
        ls_range: (min, max) for Ls (step length).
        lswb_range: (min, max) for Lswb (lateral sway).
        lforward_range: (min, max) for Lforward (forward distance).
    """

    hl_range: tuple[int, int] = (20, 60)
    ls_range: tuple[int, int] = (40, 95)
    lswb_range: tuple[int, int] = (15, 40)
    lforward_range: tuple[int, int] = (5, 40)

    @property
    def hl_size(self) -> int:
        return self.hl_range[1] - self.hl_range[0] + 1

    @property
    def ls_size(self) -> int:
        return self.ls_range[1] - self.ls_range[0] + 1

    @property
    def lswb_size(self) -> int:
        return self.lswb_range[1] - self.lswb_range[0] + 1

    @property
    def lforward_size(self) -> int:
        return self.lforward_range[1] - self.lforward_range[0] + 1

    @property
    def total_actions(self) -> int:
        return self.hl_size * self.ls_size * self.lswb_size * self.lforward_size

    def params_to_action_idx(self, hl: int, ls: int, lswb: int, lforward: int) -> int:
        """Convert 4 gait parameters to a flat action index.

        Args:
            hl: Foot clearance (integer in [20, 60]).
            ls: Step length (integer in [40, 95]).
            lswb: Lateral sway (integer in [15, 40]).
            lforward: Forward distance (integer in [5, 40]).

        Returns:
            Flat action index in [0, total_actions).
        """
        hl_i = hl - self.hl_range[0]
        ls_i = ls - self.ls_range[0]
        lswb_i = lswb - self.lswb_range[0]
        lforward_i = lforward - self.lforward_range[0]
        return (
            hl_i * (self.ls_size * self.lswb_size * self.lforward_size)
            + ls_i * (self.lswb_size * self.lforward_size)
            + lswb_i * self.lforward_size
            + lforward_i
        )

    def action_idx_to_params(self, action_idx: int) -> tuple[int, int, int, int]:
        """Convert a flat action index to 4 gait parameters.

        Args:
            action_idx: Flat action index in [0, total_actions).

        Returns:
            Tuple of (HL, Ls, Lswb, Lforward) integers.
        """
        lforward_size = self.lforward_size
        lswb_size = self.lswb_size
        ls_size = self.ls_size

        lforward_i = action_idx % lforward_size
        remainder = action_idx // lforward_size
        lswb_i = remainder % lswb_size
        remainder //= lswb_size
        ls_i = remainder % ls_size
        hl_i = remainder // ls_size

        return (
            hl_i + self.hl_range[0],
            ls_i + self.ls_range[0],
            lswb_i + self.lswb_range[0],
            lforward_i + self.lforward_range[0],
        )

    def random_params(self) -> tuple[int, int, int, int]:
        """Sample random parameter tuple.

        Returns:
            Tuple of (HL, Ls, Lswb, Lforward) integers.
        """
        return (
            random.randint(*self.hl_range),
            random.randint(*self.ls_range),
            random.randint(*self.lswb_range),
            random.randint(*self.lforward_range),
        )


@dataclass
class VelocityStateDiscretizer:
    """Discretizes forward velocity into state bins for Q-learning.

    Attributes:
        vel_min: Minimum velocity for clipping.
        vel_max: Maximum velocity for clipping.
        num_bins: Number of discrete bins.
    """

    vel_min: float = -0.5
    vel_max: float = 2.0
    num_bins: int = 25

    def discretize(self, forward_vel: float) -> int:
        """Map a scalar forward velocity to a discrete state id.

        Args:
            forward_vel: Forward velocity in m/s.

        Returns:
            Integer state id in [0, num_bins).
        """
        clamped = max(self.vel_min, min(self.vel_max, forward_vel))
        bin_width = (self.vel_max - self.vel_min) / self.num_bins
        state_id = int((clamped - self.vel_min) / bin_width)
        return min(state_id, self.num_bins - 1)


@dataclass
class SparseQTable:
    """Lazy-initialized sparse Q-table for Q-learning.

    Uses a Python dict to avoid allocating a 2M+ entry dense table.
    Keys are (state_bin, action_idx) tuples; values are Q-values
    initialized to 0 on first access. A per-state action index
    (_state_actions) enables efficient max_q / best_action lookups
    over only the visited actions for a given state.

    Attributes:
        default_q: Initial Q-value for unseen state-action pairs.
        _table: Internal dict mapping (state, action) -> q_value.
        _state_actions: Per-state set of visited action indices.
    """

    default_q: float = 0.0
    _table: dict[tuple[int, int], float] = field(default_factory=dict)
    _state_actions: dict[int, set[int]] = field(default_factory=dict)

    def get_q(self, state: int, action: int) -> float:
        """Get Q-value for a state-action pair.

        Args:
            state: Discrete state bin id.
            action: Flat action index.

        Returns:
            Q-value (default_q if never visited).
        """
        return self._table.get((state, action), self.default_q)

    def set_q(self, state: int, action: int, value: float) -> None:
        """Set Q-value for a state-action pair.

        Args:
            state: Discrete state bin id.
            action: Flat action index.
            value: New Q-value.
        """
        self._table[(state, action)] = value
        if state not in self._state_actions:
            self._state_actions[state] = set()
        self._state_actions[state].add(action)

    def has_state(self, state: int) -> bool:
        """Check if any action has been visited for a given state.

        Args:
            state: Discrete state bin id.

        Returns:
            True if at least one action has been visited.
        """
        return state in self._state_actions

    def max_q(self, state: int) -> float:
        """Get max Q-value over visited actions for a given state.

        Only iterates over actions that have been visited for this state,
        making it efficient for sparse exploration.

        Args:
            state: Discrete state bin id.

        Returns:
            Maximum Q-value for the state (default_q if no actions visited).
        """
        visited = self._state_actions.get(state)
        if not visited:
            return self.default_q
        return max(self._table.get((state, a), self.default_q) for a in visited)

    def best_action(self, state: int) -> int | None:
        """Get the action with highest Q-value for a given state.

        Only considers visited actions. Returns None if no actions have
        been visited for this state.

        Args:
            state: Discrete state bin id.

        Returns:
            Action index with the highest Q-value, or None if unvisited.
        """
        visited = self._state_actions.get(state)
        if not visited:
            return None
        best_q = self.default_q
        best_a = -1
        for a in visited:
            q = self._table.get((state, a), self.default_q)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def size(self) -> int:
        """Return the number of visited state-action pairs."""
        return len(self._table)

    def to_dict(self) -> dict[str, object]:
        """Serialize the Q-table to a JSON-compatible dict.

        Returns:
            Dict with 'default_q', 'size', and 'entries' keys.
        """
        return {
            "default_q": self.default_q,
            "size": self.size(),
            "entries": {f"{s},{a}": v for (s, a), v in self._table.items()},
        }

    def to_dict_with_meta(self, epsilon: float, episode: int) -> dict[str, object]:
        """Serialize Q-table with training metadata for checkpoint resume.

        Args:
            epsilon: Current exploration rate.
            episode: Current episode count.

        Returns:
            Dict with Q-table data plus 'epsilon' and 'episode' metadata.
        """
        d = self.to_dict()
        d["epsilon"] = epsilon
        d["episode"] = episode
        return d

    def save(self, path: str) -> None:
        """Save Q-table to a JSON file.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> tuple[SparseQTable, float, int]:
        """Load Q-table from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Tuple of (loaded SparseQTable, epsilon, episode_count).
            If metadata is missing, returns defaults (epsilon=1.0, episode=0).
        """
        with open(path) as f:
            data = json.load(f)
        table = cls(default_q=data.get("default_q", 0.0))
        entries = data.get("entries", {})
        for key, val in entries.items():
            s_str, a_str = key.split(",")
            s, a = int(s_str), int(a_str)
            table._table[(s, a)] = val
            if s not in table._state_actions:
                table._state_actions[s] = set()
            table._state_actions[s].add(a)
        epsilon = data.get("epsilon", 1.0)
        episode = data.get("episode", 0)
        return table, epsilon, episode


@dataclass
class RolloutResult:
    """Result of a single parameter rollout evaluation.

    Attributes:
        avg_forward_velocity: Mean forward velocity over the episode.
        avg_lateral_offset: Mean absolute lateral offset.
        alive_time: Time survived in seconds.
        terminated: Whether the episode ended in termination (fall).
        truncated: Whether the episode was truncated (time limit).
        total_reward: Accumulated Q-learning reward.
        step_count: Number of simulation steps completed.
        hl: HL parameter used.
        ls: Ls parameter used.
        lswb: Lswb parameter used.
        lforward: Lforward parameter used.
    """

    avg_forward_velocity: float = 0.0
    avg_lateral_offset: float = 0.0
    alive_time: float = 0.0
    terminated: bool = False
    truncated: bool = False
    total_reward: float = 0.0
    step_count: int = 0
    hl: int = 0
    ls: int = 0
    lswb: int = 0
    lforward: int = 0

    @property
    def success(self) -> bool:
        return self.truncated and not self.terminated

    @property
    def score(self) -> float:
        return self.total_reward


def evaluate_gait_params(
    env: ManagerBasedRLEnv,
    controller: SimbiconController,
    joint_indices: G1JointIndices,
    params: tuple[int, int, int, int],
    max_steps: int,
    k_v: float = 1.0,
    k_y: float = 3.0,
    k_alive: float = 10.0,
    skip_set_params: bool = False,
) -> RolloutResult:
    """Evaluate a single set of gait parameters via full episode rollout.

    Args:
        env: The ManagerBasedRLEnv instance (unwrapped).
        controller: The SimbiconController instance.
        joint_indices: Resolved G1 joint indices.
        params: Tuple of (HL, Ls, Lswb, Lforward) integers.
        max_steps: Maximum number of simulation steps.
        k_v: Velocity reward coefficient.
        k_y: Lateral offset penalty coefficient.
        k_alive: Survival reward coefficient.
        skip_set_params: If True, skip set_gait_params (diagnostic mode).

    Returns:
        RolloutResult with episode metrics.
    """
    hl, ls, lswb, lforward = params
    unwrapped_env = env.unwrapped
    dt = unwrapped_env.step_dt

    if not skip_set_params:
        controller.set_gait_params(hl=hl, ls=ls, lswb=lswb, lforward=lforward)
    controller.reset()

    with torch.inference_mode():
        obs, _ = env.reset()

    state_data = _extract_state_data(unwrapped_env, joint_indices)

    initial_y = state_data.root_pos[0, 1].item()

    total_reward = 0.0
    vel_sum = 0.0
    lateral_sum = 0.0
    step_count = 0
    terminated_flag = False
    truncated_flag = False

    for _ in range(max_steps):
        with torch.inference_mode():
            state_data = _extract_state_data(unwrapped_env, joint_indices)

            targets = controller.step(state_data, dt)
            actions = controller.compute_actions_from_targets(targets)

            has_nan = torch.isnan(targets).any() or torch.isnan(actions).any()
            if has_nan:
                total_reward += -2.0
                terminated_flag = True
                break

            obs, rewards, terminated, truncated, extras = env.step(actions)

        step_count += 1
        forward_vel = state_data.root_lin_vel[0, 0].item()
        lateral_offset = abs(state_data.root_pos[0, 1].item() - initial_y)

        r_v = k_v * max(forward_vel, 0.0)
        r_y = -k_y * lateral_offset
        r_alive = k_alive / max_steps

        total_reward += r_v + r_y + r_alive
        vel_sum += forward_vel
        lateral_sum += lateral_offset

        if terminated[0].item():
            base_z = state_data.root_pos[0, 2].item()
            total_reward += -5.0
            terminated_flag = True
            print(f"  [TERM] step={step_count} base_z={base_z:.3f} " f"params=({hl},{ls},{lswb},{lforward})")
            break

        if truncated[0].item():
            total_reward += 1.0
            truncated_flag = True
            break

    alive_time = step_count * dt
    avg_vel = vel_sum / max(step_count, 1)
    avg_lateral = lateral_sum / max(step_count, 1)

    return RolloutResult(
        avg_forward_velocity=avg_vel,
        avg_lateral_offset=avg_lateral,
        alive_time=alive_time,
        terminated=terminated_flag,
        truncated=truncated_flag,
        total_reward=total_reward,
        step_count=step_count,
        hl=hl,
        ls=ls,
        lswb=lswb,
        lforward=lforward,
    )


def _extract_state_data(env: ManagerBasedRLEnv, joint_indices: G1JointIndices) -> SimbiconStateData:
    """Extract robot state from the environment.

    Args:
        env: The ManagerBasedRLEnv instance (unwrapped).
        joint_indices: Resolved G1 joint indices.

    Returns:
        SimbiconStateData with current robot state.
    """
    from .simbicon_controller import SimbiconStateData

    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]

    root_state = robot.data.root_state_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    body_pos = robot.data.body_pos_w

    contact_data = contact_sensor.data
    net_forces = contact_data.net_forces_w

    left_force = torch.zeros(env.num_envs, device=env.device)
    right_force = torch.zeros(env.num_envs, device=env.device)

    if net_forces is not None and net_forces.shape[1] > 0:
        if joint_indices.left_foot_sensor_id >= 0:
            left_force = net_forces[:, joint_indices.left_foot_sensor_id, :].norm(dim=1)
        if joint_indices.right_foot_sensor_id >= 0:
            right_force = net_forces[:, joint_indices.right_foot_sensor_id, :].norm(dim=1)

    return SimbiconStateData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_state[:, :3],
        root_quat=root_state[:, 3:7],
        root_lin_vel=root_state[:, 7:10],
        root_ang_vel=root_state[:, 10:13],
        body_pos=body_pos,
        left_foot_contact=left_force > 1.0,
        right_foot_contact=right_force > 1.0,
        left_foot_force=left_force,
        right_foot_force=right_force,
    )


@dataclass
class SimbiconQLearningSearcher:
    """Q-learning based gait parameter searcher for MSLPO Phase 1.

    Uses epsilon-greedy exploration over the sparse Q-table to search
    for optimal gait parameters. Maintains a top-k leaderboard of the
    best parameter sets found.

    Args:
        param_space: Gait parameter space definition.
        state_discretizer: Velocity state discretizer.
        alpha: Q-learning learning rate.
        gamma: Q-learning discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        top_k: Number of best results to track.
    """

    param_space: GaitParameterSpace = field(default_factory=GaitParameterSpace)
    state_discretizer: VelocityStateDiscretizer = field(default_factory=VelocityStateDiscretizer)
    q_table: SparseQTable = field(default_factory=SparseQTable)
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    top_k: int = 5

    _epsilon: float = 1.0
    _episode: int = 0
    _top_results: list[tuple[float, RolloutResult]] = field(default_factory=list)
    _episode_log: list[dict] = field(default_factory=list)
    _prev_state: int = 0
    _prev_action: int = 0

    def select_action(self, state: int) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current discrete state id.

        Returns:
            Flat action index.
        """
        if torch.rand(1).item() < self._epsilon:
            hl, ls, lswb, lforward = self.param_space.random_params()
            return self.param_space.params_to_action_idx(hl, ls, lswb, lforward)
        else:
            best = self.q_table.best_action(state)
            if best is None:
                hl, ls, lswb, lforward = self.param_space.random_params()
                return self.param_space.params_to_action_idx(hl, ls, lswb, lforward)
            return best

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Perform a single Q-learning update.

        Args:
            state: Previous state id.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state id.
        """
        old_q = self.q_table.get_q(state, action)
        max_next_q = self.q_table.max_q(next_state)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table.set_q(state, action, new_q)

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)

    def record_result(self, result: RolloutResult) -> None:
        """Record a rollout result and update top-k leaderboard.

        Args:
            result: The rollout evaluation result.
        """
        key = (result.hl, result.ls, result.lswb, result.lforward)
        self._top_results = [(s, r) for s, r in self._top_results if (r.hl, r.ls, r.lswb, r.lforward) != key]
        heapq.heappush(self._top_results, (result.score, result))
        if len(self._top_results) > self.top_k:
            heapq.heappop(self._top_results)

        self._episode_log.append(
            {
                "episode": self._episode,
                "hl": result.hl,
                "ls": result.ls,
                "lswb": result.lswb,
                "lforward": result.lforward,
                "avg_forward_velocity": result.avg_forward_velocity,
                "avg_lateral_offset": result.avg_lateral_offset,
                "alive_time": result.alive_time,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "success": result.success,
                "total_reward": result.total_reward,
                "score": result.score,
                "epsilon": self._epsilon,
                "q_table_size": self.q_table.size(),
            }
        )

    def get_top_results(self) -> list[RolloutResult]:
        """Get top-k results sorted by score (best first).

        Returns:
            List of top RolloutResult entries.
        """
        sorted_results = sorted(self._top_results, key=lambda x: x[0], reverse=True)
        return [r for _, r in sorted_results]

    def save_checkpoint(self, output_dir: str) -> None:
        """Save Q-table, top results, and episode log.

        Args:
            output_dir: Directory to save outputs.
        """
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_data = self.q_table.to_dict_with_meta(self._epsilon, self._episode)
        with open(os.path.join(output_dir, "q_table_checkpoint.json"), "w") as f:
            json.dump(checkpoint_data, f)

        top_results = self.get_top_results()
        top_data = []
        for r in top_results:
            top_data.append(
                {
                    "rank": len(top_data) + 1,
                    "HL": r.hl,
                    "Ls": r.ls,
                    "Lswb": r.lswb,
                    "Lforward": r.lforward,
                    "avg_forward_velocity": round(r.avg_forward_velocity, 6),
                    "avg_lateral_offset": round(r.avg_lateral_offset, 6),
                    "alive_time": round(r.alive_time, 4),
                    "success": r.success,
                    "total_reward": round(r.total_reward, 4),
                    "score": round(r.score, 4),
                }
            )
        with open(os.path.join(output_dir, "top5_pose_params.json"), "w") as f:
            json.dump(top_data, f, indent=2)

        with open(os.path.join(output_dir, "episode_log.json"), "w") as f:
            json.dump(self._episode_log, f, indent=2)

    def run_episode(
        self,
        env: ManagerBasedRLEnv,
        controller: SimbiconController,
        joint_indices: G1JointIndices,
        max_steps: int,
        k_v: float = 1.0,
        k_y: float = 3.0,
        k_alive: float = 10.0,
        skip_set_params: bool = False,
    ) -> RolloutResult:
        """Run a single Q-learning episode.

        Flow:
            1. Select action (parameter set) via epsilon-greedy
            2. Evaluate with rollout
            3. Compute reward and next state
            4. Update Q-table
            5. Record result

        Args:
            env: The ManagerBasedRLEnv instance (unwrapped).
            controller: The SimbiconController instance.
            joint_indices: Resolved G1 joint indices.
            max_steps: Maximum simulation steps per rollout.
            k_v: Velocity reward coefficient.
            k_y: Lateral offset penalty coefficient.
            k_alive: Survival reward coefficient.
            skip_set_params: If True, skip set_gait_params (diagnostic mode).

        Returns:
            RolloutResult from the evaluation.
        """
        state = self._prev_state
        action = self.select_action(state)
        params = self.param_space.action_idx_to_params(action)

        result = evaluate_gait_params(
            env,
            controller,
            joint_indices,
            params,
            max_steps,
            k_v=k_v,
            k_y=k_y,
            k_alive=k_alive,
            skip_set_params=skip_set_params,
        )

        next_state = self.state_discretizer.discretize(result.avg_forward_velocity)
        self.update(state, action, result.total_reward, next_state)

        self._prev_state = next_state
        self._prev_action = action
        self.decay_epsilon()
        self.record_result(result)
        self._episode += 1

        return result
