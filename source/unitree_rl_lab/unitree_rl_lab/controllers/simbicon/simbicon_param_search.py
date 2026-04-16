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
        time_to_fall: Seconds until termination (alive_time if fell, max_steps*dt if survived).
        early_window_forward_velocity: Mean forward velocity over the first 50 steps.
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
    time_to_fall: float = 0.0
    early_window_forward_velocity: float = 0.0
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
    early_vel_sum = 0.0
    early_window_steps = 50

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
        if step_count <= early_window_steps:
            early_vel_sum += forward_vel

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
    time_to_fall = alive_time if terminated_flag else (max_steps * dt)
    early_vel = early_vel_sum / min(step_count, early_window_steps)

    return RolloutResult(
        avg_forward_velocity=avg_vel,
        avg_lateral_offset=avg_lateral,
        alive_time=alive_time,
        terminated=terminated_flag,
        truncated=truncated_flag,
        total_reward=total_reward,
        step_count=step_count,
        time_to_fall=time_to_fall,
        early_window_forward_velocity=early_vel,
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
                "time_to_fall": result.time_to_fall,
                "early_window_forward_velocity": result.early_window_forward_velocity,
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


_DEFAULT_TOP5 = [
    {"HL": 20, "Ls": 52, "Lswb": 40, "Lforward": 18, "total_reward": 15.34},
    {"HL": 20, "Ls": 48, "Lswb": 37, "Lforward": 12, "total_reward": 14.90},
    {"HL": 24, "Ls": 48, "Lswb": 23, "Lforward": 40, "total_reward": 14.37},
    {"HL": 25, "Ls": 41, "Lswb": 35, "Lforward": 6, "total_reward": 14.03},
    {"HL": 47, "Ls": 41, "Lswb": 33, "Lforward": 39, "total_reward": 13.96},
]

_PARAM_NAMES = ("HL", "Ls", "Lswb", "Lforward")

_PARAM_RANGES: dict[str, tuple[int, int]] = {
    "HL": (20, 60),
    "Ls": (40, 95),
    "Lswb": (15, 40),
    "Lforward": (5, 40),
}

_PARAM_SCAN_STEPS: dict[str, int] = {
    "HL": 4,
    "Ls": 5,
    "Lswb": 3,
    "Lforward": 4,
}


def _make_scan_grid(lo: int, hi: int, step: int) -> list[int]:
    grid = list(range(lo, hi + 1, step))
    if grid[-1] != hi:
        grid.append(hi)
    return sorted(set(grid))


def _compute_base_params(top5: list[dict], mode: str) -> dict[str, int]:
    if mode == "best":
        best = max(top5, key=lambda d: d.get("total_reward", 0.0))
        return {p: int(best[p]) for p in _PARAM_NAMES}
    values_by_param: dict[str, list[int]] = {p: [int(t5[p]) for t5 in top5] for p in _PARAM_NAMES}
    if mode == "median":
        return {p: int(sorted(vals)[len(vals) // 2]) for p, vals in values_by_param.items()}
    return {p: int(round(sum(vals) / len(vals))) for p, vals in values_by_param.items()}


@dataclass
class ParameterSensitivityAnalyzer:
    """Control-variable sensitivity scanner for one gait parameter at a time.

    For each parameter, all other three are fixed at *base_params* while the
    target parameter sweeps through a coarse grid.  Every grid point is
    evaluated with ``num_rollouts`` independent rollouts (fixed seed set) and
    the results are averaged.

    Attributes:
        env: The wrapped ManagerBasedRLEnv.
        controller: The SimbiconController instance.
        joint_indices: Resolved G1 joint indices.
        max_steps: Maximum simulation steps per rollout.
        k_v: Velocity reward coefficient.
        k_y: Lateral offset penalty coefficient.
        k_alive: Survival reward coefficient.
        num_rollouts: Number of rollouts per grid point.
        seeds: Fixed random seed list for reproducibility.
        base_param_mode: How to derive base params from top5 ("mean"|"median"|"best").
        sensitivity_score_metric: Metric key used for ranking best scan point
            and passed to DynamicDiscretizer ("composite"|"time_to_fall"|"total_reward"|"early_forward_velocity").
        sensitivity_alpha: Weight for time_to_fall in composite score (1-alpha = weight for early velocity).
    """

    env: "ManagerBasedRLEnv" = None  # type: ignore[assignment]
    controller: "SimbiconController" = None  # type: ignore[assignment]
    joint_indices: "G1JointIndices" = None  # type: ignore[assignment]
    max_steps: int = 2000
    k_v: float = 1.0
    k_y: float = 3.0
    k_alive: float = 10.0
    num_rollouts: int = 3
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    base_param_mode: str = "median"
    sensitivity_score_metric: str = "composite"
    sensitivity_alpha: float = 0.6

    def _base_params(self) -> dict[str, int]:
        return _compute_base_params(_DEFAULT_TOP5, self.base_param_mode)

    def _compute_composite_and_inject(self, results: list[dict]) -> None:
        ttf_vals = [r["mean_time_to_fall"] for r in results]
        ev_vals = [r["mean_early_window_forward_velocity"] for r in results]
        ttf_min, ttf_max = min(ttf_vals), max(ttf_vals)
        ev_min, ev_max = min(ev_vals), max(ev_vals)
        ttf_range = max(ttf_max - ttf_min, 1e-9)
        ev_range = max(ev_max - ev_min, 1e-9)
        for i, r in enumerate(results):
            ttf_n = (ttf_vals[i] - ttf_min) / ttf_range
            ev_n = (ev_vals[i] - ev_min) / ev_range
            r["composite_score"] = round(self.sensitivity_alpha * ttf_n + (1 - self.sensitivity_alpha) * ev_n, 6)

    def scan_parameter(
        self,
        param_name: str,
        candidate_values: list[int],
        base_params: dict[str, int],
    ) -> list[dict]:
        results: list[dict] = []
        for val in candidate_values:
            params = dict(base_params)
            params[param_name] = val
            rewards = []
            vels = []
            laterals = []
            alives = []
            ttf_list = []
            ev_list = []
            falls = 0
            for _ in range(self.num_rollouts):
                r = evaluate_gait_params(
                    self.env,
                    self.controller,
                    self.joint_indices,
                    (params["HL"], params["Ls"], params["Lswb"], params["Lforward"]),
                    self.max_steps,
                    k_v=self.k_v,
                    k_y=self.k_y,
                    k_alive=self.k_alive,
                )
                rewards.append(r.total_reward)
                vels.append(r.avg_forward_velocity)
                laterals.append(r.avg_lateral_offset)
                alives.append(r.alive_time)
                ttf_list.append(r.time_to_fall)
                ev_list.append(r.early_window_forward_velocity)
                if r.terminated:
                    falls += 1
            results.append(
                {
                    "param_name": param_name,
                    "param_value": val,
                    "mean_total_reward": round(sum(rewards) / len(rewards), 6),
                    "mean_forward_velocity": round(sum(vels) / len(vels), 6),
                    "mean_lateral_offset": round(sum(laterals) / len(laterals), 6),
                    "mean_alive_time": round(sum(alives) / len(alives), 4),
                    "mean_time_to_fall": round(sum(ttf_list) / len(ttf_list), 4),
                    "mean_early_window_forward_velocity": round(sum(ev_list) / len(ev_list), 6),
                    "fall_rate": round(falls / self.num_rollouts, 4),
                }
            )
        return results

    def _metric_key(self) -> str:
        mapping = {
            "composite": "composite_score",
            "time_to_fall": "mean_time_to_fall",
            "total_reward": "mean_total_reward",
            "early_forward_velocity": "mean_early_window_forward_velocity",
        }
        return mapping.get(self.sensitivity_score_metric, "composite_score")

    def scan_all_parameters(self) -> dict[str, list[dict]]:
        base = self._base_params()
        metric_key = self._metric_key()
        all_results: dict[str, list[dict]] = {}
        for pname in _PARAM_NAMES:
            lo, hi = _PARAM_RANGES[pname]
            step = _PARAM_SCAN_STEPS[pname]
            grid = _make_scan_grid(lo, hi, step)
            print(f"[SCAN] {pname}: sweeping {len(grid)} values in [{lo}, {hi}], base={base}")
            results = self.scan_parameter(pname, grid, base)
            if self.sensitivity_score_metric == "composite":
                self._compute_composite_and_inject(results)
            all_results[pname] = results
            best = max(results, key=lambda d: d[metric_key])
            print(
                f"[SCAN] {pname}: best_value={best['param_value']} "
                f"{metric_key}={best[metric_key]:+.4f} "
                f"reward={best['mean_total_reward']:+.4f} "
                f"ttf={best['mean_time_to_fall']:.4f}s "
                f"fall_rate={best['fall_rate']:.2f}"
            )
        return all_results

    def save_results(self, results: dict[str, list[dict]], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        total_points = sum(len(v) for v in results.values())
        print(f"[SCAN] Saved {total_points} scan points to {path}")


@dataclass
class DynamicDiscretizer:
    """Builds non-uniform discrete value lists by focusing on high-value zones.

    A composite score S = alpha * norm(time_to_fall) + (1-alpha) * norm(early_velocity)
    is computed per scan point, smoothed, then the top 30% (score >= p70) are
    identified as high-value points.  These are merged into contiguous "focus zones"
    and expanded by one scan step.  Inside focus zones every integer value is kept
    (step 1); outside, only the original coarse scan grid points and endpoints are
    retained.  This concentrates the action space density where the robot performs
    best rather than where the sensitivity gradient happens to be steep.

    Attributes:
        metric: Scan result key for performance signal ("composite_score" or others).
        sensitivity_alpha: Weight for time_to_fall in composite score.
        high_value_pct: Percentile threshold for high-value zone identification.
        focus_step: Step size inside high-value zones.
        smooth_window: Width of the moving-average smoother.
    """

    metric: str = "composite_score"
    sensitivity_alpha: float = 0.6
    high_value_pct: float = 70.0
    focus_step: int = 1
    smooth_window: int = 3

    def _smooth(self, values: list[float]) -> list[float]:
        if len(values) < 3:
            return list(values)
        half = self.smooth_window // 2
        out = list(values)
        for i in range(half, len(values) - half):
            out[i] = sum(values[i - half : i + half + 1]) / (2 * half + 1)
        return out

    def _get_score_series(self, scan_results: list[dict]) -> list[float]:
        if self.metric == "composite_score":
            ttf_vals = [r["mean_time_to_fall"] for r in scan_results]
            ev_vals = [r["mean_early_window_forward_velocity"] for r in scan_results]
            ttf_min, ttf_max = min(ttf_vals), max(ttf_vals)
            ev_min, ev_max = min(ev_vals), max(ev_vals)
            ttf_range = max(ttf_max - ttf_min, 1e-9)
            ev_range = max(ev_max - ev_min, 1e-9)
            raw = [
                self.sensitivity_alpha * (ttf_vals[i] - ttf_min) / ttf_range
                + (1 - self.sensitivity_alpha) * (ev_vals[i] - ev_min) / ev_range
                for i in range(len(ttf_vals))
            ]
        else:
            raw = [r.get(self.metric, 0.0) for r in scan_results]
        return self._smooth(raw)

    def compute_sensitivity_scores(self, scan_results: list[dict], param_name: str) -> list[dict]:
        x = [r["param_value"] for r in scan_results]
        s = self._get_score_series(scan_results)
        scores: list[dict] = []
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            if dx == 0:
                continue
            g = abs(s[i + 1] - s[i]) / dx
            scores.append(
                {
                    "param_name": param_name,
                    "range_start": x[i],
                    "range_end": x[i + 1],
                    "sensitivity_score": round(g, 6),
                    "smoothed_value_start": round(s[i], 6),
                    "smoothed_value_end": round(s[i + 1], 6),
                }
            )
        return scores

    def _find_high_value_zones(self, scan_results: list[dict], param_name: str) -> list[tuple[int, int]]:
        s = self._get_score_series(scan_results)
        x = [r["param_value"] for r in scan_results]
        sorted_s = sorted(s)
        p70 = sorted_s[min(len(sorted_s) - 1, int(len(sorted_s) * self.high_value_pct / 100))]
        high_points = [x[i] for i in range(len(x)) if s[i] >= p70]
        if not high_points:
            return []
        scan_step = _PARAM_SCAN_STEPS[param_name]
        lo_param, hi_param = _PARAM_RANGES[param_name]
        intervals: list[tuple[int, int]] = []
        cur_start = high_points[0]
        cur_end = high_points[0]
        for px in high_points[1:]:
            if px - cur_end <= scan_step * 1.5:
                cur_end = px
            else:
                intervals.append((cur_start, cur_end))
                cur_start = px
                cur_end = px
        intervals.append((cur_start, cur_end))
        expanded = [(max(cs - scan_step, lo_param), min(ce + scan_step, hi_param)) for cs, ce in intervals]
        merged = [expanded[0]]
        for iv in expanded[1:]:
            if iv[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], iv[1]))
            else:
                merged.append(iv)
        return merged

    def build_value_list(self, scan_results: list[dict], param_name: str) -> list[int]:
        lo, hi = _PARAM_RANGES[param_name]
        scan_step = _PARAM_SCAN_STEPS[param_name]
        scan_grid = set(_make_scan_grid(lo, hi, scan_step))
        focus_zones = self._find_high_value_zones(scan_results, param_name)
        values: set[int] = set()
        if focus_zones:
            for f_lo, f_hi in focus_zones:
                values.update(range(int(f_lo), int(f_hi) + 1))
        for v in scan_grid:
            values.add(v)
        values.add(lo)
        values.add(hi)
        result = sorted(v for v in values if lo <= v <= hi)
        return result

    def build_all(self, scan_results_by_param: dict[str, list[dict]]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for pname in _PARAM_NAMES:
            vals = self.build_value_list(scan_results_by_param[pname], pname)
            result[pname] = vals
            zones = self._find_high_value_zones(scan_results_by_param[pname], pname)
            zone_str = ", ".join(f"[{z[0]},{z[1]}]" for z in zones) if zones else "none"
            print(f"[DISC] {pname}: {len(vals)} values, focus zones: {zone_str}")
        return result

    def save_config(self, value_lists: dict[str, list[int]], scan_results_by_param: dict, path: str) -> None:
        config: dict = {
            "sensitivity_score_mode": self.metric,
            "sensitivity_alpha": self.sensitivity_alpha,
            "parameters": {},
        }
        for pname in _PARAM_NAMES:
            zones = self._find_high_value_zones(scan_results_by_param[pname], pname)
            scores = self.compute_sensitivity_scores(scan_results_by_param[pname], pname)
            config["parameters"][pname] = {
                "original_range": list(_PARAM_RANGES[pname]),
                "focus_zones": [list(z) for z in zones],
                "smoothed_sensitivity_scores": [
                    {
                        "range": [s["range_start"], s["range_end"]],
                        "score": s["sensitivity_score"],
                        "smoothed_start": s["smoothed_value_start"],
                        "smoothed_end": s["smoothed_value_end"],
                    }
                    for s in scores
                ],
                "discrete_values": value_lists[pname],
                "num_values": len(value_lists[pname]),
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[DISC] Saved discretization config to {path}")


@dataclass
class DynamicParameterSpace:
    """Non-uniform discrete parameter space with the same public API as GaitParameterSpace.

    The four value lists are sorted and deduplicated at construction time.
    Pre-computed index lookups and strides enable O(1) encoding / decoding
    of parameter tuples to / from flat action indices.
    """

    hl_values: list[int] = field(default_factory=list)
    ls_values: list[int] = field(default_factory=list)
    lswb_values: list[int] = field(default_factory=list)
    lforward_values: list[int] = field(default_factory=list)

    _hl_idx: dict[int, int] = field(default_factory=dict, repr=False)
    _ls_idx: dict[int, int] = field(default_factory=dict, repr=False)
    _lswb_idx: dict[int, int] = field(default_factory=dict, repr=False)
    _lforward_idx: dict[int, int] = field(default_factory=dict, repr=False)
    _hl_stride: int = 0
    _ls_stride: int = 0
    _lswb_stride: int = 0

    def __post_init__(self) -> None:
        self.hl_values = sorted(set(self.hl_values))
        self.ls_values = sorted(set(self.ls_values))
        self.lswb_values = sorted(set(self.lswb_values))
        self.lforward_values = sorted(set(self.lforward_values))
        self._hl_idx = {v: i for i, v in enumerate(self.hl_values)}
        self._ls_idx = {v: i for i, v in enumerate(self.ls_values)}
        self._lswb_idx = {v: i for i, v in enumerate(self.lswb_values)}
        self._lforward_idx = {v: i for i, v in enumerate(self.lforward_values)}
        self._lswb_stride = len(self.lforward_values)
        self._ls_stride = len(self.lswb_values) * self._lswb_stride
        self._hl_stride = len(self.ls_values) * self._ls_stride

    @property
    def total_actions(self) -> int:
        return len(self.hl_values) * len(self.ls_values) * len(self.lswb_values) * len(self.lforward_values)

    def params_to_action_idx(self, hl: int, ls: int, lswb: int, lforward: int) -> int:
        return (
            self._hl_idx[hl] * self._hl_stride
            + self._ls_idx[ls] * self._ls_stride
            + self._lswb_idx[lswb] * self._lswb_stride
            + self._lforward_idx[lforward]
        )

    def action_idx_to_params(self, action_idx: int) -> tuple[int, int, int, int]:
        hl_i = action_idx // self._hl_stride
        rem = action_idx % self._hl_stride
        ls_i = rem // self._ls_stride
        rem2 = rem % self._ls_stride
        lswb_i = rem2 // self._lswb_stride
        lf_i = rem2 % self._lswb_stride
        return (self.hl_values[hl_i], self.ls_values[ls_i], self.lswb_values[lswb_i], self.lforward_values[lf_i])

    def random_params(self) -> tuple[int, int, int, int]:
        return (
            random.choice(self.hl_values),
            random.choice(self.ls_values),
            random.choice(self.lswb_values),
            random.choice(self.lforward_values),
        )


def compute_convergence_episode(
    episode_log: list[dict],
    window: int = 20,
    threshold: float = 0.05,
) -> int | None:
    """Compute the episode at which training converged.

    Convergence is defined as the first episode where the coefficient of
    variation (std / |mean|) of *total_reward* over a sliding window of
    ``window`` consecutive episodes falls below ``threshold``.

    Args:
        episode_log: List of per-episode dicts, each containing "total_reward".
        window: Sliding window size in episodes.
        threshold: Maximum coefficient of variation to declare convergence.

    Returns:
        1-indexed episode number at convergence, or None if never converged.
    """
    rewards = [ep["total_reward"] for ep in episode_log]
    for i in range(window, len(rewards) + 1):
        w = rewards[i - window : i]
        mean_r = sum(w) / window
        if mean_r == 0:
            continue
        var_r = sum((r - mean_r) ** 2 for r in w) / window
        cv = (var_r**0.5) / abs(mean_r)
        if cv < threshold:
            return i
    return None


def _best_from_log(episode_log: list[dict]) -> dict:
    return max(episode_log, key=lambda ep: ep["total_reward"])


def compute_episodes_to_target(
    episode_log: list[dict],
    metric_key: str = "total_reward",
    threshold: float = 15.0,
) -> int | None:
    """Find the first episode whose running-best *metric_key* meets or exceeds *threshold*.

    Args:
        episode_log: List of per-episode dicts.
        metric_key: Key to check (e.g. "total_reward", "avg_forward_velocity").
        threshold: Target value to reach.

    Returns:
        1-indexed episode number at which the running best first meets the
        threshold, or None if never reached.
    """
    best_so_far = float("-inf")
    for i, ep in enumerate(episode_log):
        val = ep.get(metric_key, float("-inf"))
        if isinstance(val, (int, float)) and val > best_so_far:
            best_so_far = val
        if best_so_far >= threshold:
            return i + 1
    return None


def _assess_dynamic_search(report: dict) -> str:
    d_target_r = report.get("dynamic_episodes_to_target_reward")
    u_target_r = report.get("uniform_episodes_to_target_reward")
    d_target_v = report.get("dynamic_episodes_to_target_velocity")
    u_target_v = report.get("uniform_episodes_to_target_velocity")

    d_hit_any = d_target_r is not None or d_target_v is not None
    u_hit_any = u_target_r is not None or u_target_v is not None

    if not d_hit_any and not u_hit_any:
        return "inconclusive_due_to_no_target_hit"

    d_faster = False
    if d_target_r is not None and u_target_r is not None and d_target_r <= u_target_r:
        d_faster = True
    if d_target_v is not None and u_target_v is not None and d_target_v <= u_target_v:
        d_faster = True

    if d_faster:
        return "space_reduced_and_target_reached_faster"

    d_reward = report.get("dynamic_best_total_reward", 0.0)
    u_reward = report.get("uniform_best_total_reward", 0.0)
    if u_reward != 0 and (d_reward - u_reward) / abs(u_reward) > 0.01:
        return "space_reduced_with_better_best_performance"

    return "space_reduced_but_no_efficiency_gain"


def compare_uniform_vs_dynamic(
    uniform_episode_log: list[dict],
    dynamic_episode_log: list[dict],
    uniform_search_time: float,
    dynamic_search_time: float,
    scan_time: float,
    uniform_space_size: int,
    dynamic_space_size: int,
    dynamic_space_size_before_tightening: int = 342144,
    target_reward_threshold: float = 15.0,
    target_forward_velocity_threshold: float = 0.33,
) -> dict:
    """Compare uniform vs dynamic discretisation and return a report dict.

    Args:
        uniform_episode_log: Per-episode log from the uniform-space run.
        dynamic_episode_log: Per-episode log from the dynamic-space run.
        uniform_search_time: Wall-clock seconds for uniform search only.
        dynamic_search_time: Wall-clock seconds for dynamic search only.
        scan_time: Wall-clock seconds for the sensitivity scan (Phase A).
        uniform_space_size: Total actions in the uniform space.
        dynamic_space_size: Total actions in the dynamic space (after tightening).
        dynamic_space_size_before_tightening: Total actions before step tightening.
        target_reward_threshold: Reward value for episodes_to_target computation.
        target_forward_velocity_threshold: Velocity value for episodes_to_target.

    Returns:
        Dict with all comparison metrics.
    """
    u_best = _best_from_log(uniform_episode_log) if uniform_episode_log else {}
    d_best = _best_from_log(dynamic_episode_log) if dynamic_episode_log else {}
    u_conv = compute_convergence_episode(uniform_episode_log)
    d_conv = compute_convergence_episode(dynamic_episode_log)
    u_rewards = [ep["total_reward"] for ep in uniform_episode_log]
    d_rewards = [ep["total_reward"] for ep in dynamic_episode_log]
    u_successes = sum(1 for ep in uniform_episode_log if ep.get("success", False))
    d_successes = sum(1 for ep in dynamic_episode_log if ep.get("success", False))
    reduction = (1.0 - dynamic_space_size / uniform_space_size) * 100 if uniform_space_size > 0 else 0.0
    u_best_reward = max(u_rewards) if u_rewards else 0.0
    d_best_reward = max(d_rewards) if d_rewards else 0.0
    improvement = ((d_best_reward - u_best_reward) / abs(u_best_reward) * 100) if u_best_reward != 0 else 0.0

    report = {
        "uniform_action_space_size": uniform_space_size,
        "dynamic_action_space_size_before_tightening": dynamic_space_size_before_tightening,
        "dynamic_action_space_size_after_tightening": dynamic_space_size,
        "action_space_reduction_percent": round(reduction, 2),
        "uniform_convergence_episodes": u_conv,
        "dynamic_convergence_episodes": d_conv,
        "uniform_search_time_only": round(uniform_search_time, 2),
        "dynamic_search_time_only": round(dynamic_search_time, 2),
        "total_time_including_scan": round(dynamic_search_time + scan_time, 2),
        "uniform_best_total_reward": round(u_best.get("total_reward", 0.0), 4),
        "dynamic_best_total_reward": round(d_best.get("total_reward", 0.0), 4),
        "uniform_best_forward_velocity": round(u_best.get("avg_forward_velocity", 0.0), 6),
        "dynamic_best_forward_velocity": round(d_best.get("avg_forward_velocity", 0.0), 6),
        "uniform_best_lateral_offset": round(u_best.get("avg_lateral_offset", 0.0), 6),
        "dynamic_best_lateral_offset": round(d_best.get("avg_lateral_offset", 0.0), 6),
        "uniform_best_time_to_fall": round(u_best.get("time_to_fall", 0.0), 4),
        "dynamic_best_time_to_fall": round(d_best.get("time_to_fall", 0.0), 4),
        "uniform_best_early_forward_velocity": round(u_best.get("early_window_forward_velocity", 0.0), 6),
        "dynamic_best_early_forward_velocity": round(d_best.get("early_window_forward_velocity", 0.0), 6),
        "uniform_success_rate": round(u_successes / max(len(uniform_episode_log), 1), 4),
        "dynamic_success_rate": round(d_successes / max(len(dynamic_episode_log), 1), 4),
        "uniform_best_params": {
            "HL": u_best.get("hl"),
            "Ls": u_best.get("ls"),
            "Lswb": u_best.get("lswb"),
            "Lforward": u_best.get("lforward"),
        },
        "dynamic_best_params": {
            "HL": d_best.get("hl"),
            "Ls": d_best.get("ls"),
            "Lswb": d_best.get("lswb"),
            "Lforward": d_best.get("lforward"),
        },
        "target_reward_threshold": target_reward_threshold,
        "target_forward_velocity_threshold": target_forward_velocity_threshold,
        "uniform_episodes_to_target_reward": compute_episodes_to_target(
            uniform_episode_log, "total_reward", target_reward_threshold
        ),
        "dynamic_episodes_to_target_reward": compute_episodes_to_target(
            dynamic_episode_log, "total_reward", target_reward_threshold
        ),
        "uniform_episodes_to_target_velocity": compute_episodes_to_target(
            uniform_episode_log, "avg_forward_velocity", target_forward_velocity_threshold
        ),
        "dynamic_episodes_to_target_velocity": compute_episodes_to_target(
            dynamic_episode_log, "avg_forward_velocity", target_forward_velocity_threshold
        ),
        "efficiency_improvement_percent": round(improvement, 2),
        "convergence_criteria": {
            "window": 20,
            "metric": "coefficient_of_variation",
            "threshold": 0.05,
            "description": "CV of total_reward over sliding 20-episode window < 5%",
        },
    }
    report["dynamic_search_assessment"] = _assess_dynamic_search(report)
    return report
