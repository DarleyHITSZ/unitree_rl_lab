# MSLPO — 多阶段步态姿态优化

基于 IsaacLab / Isaac Sim 的 Unitree G1-29dof 双足步态强化学习框架。

**第一阶段** 搜索最优 SIMBICON 步态参数并构建离散姿态库。
**第二阶段** 在姿态库上训练分类 PPO 策略。

---

## 第一阶段 — 姿态库生成

第一阶段包含两条并行准备的流水线，最终融合为统一的姿态库：

```
                     图例 ───────────────────────────
                       [脚本] 可执行脚本    [输出] 生成文件
                       [数据] 输入源        ╔══╗  阶段分组
                                            ╚══╝
                      ────────────────────────────────

    ╔══════════════════════════╗        ╔══════════════════════════╗
    ║ [组] SIMBICON 参数搜索    ║        ║ [组] CMU 动作捕捉预处理   ║
    ║      (需要 Isaac Sim)    ║        ║      (离线)              ║
    ╚══════════╤═══════════════╝        ╚══════════╤═══════════════╝
               │                                   │
    [脚本] qlearn_search.py            [数据] ASF + AMC 文件
    [脚本] dynamic_discretization                  │
           _search.py                  [脚本] batch_convert_cmu.py
               │                                   │
    ┌──────────┴──────────────┐        ┌───────────┴──────────────┐
    │ [输出] top5_pose_params  │        │ [输出] processed/*.npz   │
    │        *.json            │        │        conversion_summary│
    └──────────┬──────────────┘        └───────────┬──────────────┘
               │                                   │
               │                       [脚本] build_human_g1_candidates.py
               │                                   │
               │                       ┌───────────┴──────────────┐
               │                       │ [输出] human_g1_candidate │
               │                       │        _poses.npy (240,29)│
               │                       └───────────┬──────────────┘
               │                                   │
    ┌──────────┴──────────────┐                    │
    │ [脚本] build_stage1      │                    │
    │        _pose_library.py  │                    │
    └──────────┬──────────────┘                    │
               │                                   │
    ┌──────────┴──────────────┐                    │
    │ [输出] pose_library.npy  │                    │
    │        (160, 29)         │                    │
    └──────────┬──────────────┘                    │
               │                                   │
               └───────────────┬───────────────────┘
                               │
                  [脚本] fuse_pose_library.py (离线)
                               │
                  ┌────────────┴─────────────┐
                  │ [输出] pose_library       │
                  │        _expanded.npy      │
                  │        (~216, 29)          │
                  └──────────────────────────┘
```

### 1A — 动作捕捉预处理（离线，无需 Isaac Sim）

所有脚本位于 `scripts/mocap/`。

#### 转换 CMU ASF/AMC 为 NPZ

```bash
python scripts/mocap/batch_convert_cmu.py convert \
    --input_dir data/human_gait/subject_07 \
    --output_dir data/human_gait/processed/cmu_subject_07 \
    --subject_id subject_07
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--input_dir` | `data/human_gait/subject_07` | 包含 `.asf` 和 `.amc` 文件的目录 |
| `--output_dir` | `data/human_gait/processed/cmu_subject_07` | `.npz` 输出目录 |
| `--subject_id` | `subject_07` | 被试者标识 |
| `--fps` | `120.0` | 帧率（CMU 标准：120 Hz） |

**输出：** `--output_dir` 下每个 `.amc` 对应一个 `.npz` 文件，以及 `conversion_summary.json`。

---

#### 构建人体→G1 候选姿态

```bash
python scripts/mocap/build_human_g1_candidates.py \
    --input_dir data/human_gait/processed/cmu_subject_07 \
    --output_dir outputs/pose_library
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--input_dir` | `data/human_gait/processed/cmu_subject_07` | `.npz` 文件目录 |
| `--output_dir` | `outputs/pose_library` | 输出目录 |
| `--num_phases` | `16` | 每步态周期的相位数（需与第一阶段一致） |
| `--skip_analysis` | `False` | 跳过 CMU 轴重要性分析 |
| `--skip_plots` | `False` | 跳过 matplotlib 可视化 |
| `--analyze_only` | `False` | 仅运行轴分析后退出 |

**输出：**
- `human_g1_candidate_poses.npy` — 形状 `(240, 29)` float32，G1-29dof 关节位置（弧度）
- `human_g1_candidate_poses_meta.json` — 每帧姿态元数据（来源、周期、相位、clip_ratio、支撑脚）
- `plots/` — 对比图、离散化图和关节限位分布图（除非 `--skip_plots`）

---

### 1B — SIMBICON 步态参数搜索（需要 Isaac Sim）

所有脚本位于 `scripts/mslpo/`。这些脚本通过 `AppLauncher` 启动 Isaac Sim。

#### 调试 / 验证控制器

```bash
python scripts/mslpo/simbicon_debug.py --num_envs 4 --headless --max_steps 2000
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--num_envs` | `4` | 并行环境数量 |
| `--max_steps` | `2000` | 最大仿真步数 |
| `--print_interval` | `50` | 每 N 步打印调试信息 |

---

#### 均匀离散 Q-learning 参数搜索

```bash
python scripts/mslpo/qlearn_search.py \
    --num_envs 1 --max_steps 2000 --episodes 1000 --headless
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--num_envs` | `1` | 并行环境数量 |
| `--max_steps` | `2000` | 每回合最大仿真步数 |
| `--episodes` | `1000` | Q-learning 总回合数 |
| `--alpha` | `0.1` | Q-learning 学习率 |
| `--gamma` | `0.95` | 折扣因子 |
| `--epsilon_start` | `1.0` | 初始探索率 |
| `--epsilon_end` | `0.05` | 最小探索率 |
| `--epsilon_decay` | `0.995` | 每回合探索率衰减 |
| `--k_v` | `1.0` | 速度奖励系数 |
| `--k_y` | `3.0` | 横向偏移惩罚系数 |
| `--k_alive` | `10.0` | 存活奖励系数 |
| `--top_k` | `5` | 保留的最优参数组数量 |
| `--save_interval` | `50` | 每 N 回合保存检查点 |
| `--output_dir` | `outputs/qlearn_search` | 输出目录 |
| `--checkpoint` | `None` | 从 Q 表检查点恢复训练 |

**输出：**
- `top5_pose_params.json` — Top-5 参数组 `(HL, Ls, Lswb, Lforward)` 及其奖励
- `q_table_checkpoint.json` — Q 表（可恢复）
- `episode_log.json` — 每回合结果

---

#### 动态离散化搜索（含均匀对比）

```bash
python scripts/mslpo/dynamic_discretization_search.py \
    --episodes 200 --sensitivity_scan outputs/qlearn_search/parameter_sensitivity_scan.json --headless
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--episodes` | `200` | 每次搜索的回合数（动态 + 均匀） |
| `--sensitivity_scan` | `None` | 已有敏感性扫描 JSON 文件路径 |
| `--scan_only` | `False` | 仅运行敏感性扫描后退出 |
| `--compare` | `False` | 运行均匀离散对比并生成报告 |
| `--num_rollouts` | `3` | 每扫描点的采样次数（仅内联扫描） |
| `--sensitivity_score_metric` | `composite` | 评分方式：`composite`、`total_reward`、`time_to_fall`、`early_forward_velocity` |
| `--target_reward_threshold` | `15.0` | 收敛跟踪目标奖励 |
| `--target_forward_velocity_threshold` | `0.33` | 收敛跟踪目标速度 |
| `--output_dir` | `outputs/dynamic_discretization` | 输出目录 |

**输出：**
- `top5_pose_params_dynamic.json` — 动态搜索 Top-5 参数组
- `dynamic_episode_log.json` — 每回合日志
- `dynamic_discretization_config.json` — 各参数重点区间与离散值
- `dynamic_vs_uniform_report.json` — 均匀 vs 动态对比报告
- `uniform_episode_log.json` — 均匀搜索日志（如果启用 `--compare`）
- `parameter_sensitivity_scan.json` — 敏感性扫描结果（如果内联运行）

---

### 1C — 姿态库构建（离线）

#### 构建第一阶段姿态库（160 个姿态）

```bash
python scripts/mslpo/build_stage1_pose_library.py
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--top5` | `outputs/dynamic_discretization/top5_pose_params_dynamic.json` | Top-5 参数 JSON 路径 |
| `--output_dir` | `outputs/pose_library` | 输出目录 |
| `--num_phases` | `16` | 每 FSM 状态的相位采样数 |
| `--core_states` | `STEP_RIGHT_WITH_LEFT_FRONT STEP_LEFT_WITH_RIGHT_FRONT` | 包含的 FSM 状态 |
| `--skip_plots` | `False` | 跳过可视化 |

**输出：**
- `pose_library.npy` — 形状 `(160, 29)` float32，= 5 参数组 × 2 FSM 状态 × 16 相位
- `pose_library_meta.json` — 每姿态元数据（param_group_idx、fsm_state、phase_value、action_idx）
- `plots/` — 各组相位曲线、热力图、组间对比（除非 `--skip_plots`）

---

#### 融合：第一阶段 + 人类候选 → 扩充姿态库

```bash
python scripts/mslpo/fuse_pose_library.py
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--stage1_dir` | `outputs/pose_library` | 包含 `pose_library.npy` 的目录 |
| `--human_dir` | `outputs/pose_library` | 包含 `human_g1_candidate_poses.npy` 的目录 |
| `--output_dir` | `outputs/pose_library` | 输出目录 |
| `--dedup_s1_threshold` | `0.40` | 第一阶段 vs 人类去重阈值（加权 L2，弧度） |
| `--dedup_hh_threshold` | `0.25` | 人类 vs 人类同相位去重阈值 |
| `--max_per_phase` | `3` | 每相位最大人类姿态数 |
| `--max_human_total` | `80` | 人类姿态总数硬上限 |
| `--skip_plots` | `False` | 跳过可视化 |

**融合流程阶段：** 有效性过滤 → 形态学过滤 → 第一阶段去重 → 人类去重 → 相位预算分配

**输出：**
- `pose_library_expanded.npy` — 形状 `(~216, 29)` float32，160 个第一阶段 + ~56 个人类
- `pose_library_expanded_meta.json` — 每姿态元数据，含 `source` 字段（`stage1` / `human_cmu`）
- `pose_library_expanded_report.json` — 完整融合报告（过滤统计、评分分布、预算）
- `nearest_stage1_distances.json` — 人类→第一阶段距离诊断
- `plots/` — 关节叠加图、相位覆盖图、评分分布图、漏斗图

---

### 1D — 分析报告（离线）

```bash
python scripts/mslpo/build_stage1_analysis_report.py
```

生成包含数据表、统计和论文级插图的第一阶段综合结果报告。

**输出：** `outputs/analysis_stage1/` — CSV、JSON、Markdown 报告及 16 张插图（PNG + PDF）

---

## 第二阶段 — 姿态库上的 PPO 训练

第二阶段训练一个**分类 PPO** 策略。每个仿真步策略选择一个离散动作索引，映射到扩充姿态库中的 29 维关节位置目标。

### 训练

```bash
python scripts/mslpo/stage2_ppo_train.py \
    --task Unitree-G1-29dof-Stage2-PPO \
    --num_envs 4096 --headless --max_iterations 50000
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--num_envs` | `None`（使用环境配置默认值） | 并行环境数量 |
| `--max_iterations` | `50000` | 训练迭代次数 |
| `--num_steps_per_env` | `24` | 每环境每次迭代的 rollout 步数 |
| `--lr` | `3e-4` | 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--gae_lambda` | `0.95` | GAE lambda |
| `--clip_ratio` | `0.2` | PPO 裁剪率 |
| `--entropy_coef` | `0.01` | 熵奖励系数 |
| `--value_coef` | `0.5` | 价值损失系数 |
| `--max_grad_norm` | `1.0` | 最大梯度范数 |
| `--update_epochs` | `5` | 每次迭代 PPO 更新轮数 |
| `--num_mini_batches` | `4` | 每次更新 Mini-batch 数量 |
| `--actor_hidden_dims` | `256 256 128` | Actor MLP 隐藏层尺寸 |
| `--critic_hidden_dims` | `256 256 128` | Critic MLP 隐藏层尺寸 |
| `--activation` | `elu` | 激活函数 |
| `--pose_library_path` | `outputs/pose_library/pose_library_expanded.npy` | 训练用姿态库 |
| `--save_interval` | `500` | 每 N 次迭代保存检查点 |
| `--print_interval` | `100` | 每 N 次迭代打印进度 |
| `--seed` | `42` | 随机种子 |
| `--checkpoint` | `None` | 从 `.pt` 检查点恢复训练 |
| `--output_dir` | `outputs/stage2_ppo` | 输出目录 |

**输出**（在 `outputs/stage2_ppo/<时间戳>/`）：
- `stage2_ppo_final.pt` — 最终模型检查点（策略 + 优化器 + 配置）
- `stage2_ppo_<iter>.pt` — 周期性检查点
- `config.json` — 训练配置
- TensorBoard 日志（如可用）

---

### 评估

```bash
python scripts/mslpo/stage2_ppo_play.py \
    --checkpoint outputs/stage2_ppo/.../stage2_ppo_final.pt \
    --num_envs 32 --num_episodes 100
```

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--checkpoint` | **必填** | `.pt` 检查点路径 |
| `--num_envs` | `32` | 并行环境数量 |
| `--num_episodes` | `100` | 评估总回合数（无头模式） |
| `--pose_library_path` | `outputs/pose_library/pose_library_expanded.npy` | 与训练所用相同的姿态库 |
| `--video` | `False` | 录制评估视频 |
| `--video_length` | `200` | 视频步数 |
| `--real_time` | `False` | 实时速度运行（GUI 模式） |

**无头模式** 打印：平均奖励、前向速度、横向偏移、存活时间、摔倒率。
评估结果保存至检查点同目录下的 `evaluation_results.json`。

**GUI 模式**（省略 `--headless`）：交互式可视化，关闭窗口后退出。
