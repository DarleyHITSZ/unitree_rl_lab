# H1_2 官方 mjlab 方案迁移

现有任务 `Unitree-H1-2-Velocity` 已原地更新为宇树官方 mjlab 的 **27 关节平地速度跟踪方案**。没有增加第二个任务或配置变体。

## 启动

在仓库根目录执行：

```bash
conda activate env_isaacsim
./unitree_rl_lab.sh -t --task Unitree-H1-2-Velocity --num_envs 512
```

默认训练 10001 次 PPO 更新，每 100 次保存，目录为 `logs/rsl_rl/h1_2/`。显存允许时可增大并行环境数；配置默认值为 4096。

```bash
./unitree_rl_lab.sh -p --task Unitree-H1-2-Velocity --num_envs 4
./unitree_rl_lab.sh -t --task Unitree-H1-2-Velocity --num_envs 512 --resume
```

旧的 12 关节 LSTM 检查点与当前网络、观测和动作维度不兼容，需要重新训练。
使用 `--load_run` 或 `--checkpoint /absolute/path/to/model_N.pt` 选择当前方案生成的检查点，避免误选旧训练结果。
演示配置遵循官方设置：关闭观测噪声和推扰、禁用课程，保留启动时的随机化，回合时长为 1e9 秒。
因此成熟策略的无界面演示可能长时间不结束。当前仓库保留离线数值测试和短程训练验证命令；迁移时使用的临时仿真检查脚本已删除。

## 官方来源

上游：[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)，固定提交：
`1425b15f73bd4095f0df53709d7c389c3eb9e790`。

对照文件：

- `src/tasks/velocity/velocity_env_cfg.py`
- `src/tasks/velocity/config/h1_2/env_cfgs.py`
- `src/tasks/velocity/config/h1_2/rl_cfg.py`
- `src/tasks/velocity/mdp/`
- `src/assets/robots/unitree_h1_2/h1_2_constants.py`
- `src/assets/robots/unitree_h1_2/xmls/h1_2.xml`

还核对了其固定依赖 `mjlab==1.2.0` 的编码器偏置、动作、观测及 PPO 默认设置。

## 配置对应关系

| 项目 | 当前配置 |
|---|---|
| 关节与动作 | 27：双腿 12、躯干 1、双臂和腕部 14 |
| 策略输入 | 92 维，单帧，无历史堆叠 |
| Critic 输入 | 107 维，增加 IMU 线速度及足部特权观测 |
| 网络 | Actor、Critic 均为 MLP `[512, 256, 128]`，ELU，开启各自观测归一化 |
| 初始动作噪声 | 1.0 |
| PPO | 自适应学习率 1e-3、clip 0.2、熵 0.01、gamma 0.99、GAE lambda 0.95 |
| 更新 | 每环境 24 步，5 个 epoch、4 个 minibatch，目标 KL 0.01，梯度裁剪 1.0 |
| 仿真与控制 | 0.005 秒、降采样 4，策略频率 50 Hz |
| 回合与步态 | 训练回合 20 秒，步态周期 0.6 秒，左右相位偏移 0.5 |
| 随机种子 | 42，来自 mjlab 1.2.0 的默认配置 |

策略观测顺序为：躯干 IMU 角速度、骨盆投影重力、速度命令、正弦/余弦相位、27 个关节位置偏移、27 个关节速度、27 个上一步动作。
Critic 按同一顺序接收无噪声输入，再增加 IMU 线速度 3、脚底高度 2、腾空时间 2、接触标志 2、接触力 6。
站立命令下相位置零；接触力使用带符号的 `log1p(abs(force))`。

| 执行器组 | 刚度 | 阻尼 | 力矩上限 | Armature |
|---|---:|---:|---:|---:|
| 髋与躯干 | 98.7 | 6.3 | 200 | 0.025 |
| 膝 | 157.7 | 10.1 | 300 | 0.04 |
| 踝、肩 pitch/roll | 19.7 | 1.3 | 40 | 0.005 |
| 肩 yaw、肘、腕 | 7.9 | 0.5 | 18 | 0.002 |

每组动作缩放均为 `0.25 * effort_limit / stiffness`，目标位置相对 `robot.data.default_joint_pos`。
保留官方编码器偏置语义：启动时采样 +/-0.015 rad，执行目标减去偏置；位置观测默认不叠加偏置。

## 奖励、课程与随机化

完整保留 16 项奖励：线速度跟踪、角速度跟踪、躯干姿态、随命令速度变化的关节姿态容差、躯干角速度、躯干子树角动量、摔倒终止、关节加速度、关节限位、动作变化、步态支撑、脚底离地高度、滑动、落地冲击、静止姿态、自碰撞。
保留终止权重 -200，取消旧方案的总奖励非负截断。记录角动量、滑动速度、落地力和动作二阶差分诊断指标。

- 命令重采样 3–8 秒，5% 站立，航向误差反馈系数 0.5。
- 速度课程在全局环境步超过 0 和 120000 时切换范围；后者对应 5000 次、每次 24 步的 PPO 更新。
- 脚部摩擦 0.3–1.6，同一环境的双脚共享一个采样值。
- 躯干 COM 三轴偏移 +/-0.05 米；无旧方案的质量随机化。
- 每 5–6 秒进行六轴速度推扰，使用官方各轴范围。
- 重置时根位置 XY +/-0.5 米、yaw +/-3.14；关节为默认位置，速度为零。
- 训练终止条件为 20 秒超时或倾斜超过 70 度；平地任务没有地形课程。

## IsaacLab 适配

- 直接导入官方 MJCF，使用对应的 28 个网格；模型目录为 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/h1_2/`。运行时无需 mjlab 或外部 Gym 仓库。
- 为兼容 Isaac Sim MJCF 导入器，将继承的视觉/碰撞几何属性展开为显式属性，网格目录改为 `meshes`，移除生成 USD 中多余的非刚体 articulation root。
- 使用 PhysX 隐式位置驱动，保留上游刚度、阻尼、力矩上限、armature、关节和惯性参数；转移 MJCF 显式碰撞排除对和相邻链接排除。
- IMU 使用躯干上的实际偏移；脚底使用踝 roll link 上 `(0.04, 0, -0.04)` 采样点，速度包含旋转产生的线速度。
- `root_angmom` 按上游 MJCF 定义计算躯干及双臂子树相对其 COM 的角动量，不包含骨盆和腿部。惯性张量按 PhysX 文档使用 link actor frame。
- 每个足部及自碰撞链接单独建立过滤接触传感器；接触力包含法向和切向力，接触时长与落地判定仅依据地面过滤后的数据。
- 地面采用固定姿态的运动学地板，尺寸 2000×2000×0.2 米，顶面 z=0，以适配当前 GPU 接触过滤；训练区域和正常回合行走范围均位于其内部。

MuJoCo 的接触维度、摩擦混合、求解器和 PhysX 不完全等价。因此这里迁移的是完整训练定义及其数值参数，不能保证逐步轨迹或训练曲线与 mjlab 相同；仍需重新训练并评价收敛。

## 验证命令

```bash
python -m unittest discover -s tests -p test_h1_2_mdp.py
./unitree_rl_lab.sh -t --task Unitree-H1-2-Velocity --num_envs 512 --max_iterations 15 --run_name mjlab_verified
```

迁移时的仿真检查覆盖 92/107 维观测、27 维动作、全部奖励、自碰撞传感器注册、真实地面接触、自动超时重置，以及强制倾斜后的负终止奖励。临时检查脚本、检查点和日志已在清理时删除；当前演示与恢复训练需先生成新的检查点。
短程验证只证明流程可运行，不代表策略已收敛。

2026-10-30 在 `env_isaacsim`（Isaac Sim 5.1.0、RSL-RL 3.1.2）完成：20 项离线测试；训练/演示配置各 4 环境、100 步与强制终止检查；512 环境、15 次 PPO 更新（184320 环境步）；检查点加载与 8 回合演示；TorchScript/ONNX 导出。Black、isort、flake8、Pyright 检查通过。
