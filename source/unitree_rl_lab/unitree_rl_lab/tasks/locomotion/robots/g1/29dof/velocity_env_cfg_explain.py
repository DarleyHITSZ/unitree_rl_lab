import math
# 导入Isaac Lab仿真核心工具库
import isaaclab.sim as sim_utils
# 导入地形生成相关模块
import isaaclab.terrains as terrain_gen
# 导入资产配置类：关节链资产（机器人）、基础资产
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# 导入基于管理器的RL环境配置基类
from isaaclab.envs import ManagerBasedRLEnvCfg
# 导入MDP核心组件配置类：课程项、事件项、观测组、观测项、奖励项、终止项
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
# 导入交互式场景配置基类
from isaaclab.scene import InteractiveSceneCfg
# 导入传感器配置类：接触力传感器、激光雷达（高度扫描仪）、扫描模式
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# 导入地形导入配置类
from isaaclab.terrains import TerrainImporterCfg
# 导入配置类装饰器（支持结构化配置解析）
from isaaclab.utils import configclass
# 导入Isaac Lab核心资源目录路径
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# 导入噪声配置类（用于观测噪声增强）
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# 导入Unitree G1机器人29自由度配置
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
# 导入 locomotion 任务的MDP工具函数（奖励、重置、事件等）
from unitree_rl_lab.tasks.locomotion import mdp

# 地形生成配置：鹅卵石路地形（以平坦为主，含轻微起伏）
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),          # 单块地形尺寸（8m×8m）
    border_width=20.0,        # 地形边界宽度（避免机器人跑出地形范围）
    num_rows=9,               # 地形网格行数（生成9行子地形）
    num_cols=21,              # 地形网格列数（生成21列子地形）
    horizontal_scale=0.1,     # 水平缩放系数（控制地形细节密度）
    vertical_scale=0.005,     # 垂直缩放系数（控制地形起伏幅度，最大~4cm）
    slope_threshold=0.75,     # 坡度阈值（过滤过陡地形，保证机器人可行走）
    difficulty_range=(0.0, 1.0),  # 地形难度范围（0=最简单，1=最复杂）
    use_cache=False,          # 不使用缓存（每次生成新地形，增加多样性）
    sub_terrains={
        # 子地形配置：50%比例的平坦地形
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """机器人场景配置类：定义仿真世界的物理实体（地形、机器人、传感器、灯光）"""
    # 地面地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在USD场景中的路径
        terrain_type="generator",   # 地形类型：基于生成器（另可选"plane"平面）
        terrain_generator=COBBLESTONE_ROAD_CFG,  # 关联上述地形生成规则
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,  # 初始最大地形难度等级
        collision_group=-1,         # 碰撞组ID（-1表示默认组）
        # 物理材质配置（地面摩擦、恢复系数）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦组合模式：相乘
            restitution_combine_mode="multiply",  # 恢复系数组合模式：相乘
            static_friction=1.0,  # 静摩擦系数（地面抓地力）
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        # 视觉材质配置（地面纹理）
        visual_material=sim_utils.MdlFileCfg(
            # 纹理文件路径（Isaac Lab内置大理石纹理）
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,  # 启用UVW投影（保证纹理正确映射）
            texture_scale=(0.25, 0.25),  # 纹理缩放比例（控制纹理密度）
        ),
        debug_vis=False,  # 关闭地形调试可视化
    )

    # 机器人配置：复用Unitree G1预定义配置，修改路径适配多环境
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描仪（激光雷达）配置：用于感知前方地形高度
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 挂载在机器人躯干
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 挂载偏移量（z轴20cm，避免碰撞）
        ray_alignment="yaw",  # 射线对齐方式：偏航角对齐
        # 扫描模式：网格扫描（分辨率0.1m，扫描范围1.6m×1.0m）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,  # 关闭扫描调试可视化
        mesh_prim_paths=["/World/ground"],  # 扫描目标：地面地形
    )

    # 接触力传感器配置：监测机器人所有关节的接触力
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 匹配机器人所有关节
        history_length=3,  # 记录3步接触力历史数据
        track_air_time=True,  # 跟踪关节离地时间（用于步态判断）
    )

    # 天空光配置：模拟自然光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 灯光在USD场景中的路径
        # 穹顶光配置（HDR纹理+强度）
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光照强度
            # HDR天空纹理路径（Isaac Lab内置高清天空纹理）
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class EventCfg:
    """事件配置类：定义仿真过程中的触发式动作（启动/重置/间隔触发）"""
    # 启动事件：随机化机器人身体的物理材质（增加训练鲁棒性）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 物理材质随机化函数
        mode="startup",  # 触发模式：环境启动时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 作用对象：机器人所有身体
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数随机范围
            "dynamic_friction_range": (0.3, 1.0),  # 动摩擦系数随机范围
            "restitution_range": (0.0, 0.0),  # 恢复系数（固定为0，无弹性）
            "num_buckets": 64,  # 采样桶数量（控制随机粒度）
        },
    )

    # 启动事件：给机器人躯干添加随机质量（模拟负载变化）
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # 质量随机化函数
        mode="startup",  # 触发模式：环境启动时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 作用对象：机器人躯干
            "mass_distribution_params": (-1.0, 3.0),  # 质量增量范围（-1~3kg）
            "operation": "add",  # 操作类型：加法（在原有质量上增减）
        },
    )

    # 重置事件：给机器人基座施加随机外力/力矩（初始扰动）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,  # 施加外力/力矩函数
        mode="reset",  # 触发模式：episode重置时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 作用对象：机器人躯干
            "force_range": (0.0, 0.0),  # 外力范围（默认0，可调整为扰动）
            "torque_range": (-0.0, 0.0),  # 外力矩范围（默认0，可调整为扰动）
        },
    )

    # 重置事件：重置机器人基座状态（位置、姿态、速度）
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # 基座状态重置函数
        mode="reset",  # 触发模式：episode重置时
        params={
            # 位置随机范围：x/y轴±0.5m，偏航角±π（360°）
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 速度随机范围：所有方向速度为0（初始静止）
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # 重置事件：重置机器人关节状态（位置、速度）
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,  # 关节状态重置函数
        mode="reset",  # 触发模式：episode重置时
        params={
            "position_range": (1.0, 1.0),  # 关节位置范围（1.0=默认位置）
            "velocity_range": (-1.0, 1.0),  # 关节速度随机范围（±1 rad/s）
        },
    )

    # 间隔事件：每隔固定时间给机器人施加速度扰动（训练抗干扰能力）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # 速度扰动函数
        mode="interval",  # 触发模式：固定时间间隔
        interval_range_s=(5.0, 5.0),  # 间隔时间（固定5秒一次）
        params={
            # 速度扰动范围：x/y轴±0.5m/s（随机方向）
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class CommandsCfg:
    """命令配置类：定义机器人的学习目标（速度跟踪指令）"""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",  # 指令作用对象：机器人
        resampling_time_range=(10.0, 10.0),  # 指令重采样时间（每10秒更新一次目标速度）
        rel_standing_envs=0.02,  # 2%的环境固定为"站立"（目标速度为0）
        rel_heading_envs=1.0,  # 100%的环境启用航向指令（与速度指令匹配）
        heading_command=False,  # 不单独发送航向指令（融入速度指令）
        debug_vis=True,  # 启用指令调试可视化（显示目标速度）
        # 目标速度范围（训练时的基础范围）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),  # x轴（前后）线速度范围
            lin_vel_y=(-0.1, 0.1),  # y轴（左右）线速度范围
            ang_vel_z=(-0.1, 0.1),  # z轴（转向）角速度范围
        ),
        # 目标速度限制（最大允许速度，防止失控）
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),  # x轴最大速度（后退0.5m/s，前进1.0m/s）
            lin_vel_y=(-0.3, 0.3),  # y轴最大速度（±0.3m/s）
            ang_vel_z=(-0.2, 0.2),  # z轴最大角速度（±0.2 rad/s）
        ),
    )

@configclass
class ActionsCfg:
    """动作配置类：定义机器人的控制方式（关节位置控制）"""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",  # 动作作用对象：机器人
        joint_names=[".*"],  # 控制所有关节
        scale=0.25,  # 动作缩放系数（降低关节运动灵敏度，避免突变）
        use_default_offset=True,  # 使用关节默认偏移（保证初始姿态正确）
    )

@configclass
class ObservationsCfg:
    """观测配置类：定义智能体可感知的环境信息（策略/评论家网络输入）"""
    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组：用于生成动作的观测数据（含噪声增强鲁棒性）"""
        # 观测项1：基座角速度（roll/pitch/yaw），缩放+噪声
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 观测项2：投影重力（判断机器人姿态稳定性），添加噪声
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 观测项3：速度指令（目标速度，来自CommandsCfg）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 观测项4：关节相对位置（相对于默认姿态的偏差），添加噪声
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 观测项5：关节相对速度，缩放+噪声
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 观测项6：上一步动作（保证动作平滑性）
        last_action = ObsTerm(func=mdp.last_action)
        # 注释：步态相位观测（可根据需求启用）
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            """初始化后配置：观测组全局设置"""
            self.history_length = 5  # 历史观测长度（拼接前5步数据，捕捉时间序列）
            self.enable_corruption = True  # 启用观测噪声（增强泛化性）
            self.concatenate_terms = True  # 所有观测项拼接为一维向量（适配神经网络）

    # 注册策略观测组（智能体决策时使用）
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论家网络观测组：用于评估动作价值（无噪声，保证评估准确性）"""
        # 观测项1：基座线速度（x/y/z）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 观测项2：基座角速度（roll/pitch/yaw），缩放
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        # 观测项3：投影重力
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 观测项4：速度指令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 观测项5：关节相对位置
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 观测项6：关节相对速度，缩放
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        # 观测项7：上一步动作
        last_action = ObsTerm(func=mdp.last_action)
        # 注释：步态相位观测（可启用）
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # 注释：高度扫描仪观测（可启用，增强地形感知）
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            """初始化后配置：观测组全局设置"""
            self.history_length = 5  # 历史观测长度（拼接前5步数据）

    # 注册评论家观测组（价值评估时使用）
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """奖励配置类：定义智能体的奖惩规则（引导学习目标行为）"""
    # 核心任务奖励：跟踪xy平面线速度（高斯分布，误差越小奖励越高）
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,  # 奖励权重（核心任务，权重最高）
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # std=0.5（高斯分布参数）
    )

    # 核心任务奖励：跟踪z轴角速度（高斯分布）
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,  # 奖励权重（次要核心任务）
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 存活奖励：每步存活得0.15分（鼓励保持稳定）
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # 基座惩罚项：z轴（垂直）线速度（惩罚跳跃/塌陷）
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 基座惩罚项：xy平面角速度（惩罚倾斜/晃动）
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 关节惩罚项：关节速度（惩罚高速运动，减少能耗）
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # 关节惩罚项：关节加速度（惩罚急加速，保证运动平滑）
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 动作惩罚项：动作速率（惩罚动作突变，避免控制震荡）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # 关节惩罚项：关节限位（惩罚接近机械限位，避免损伤）
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # 能耗惩罚项：机器人能耗（惩罚高能耗运动）
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # 关节偏差惩罚项：手臂关节（惩罚偏离默认姿态）
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,  # 权重较低（手臂非核心运动关节）
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",  # 肩膀所有关节
                    ".*_elbow_joint",        # 肘部关节
                    ".*_wrist_.*",           # 手腕所有关节
                ],
            )
        },
    )

    # 关节偏差惩罚项：腰部关节（惩罚偏离默认姿态，权重较高）
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],  # 腰部所有关节
            )
        },
    )

    # 关节偏差惩罚项：腿部关节（髋部侧倾/偏航关节，惩罚偏离默认姿态）
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # 姿态惩罚项：躯干平坦度（惩罚倾斜，鼓励水平姿态）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # 姿态惩罚项：基座高度（惩罚偏离目标高度0.78m）
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # 脚部奖励项：步态（鼓励按周期运动，周期0.8s）
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,  # 步态周期（0.8秒一步）
            "offset": [0.0, 0.5],  # 步态偏移（左右脚交替）
            "threshold": 0.55,  # 着地判断阈值
            "command_name": "base_velocity",  # 关联速度指令（步态与速度匹配）
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 脚部接触传感器
        },
    )

    # 脚部惩罚项：脚部滑动（惩罚着地时滑动，保证稳定）
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 脚部关节
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 接触传感器
        },
    )

    # 脚部奖励项：离地间隙（鼓励脚部离地间隙接近0.1m，避免蹭地）
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,  # 高斯分布参数（误差容忍度）
            "tanh_mult": 2.0,  # tanh缩放系数（增强奖励区分度）
            "target_height": 0.1,  # 目标离地间隙（0.1m）
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 脚部关节
        },
    )

    # 其他惩罚项：非期望接触（惩罚躯干/手臂等非脚部接触地面，避免摔倒）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,  # 接触力阈值（超过则判定为非期望接触）
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),  # 排除脚部关节
        },
    )

@configclass
class TerminationsCfg:
    """终止配置类：定义episode结束条件（满足任一即重置）"""
    # 超时终止：达到最大episode长度（20秒，见RobotEnvCfg）
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 基座高度终止：基座高度低于0.2m（机器人摔倒）
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # 姿态终止：姿态角度超过0.8rad（≈45.8°，严重倾斜）
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

@configclass
class CurriculumCfg:
    """课程配置类：定义训练难度递进规则（逐步提高任务难度）"""
    # 地形难度递进：随着训练推进，地形难度从低到高
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 速度指令难度递进：随着训练推进，目标速度范围从窄到宽
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """机器人速度跟踪任务总配置类：整合所有子配置，定义仿真核心参数"""
    # 场景配置：关联上述场景定义（地形、机器人、传感器、灯光）
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基础配置：关联动作、观测、命令配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP配置：关联奖励、终止、事件、课程配置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """初始化后配置：补充仿真、传感器、课程相关参数"""
        # 通用设置：下采样率（每4步仿真输出1次观测，降低计算量）
        self.decimation = 4
        # episode长度：20秒（超过则超时终止）
        self.episode_length_s = 20.0

        # 仿真设置：物理步长0.005秒（200Hz，兼顾精度和速度）
        self.sim.dt = 0.005
        # 渲染间隔：与下采样率一致（每4步渲染1次，提升可视化流畅度）
        self.sim.render_interval = self.decimation
        # 仿真物理材质：复用地形的物理材质配置
        self.sim.physics_material = self.scene.terrain.physics_material
        # PhysX GPU参数：最大刚性体补丁数（适配多环境并行仿真）
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器更新周期：接触力传感器每步更新（0.005秒）
        self.scene.contact_forces.update_period = self.sim.dt
        # 高度扫描仪更新周期：每4步更新（与下采样率一致）
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 课程学习开关：如果启用地形难度课程，开启地形生成器的课程模式
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """测试/演示用环境配置：继承训练环境，优化可视化和测试体验"""
    def __post_init__(self):
        super().__post_init__()  # 继承父类的初始化配置
        self.scene.num_envs = 32  # 并行环境数减少为32（降低资源占用）
        # 地形简化：减少行列数（2行10列），降低地形复杂度
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # 速度指令：直接使用最大限制范围（演示机器人最大性能）
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
