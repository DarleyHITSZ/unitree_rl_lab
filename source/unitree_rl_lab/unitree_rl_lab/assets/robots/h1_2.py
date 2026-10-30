"""Official mjlab H1_2 model and implicit position actuator parameters."""

import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim.core.utils.extensions import enable_extension
from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg

H1_2_MODEL_PATH = Path(__file__).with_suffix("") / "h1_2.xml"
H1_2_MODEL = ET.parse(H1_2_MODEL_PATH).getroot()
H1_2_JOINT_NAMES = [joint.attrib["name"] for joint in H1_2_MODEL.findall(".//body/joint")]
H1_2_COLLISION_BODIES = [
    body.attrib["name"]
    for body in H1_2_MODEL.findall(".//body")
    if any(geom.get("class") in ("collision", "foot_capsule") for geom in body.findall("geom"))
]
H1_2_EXCLUDED_PAIRS = {
    frozenset((pair.attrib["body1"], pair.attrib["body2"])) for pair in H1_2_MODEL.findall("contact/exclude")
}
for parent in H1_2_MODEL.findall(".//body"):
    for child in parent.findall("body"):
        H1_2_EXCLUDED_PAIRS.add(frozenset((parent.attrib["name"], child.attrib["name"])))


def spawn_h1_2(prim_path, cfg, translation=None, orientation=None):
    """Enable the MJCF importer before converting the official robot model."""
    enable_extension("isaacsim.asset.importer.mjcf")
    result = sim_utils.spawn_from_mjcf(prim_path, cfg, translation=translation, orientation=orientation)
    for root in sim_utils.find_matching_prims(prim_path):
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        bodies = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        for pair in H1_2_EXCLUDED_PAIRS:
            first, second = sorted(pair)
            if first in bodies and second in bodies:
                api = UsdPhysics.FilteredPairsAPI.Apply(bodies[first])
                api.CreateFilteredPairsRel().AddTarget(bodies[second].GetPath())
    return result


UNITREE_H1_2_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=str(H1_2_MODEL_PATH),
        func=spawn_h1_2,
        fix_base=False,
        self_collision=True,
        activate_contact_sensors=True,
        import_inertia_tensor=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.02),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.3,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_elbow_joint": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107_24_2": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", "torso_joint"],
            stiffness=98.7,
            damping=6.3,
            effort_limit_sim=200.0,
            armature=0.025,
        ),
        "M107_24_1": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            stiffness=157.7,
            damping=10.1,
            effort_limit_sim=300.0,
            armature=0.04,
        ),
        "GO2HV_1": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
            stiffness=19.7,
            damping=1.3,
            effort_limit_sim=40.0,
            armature=0.005,
        ),
        "GO2HV_2": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*"],
            stiffness=7.9,
            damping=0.5,
            effort_limit_sim=18.0,
            armature=0.002,
        ),
    },
    joint_sdk_names=H1_2_JOINT_NAMES.copy(),
)

H1_2_ACTION_SCALE = {
    name: 0.25 * actuator.effort_limit_sim / actuator.stiffness
    for actuator in UNITREE_H1_2_CFG.actuators.values()
    for name in actuator.joint_names_expr
}
