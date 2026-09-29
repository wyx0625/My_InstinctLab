"""G1 圆洞动作模仿任务配置。"""

import math
import os

import yaml
from isaaclab.envs import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

import instinctlab.tasks.shadowing.mdp as shadowing_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets import CircularHoleObstacleCfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_PELVISBASE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.motion_reference.motion_files.amass_motion_cfg import (
    AmassMotionCfg as AmassMotionCfgBase,
)
from instinctlab.motion_reference.motion_reference_cfg import NoCollisionPropertiesCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets

from .perceptive_shadowing_cfg import (
    motion_reference_cfg as perceptive_motion_reference_cfg,
)

G1_CFG = G1_29DOF_PELVISBASE_CFG
G1_REFERENCE_CFG = G1_CFG.copy()
G1_REFERENCE_CFG.spawn.collision_props = NoCollisionPropertiesCfg()
G1_REFERENCE_CFG.spawn.rigid_props.disable_gravity = True
G1_REFERENCE_CFG.spawn.activate_contact_sensors = False
G1_REFERENCE_CFG.actuators = {}
DEFAULT_MOTION_FOLDER = os.path.expanduser(
    "~/Wyx/Project_Instinct/motions/circular_hole_bound"
)
MOTION_FOLDER = os.environ.get(
    "INSTINCTLAB_CIRCULAR_HOLE_MOTION_DIR", DEFAULT_MOTION_FOLDER
)


def _load_obstacle_binding() -> dict:
    binding_path = os.path.join(MOTION_FOLDER, "circular_hole_binding.yaml")
    if not os.path.isfile(binding_path):
        return {
            "panel_width": 1.20,
            "panel_height": 1.20,
            "panel_thickness": 0.15,
            "hole_radius": 0.40,
            "hole_center_z": 0.70,
        }
    with open(binding_path, encoding="utf-8") as binding_file:
        return yaml.safe_load(binding_file)["obstacle"]


OBSTACLE_BINDING = _load_obstacle_binding()


@configclass
class CircularHoleMotionCfg(AmassMotionCfgBase):
    path = MOTION_FOLDER
    filtered_motion_selection_filepath = os.path.join(
        MOTION_FOLDER, "selected_motion.yaml"
    )
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = (0.0, 0.0)
    motion_start_height_offset = 0.0
    motion_bin_length_s = None
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "independent"


circular_hole_motion_reference_cfg = perceptive_motion_reference_cfg.replace(
    prim_path="{ENV_REGEX_NS}/Robot/pelvis",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
    motion_buffers={"CircularHoleMotion": CircularHoleMotionCfg()},
    visualizing_robot_offset=(0.0, 0.0, 0.0),
    visualizing_marker_types=["root", "links"],
)


@configclass
class G1CircularHoleSceneCfg(perceptual_cfg.PerceptiveShadowingSceneCfg):
    obstacle = CircularHoleObstacleCfg(
        prim_path="{ENV_REGEX_NS}/CircularHoleObstacle",
        panel_width=OBSTACLE_BINDING["panel_width"],
        panel_height=OBSTACLE_BINDING["panel_height"],
        panel_thickness=OBSTACLE_BINDING["panel_thickness"],
        hole_radius=OBSTACLE_BINDING["hole_radius"],
        hole_center_z=OBSTACLE_BINDING["hole_center_z"],
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )


@configclass
class G1CircularHoleShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    require_compatible_binding: bool = True

    scene: G1CircularHoleSceneCfg = G1CircularHoleSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=circular_hole_motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()

        binding_path = os.path.join(MOTION_FOLDER, "circular_hole_binding.yaml")
        if self.require_compatible_binding:
            if not os.path.isfile(binding_path):
                raise FileNotFoundError(f"缺少圆洞绑定元数据：{binding_path}")
            with open(binding_path, encoding="utf-8") as binding_file:
                binding = yaml.safe_load(binding_file)
            validation = binding.get("compatibility_validation")
            if validation is None:
                raise RuntimeError(
                    "圆洞动作尚未进行全身几何验证，请先运行 scripts/validate_circular_hole_binding.py。"
                )
            if validation["status"] != "compatible":
                exact_validation = validation.get("exact_collision_mesh", {})
                raise RuntimeError(
                    "圆洞动作与当前障碍物尺寸不兼容，禁止启动训练。"
                    f" 实网格穿透帧数={exact_validation.get('penetration_frame_count', 'unknown')}，"
                    f"径向余量={exact_validation.get('radial_clearance', float('nan')):.3f}m。"
                )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.sim.physx.enable_ccd = True
        self.scene.camera.update_period = 0.02
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        self.scene.camera.mesh_prim_paths.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/CircularHoleObstacle/geometry/visual"
            )
        )

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale
        self.observations.critic.link_pos.params[
            "asset_cfg"
        ].body_names = self.scene.motion_reference.link_of_interests
        self.observations.critic.link_rot.params[
            "asset_cfg"
        ].body_names = self.scene.motion_reference.link_of_interests

        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        self.events.push_robot = None
        self.events.reset_robot.params["randomize_pose_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.radians(5.0), math.radians(5.0)),
        }
        if self.require_compatible_binding:
            obstacle_contact_sensor_names = []
            for link_index, link_name in enumerate(G1_29DOF_LINKS):
                sensor_name = f"obstacle_contact_{link_index}"
                obstacle_contact_sensor_names.append(sensor_name)
                setattr(
                    self.scene,
                    sensor_name,
                    ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
                        filter_prim_paths_expr=[
                            "{ENV_REGEX_NS}/CircularHoleObstacle/geometry/collision"
                        ],
                        history_length=1,
                        track_contact_points=False,
                        max_contact_data_count_per_prim=32,
                    ),
                )
            self.terminations.obstacle_contact = DoneTermCfg(
                func=shadowing_mdp.obstacle_contact,
                params={
                    "sensor_names": obstacle_contact_sensor_names,
                    "threshold": 1.0,
                },
            )
        self.terminations.out_of_border = None
        self.run_name = "g1CircularHole_singleMotion_depth_initialPoseRandomization"


@configclass
class G1CircularHoleShadowingEnvCfg_PLAY(G1CircularHoleShadowingEnvCfg):
    require_compatible_binding: bool = False

    scene: G1CircularHoleSceneCfg = G1CircularHoleSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotReference"
        ),
        motion_reference=circular_hole_motion_reference_cfg.replace(debug_vis=True),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=[-2.5, -2.0, 1.6],
        lookat=[0.0, 0.0, 0.65],
        origin_type="env",
        env_index=0,
    )

    def __post_init__(self):
        super().__post_init__()

        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.reset_robot.params["randomize_pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot.params["randomize_velocity_range"] = {}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True
