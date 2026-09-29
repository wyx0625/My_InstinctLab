"""显示带真实圆形贯穿孔碰撞的静态障碍物。"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="显示静态圆洞障碍物和位于障碍物前方的 G1 机器人。"
)
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量。")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from instinctlab.assets import CircularHoleObstacleCfg
from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_CFG
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

ROBOT_CFG = G1_29DOF_TORSOBASE_CFG.copy()
ROBOT_CFG.init_state.pos = (-1.5, 0.0, 0.8)


@configclass
class CircularHoleDemoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0), color=(0.18, 0.18, 0.18)),
    )
    obstacle = CircularHoleObstacleCfg(
        prim_path="{ENV_REGEX_NS}/CircularHoleObstacle",
        panel_width=1.20,
        panel_height=1.20,
        panel_thickness=0.15,
        hole_radius=0.40,
        hole_center_z=0.70,
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )
    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=(-3.0, -2.4, 1.7), target=(0.0, 0.0, 0.65))

    scene_cfg = CircularHoleDemoSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    print(
        "[INFO] 圆洞障碍物已加载：板 1.20 x 1.20 x 0.15 m，洞半径 0.40 m，洞底高度 0.30 m。"
    )
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    simulation_app.close()


if __name__ == "__main__":
    main()
