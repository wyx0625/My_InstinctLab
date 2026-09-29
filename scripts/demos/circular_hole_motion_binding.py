"""在 Isaac Sim 中播放已绑定的 G1 reference 与圆洞障碍物。"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="播放圆洞坐标系中的 G1 reference 动作。")
parser.add_argument(
    "--playback_speed", type=float, default=1.0, help="Reference 播放速度倍率。"
)
parser.add_argument(
    "--cycles", type=int, default=0, help="播放循环次数；0 表示持续播放。"
)
parser.add_argument(
    "--no_depth_vis",
    action="store_true",
    help="关闭 Isaac Sim 射线命中点和实时深度图窗口。",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import instinctlab.tasks  # noqa: F401
import isaaclab.sim as sim_utils
import torch
from instinctlab.tasks.shadowing.perceptive.config.g1.circular_hole_shadowing_cfg import (
    G1CircularHoleShadowingEnvCfg_PLAY,
)


def main() -> None:
    if args_cli.playback_speed <= 0.0:
        raise ValueError("--playback_speed 必须大于 0。")
    env_cfg = G1CircularHoleShadowingEnvCfg_PLAY()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.motion_reference.update_period = env_cfg.sim.dt
    env_cfg.scene.motion_reference.motion_buffers[
        "CircularHoleMotion"
    ].motion_target_framerate = 1.0 / env_cfg.sim.dt
    show_depth = not args_cli.no_depth_vis and not args_cli.headless
    env_cfg.scene.camera.prim_path = "{ENV_REGEX_NS}/RobotReference/torso_link"
    env_cfg.scene.camera.update_period = env_cfg.sim.dt
    env_cfg.scene.camera.mesh_prim_paths = [
        target
        for target in env_cfg.scene.camera.mesh_prim_paths
        if "/Robot/" not in getattr(target, "prim_expr", str(target))
    ]
    env_cfg.scene.camera.debug_vis = show_depth
    env_cfg.observations.policy.depth_image.params["debug_vis"] = show_depth
    env = gym.make(
        "Instinct-Perceptive-Shadowing-CircularHole-G1-Play-v0",
        cfg=env_cfg,
    )
    env.reset()

    unwrapped = env.unwrapped
    scene = unwrapped.scene
    sim = unwrapped.sim
    motion_reference = scene["motion_reference"]
    camera = scene["camera"]
    stage = sim.stage
    sim_utils.set_prim_visibility(stage.GetPrimAtPath("/World/envs/env_0/Robot"), False)
    sim.set_camera_view(eye=(-2.4, -2.2, 1.55), target=(0.0, 0.0, 0.65))

    env_ids = torch.tensor([0], device=unwrapped.device, dtype=torch.long)
    duration = float(motion_reference.assigned_motion_lengths[0].item())
    elapsed = 0.0
    completed_cycles = 0
    sim_dt = sim.get_physics_dt()
    dt = sim_dt * args_cli.playback_speed
    next_frame_deadline = time.perf_counter()
    print(f"[INFO] 播放绑定 reference，时长 {duration:.3f}s；关闭窗口即可退出。")

    while simulation_app.is_running():
        scene.update(dt)
        _ = motion_reference.reference_frame
        motion_reference._set_reference_view_state()
        sim.render()
        camera.update(dt, force_recompute=True)
        if show_depth:
            unwrapped.observation_manager.compute_group("policy", update_history=False)
        next_frame_deadline += sim_dt / args_cli.playback_speed
        time.sleep(max(0.0, next_frame_deadline - time.perf_counter()))
        elapsed += dt
        if elapsed >= duration:
            completed_cycles += 1
            if args_cli.cycles > 0 and completed_cycles >= args_cli.cycles:
                break
            motion_reference.reset(env_ids)
            elapsed = 0.0

    print(f"[INFO] 已完成 {completed_cycles} 个 reference 播放循环。")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
