"""使用 G1 完整 link 轨迹验证动作是否能够通过指定圆洞。"""

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="验证圆洞动作绑定的几何可行性。")
parser.add_argument(
    "--binding_dir",
    default=os.path.expanduser("~/Wyx/Project_Instinct/motions/circular_hole_bound"),
    help="包含 circular_hole_binding.yaml 的目录。",
)
parser.add_argument(
    "--geometry_margin",
    type=float,
    default=0.06,
    help="在 link 中心半径之外增加的实体余量。",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import instinctlab.tasks  # noqa: F401
import numpy as np
import yaml
from instinctlab.assets.unitree_g1 import G1_29DOF_LINKS
from instinctlab.tasks.shadowing.perceptive.config.g1.circular_hole_shadowing_cfg import (
    G1CircularHoleShadowingEnvCfg_PLAY,
)


def _crossings(
    positions: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    plane_coordinate: float,
) -> list[tuple[float, float, int, int]] | None:
    along = positions[:, :, :2] @ normal
    lateral = positions[:, :, :2] @ tangent
    crossing_points: list[tuple[float, float, int, int]] = []
    for link_index in range(positions.shape[1]):
        signed_distance = along[:, link_index] - plane_coordinate
        crossing_indices = np.where(signed_distance[:-1] * signed_distance[1:] <= 0.0)[
            0
        ]
        if len(crossing_indices) == 0:
            return None
        for frame_index in crossing_indices:
            denominator = (
                signed_distance[frame_index] - signed_distance[frame_index + 1]
            )
            alpha = (
                0.0
                if abs(denominator) < 1.0e-9
                else signed_distance[frame_index] / denominator
            )
            lateral_position = lateral[frame_index, link_index] + alpha * (
                lateral[frame_index + 1, link_index] - lateral[frame_index, link_index]
            )
            height = positions[frame_index, link_index, 2] + alpha * (
                positions[frame_index + 1, link_index, 2]
                - positions[frame_index, link_index, 2]
            )
            crossing_points.append(
                (float(lateral_position), float(height), link_index, int(frame_index))
            )
    return crossing_points


def _required_radius(
    crossing_points: list[tuple[float, float, int, int]],
    hole_lateral: float,
    hole_center_z: float,
) -> tuple[float, tuple[float, float, int, int]]:
    radii = [
        math.hypot(lateral - hole_lateral, height - hole_center_z)
        for lateral, height, _, _ in crossing_points
    ]
    worst_index = int(np.argmax(radii))
    return float(radii[worst_index]), crossing_points[worst_index]


def _search_best_se2(
    positions: np.ndarray,
    hole_center_z: float,
) -> tuple[float, float, float, float, tuple[float, float, int, int]]:
    best: tuple[float, float, float, float, tuple[float, float, int, int]] | None = None
    for angle_deg in np.linspace(-45.0, 45.0, 37):
        angle = math.radians(float(angle_deg))
        normal = np.asarray([math.cos(angle), math.sin(angle)])
        tangent = np.asarray([-math.sin(angle), math.cos(angle)])
        along = positions[:, :, :2] @ normal
        plane_min = float(along[0].max() + 0.075)
        plane_max = float(along[-1].min() - 0.075)
        if plane_min >= plane_max:
            continue
        for plane_coordinate in np.linspace(plane_min, plane_max, 81):
            crossing_points = _crossings(
                positions, normal, tangent, float(plane_coordinate)
            )
            if crossing_points is None:
                continue
            for hole_lateral in np.linspace(-0.20, 0.20, 81):
                required_radius, worst = _required_radius(
                    crossing_points, float(hole_lateral), hole_center_z
                )
                candidate = (
                    required_radius,
                    float(angle_deg),
                    float(plane_coordinate),
                    float(hole_lateral),
                    worst,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("动作中存在未穿过任何候选板平面的 link。")
    return best


def _validate_exact_collision_mesh(
    binding: dict,
    binding_path: Path,
    clearance_required: float,
) -> dict:
    import mujoco

    source_motion = Path(binding["source_motion"]).expanduser()
    if not source_motion.exists():
        source_motion = (binding_path.parent / source_motion).resolve()
    with np.load(source_motion, allow_pickle=True) as source_data:
        qpos = np.asarray(source_data["qpos"], dtype=np.float64)
        robot_xml = Path(str(np.asarray(source_data["robot_xml"]).item())).expanduser()
    if not robot_xml.exists():
        raise FileNotFoundError(f"找不到 UMR robot_xml：{robot_xml}")

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    collision_geometries = []
    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
            or int(model.geom_contype[geom_id]) == 0
        ):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=np.float64,
        )
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
        )
        geom_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or f"geom_{geom_id}"
        )
        collision_geometries.append((geom_id, body_name, geom_name, vertices))

    motion_binding = binding["motion_to_hole"]
    raw_origin = np.asarray(motion_binding["raw_panel_origin_xy"], dtype=np.float64)
    raw_yaw = float(motion_binding["raw_approach_yaw_rad"])
    raw_cosine = math.cos(raw_yaw)
    raw_sine = math.sin(raw_yaw)
    post_alignment = motion_binding.get("post_alignment", {})
    post_yaw = math.radians(float(post_alignment.get("yaw_deg", 0.0)))
    post_cosine = math.cos(post_yaw)
    post_sine = math.sin(post_yaw)
    post_x = float(post_alignment.get("x", 0.0))
    post_y = float(post_alignment.get("y", 0.0))
    post_z = float(post_alignment.get("z", 0.0))
    z_profile_cfg = post_alignment.get("z_profile", {})
    z_profile_peak = float(z_profile_cfg.get("peak", 0.0))
    z_profile_center = int(z_profile_cfg.get("center_frame", 0))
    z_profile_half_width = int(z_profile_cfg.get("half_width_frames", 0))

    obstacle = binding["obstacle"]
    half_thickness = float(obstacle["panel_thickness"]) / 2.0
    half_width = float(obstacle["panel_width"]) / 2.0
    panel_height = float(obstacle["panel_height"])
    hole_radius = float(obstacle["hole_radius"])
    hole_center_z = float(obstacle["hole_center_z"])
    penetration_events = []
    max_crossing_radius = 0.0
    worst_crossing = None

    for frame_index, frame_qpos in enumerate(qpos):
        frame_distance = abs(frame_index - z_profile_center)
        frame_z_offset = post_z
        if (
            z_profile_peak > 0.0
            and z_profile_half_width > 0
            and frame_distance < z_profile_half_width
        ):
            frame_z_offset += (
                z_profile_peak
                * 0.5
                * (1.0 + math.cos(math.pi * frame_distance / z_profile_half_width))
            )
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        for geom_id, body_name, geom_name, vertices in collision_geometries:
            rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            points = vertices @ rotation.T + np.asarray(data.geom_xpos[geom_id])
            raw_delta = points[:, :2] - raw_origin
            canonical_x = raw_cosine * raw_delta[:, 0] + raw_sine * raw_delta[:, 1]
            canonical_y = -raw_sine * raw_delta[:, 0] + raw_cosine * raw_delta[:, 1]
            task_x = post_cosine * canonical_x + post_sine * canonical_y - post_x
            task_y = -post_sine * canonical_x + post_cosine * canonical_y - post_y
            task_z = points[:, 2] + frame_z_offset
            inside_slab = (
                (np.abs(task_x) <= half_thickness)
                & (np.abs(task_y) <= half_width)
                & (task_z >= 0.0)
                & (task_z <= panel_height)
            )
            if not inside_slab.any():
                continue
            slab_indices = np.where(inside_slab)[0]
            radii = np.sqrt(
                task_y[slab_indices] ** 2 + (task_z[slab_indices] - hole_center_z) ** 2
            )
            local_max_index = int(np.argmax(radii))
            if float(radii[local_max_index]) > max_crossing_radius:
                vertex_index = int(slab_indices[local_max_index])
                max_crossing_radius = float(radii[local_max_index])
                worst_crossing = {
                    "frame": frame_index,
                    "body": body_name,
                    "geometry": geom_name,
                    "xyz": [
                        float(task_x[vertex_index]),
                        float(task_y[vertex_index]),
                        float(task_z[vertex_index]),
                    ],
                }
            penetration_count = int(np.count_nonzero(radii >= hole_radius))
            if penetration_count > 0:
                penetration_events.append(
                    {
                        "frame": frame_index,
                        "body": body_name,
                        "geometry": geom_name,
                        "vertex_count": penetration_count,
                    }
                )

    clearance = hole_radius - max_crossing_radius
    return {
        "status": "compatible"
        if not penetration_events and clearance >= clearance_required
        else "incompatible",
        "collision_mesh_geometries": len(collision_geometries),
        "penetration_event_count": len(penetration_events),
        "penetration_frame_count": len(
            {event["frame"] for event in penetration_events}
        ),
        "penetrating_bodies": sorted({event["body"] for event in penetration_events}),
        "max_crossing_radius": max_crossing_radius,
        "radial_clearance": clearance,
        "required_clearance": clearance_required,
        "worst_crossing": worst_crossing,
        "first_penetration_events": penetration_events[:20],
    }


def main() -> None:
    binding_path = (
        Path(args_cli.binding_dir).expanduser() / "circular_hole_binding.yaml"
    )
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    env_cfg = G1CircularHoleShadowingEnvCfg_PLAY()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.motion_reference.link_of_interests = G1_29DOF_LINKS
    env = gym.make("Instinct-Perceptive-Shadowing-CircularHole-G1-Play-v0", cfg=env_cfg)

    motion_reference = env.unwrapped.scene["motion_reference"]
    sequence = motion_reference.motion_buffers[
        "CircularHoleMotion"
    ]._all_motion_sequences
    positions = sequence.link_pos_w[0].cpu().numpy()
    hole_radius = float(binding["obstacle"]["hole_radius"])
    hole_center_z = float(binding["obstacle"]["hole_center_z"])

    fixed_crossings = _crossings(
        positions,
        normal=np.asarray([1.0, 0.0]),
        tangent=np.asarray([0.0, 1.0]),
        plane_coordinate=0.0,
    )
    if fixed_crossings is None:
        raise RuntimeError("至少一个 link 没有穿过当前板平面。")
    fixed_required, fixed_worst = _required_radius(fixed_crossings, 0.0, hole_center_z)
    best_required, best_angle, best_plane, best_lateral, best_worst = _search_best_se2(
        positions, hole_center_z
    )
    recommended_radius = best_required + args_cli.geometry_margin
    exact_validation = _validate_exact_collision_mesh(
        binding,
        binding_path,
        clearance_required=0.005,
    )
    status = exact_validation["status"]

    fixed_link = G1_29DOF_LINKS[fixed_worst[2]]
    best_link = G1_29DOF_LINKS[best_worst[2]]
    binding["compatibility_validation"] = {
        "status": status,
        "configured_hole_radius": hole_radius,
        "geometry_margin": args_cli.geometry_margin,
        "current_binding": {
            "center_only_required_radius": fixed_required,
            "worst_link": fixed_link,
            "worst_crossing_yz": [fixed_worst[0], fixed_worst[1]],
            "worst_crossing_frame": fixed_worst[3],
        },
        "best_se2_search": {
            "center_only_required_radius": best_required,
            "recommended_radius_with_margin": recommended_radius,
            "panel_yaw_offset_deg": best_angle,
            "panel_plane_coordinate": best_plane,
            "hole_lateral_coordinate": best_lateral,
            "worst_link": best_link,
            "worst_crossing_yz": [best_worst[0], best_worst[1]],
            "worst_crossing_frame": best_worst[3],
        },
        "exact_collision_mesh": exact_validation,
    }
    binding_path.write_text(
        yaml.safe_dump(binding, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"[RESULT] compatibility={status}")
    print(
        f"[RESULT] current center-only required radius={fixed_required:.3f}m, worst={fixed_link}"
    )
    print(
        f"[RESULT] best SE2 center-only required radius={best_required:.3f}m, worst={best_link}"
    )
    print(f"[RESULT] recommended radius with margin={recommended_radius:.3f}m")
    print(
        "[RESULT] exact collision mesh: "
        f"penetration_frames={exact_validation['penetration_frame_count']}, "
        f"clearance={exact_validation['radial_clearance']:.3f}m"
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
