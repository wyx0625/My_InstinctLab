"""将钻洞动作规范到圆洞坐标系，并生成训练与部署共用的绑定元数据。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

G1_29DOF_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


def _resolve_existing_or_local(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.exists() else candidate.resolve()


def _read_first(
    data: np.lib.npyio.NpzFile, keys: tuple[str, ...]
) -> tuple[np.ndarray, str]:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key]), key
    raise KeyError(f"缺少字段，候选字段为：{keys}；实际字段为：{data.files}")


def _read_framerate(data: np.lib.npyio.NpzFile) -> float:
    value, _ = _read_first(
        data, ("framerate", "fps", "mocap_framerate", "mocap_frame_rate")
    )
    return float(np.asarray(value).reshape(-1)[0])


def _read_joint_names(data: np.lib.npyio.NpzFile, joint_count: int) -> list[str]:
    for key in ("joint_names", "dof_names", "robot_joint_names"):
        if key in data.files:
            return [str(name) for name in np.asarray(data[key], dtype=object).tolist()]
    if joint_count == len(G1_29DOF_JOINT_NAMES):
        return G1_29DOF_JOINT_NAMES.copy()
    raise KeyError(
        f"动作包含 {joint_count} 个关节，但文件没有 joint_names/dof_names，无法安全推断关节顺序。"
    )


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_w, first_x, first_y, first_z = np.moveaxis(first, -1, 0)
    second_w, second_x, second_y, second_z = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            first_w * second_w
            - first_x * second_x
            - first_y * second_y
            - first_z * second_z,
            first_w * second_x
            + first_x * second_w
            + first_y * second_z
            - first_z * second_y,
            first_w * second_y
            - first_x * second_z
            + first_y * second_w
            + first_z * second_x,
            first_w * second_z
            + first_x * second_y
            - first_y * second_x
            + first_z * second_w,
        ),
        axis=-1,
    )


def _to_wxyz(
    quaternions: np.ndarray, source_key: str, quat_order: str
) -> tuple[np.ndarray, str]:
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"根旋转必须是 (帧数, 4) 四元数，实际为 {quaternions.shape}。")
    resolved_order = quat_order
    if quat_order == "auto":
        resolved_order = "xyzw" if source_key == "root_rot" else "wxyz"
    if resolved_order == "xyzw":
        quaternions = quaternions[:, [3, 0, 1, 2]]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("根旋转中存在零长度四元数。")
    return quaternions / norms, resolved_order


def _extract_mujoco_body_poses(
    qpos: np.ndarray,
    robot_xml: Path,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos 宽度 {qpos.shape[1]} 与 MuJoCo model.nq={model.nq} 不一致。"
        )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo 模型中找不到 body：{body_name}")
    data = mujoco.MjData(model)
    positions = np.zeros((len(qpos), 3), dtype=np.float64)
    quaternions = np.zeros((len(qpos), 4), dtype=np.float64)
    for frame_index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        positions[frame_index] = data.xpos[body_id]
        quaternions[frame_index] = data.xquat[body_id]
    return positions, quaternions


def _estimate_approach_axis(root_positions: np.ndarray) -> np.ndarray:
    sample_count = max(2, min(20, len(root_positions) // 10))
    direction = root_positions[-sample_count:, :2].mean(axis=0) - root_positions[
        :sample_count, :2
    ].mean(axis=0)
    if np.linalg.norm(direction) < 0.05:
        centered = root_positions[:, :2] - root_positions[:, :2].mean(axis=0)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        direction = right_vectors[0]
        temporal_direction = root_positions[-1, :2] - root_positions[0, :2]
        if np.dot(direction, temporal_direction) < 0.0:
            direction = -direction
    norm = np.linalg.norm(direction)
    if norm < 1.0e-8:
        raise ValueError("根节点平面位移过小，无法自动估计穿洞方向；请检查动作数据。")
    return direction / norm


def _estimate_crossing_frame(
    root_positions: np.ndarray, approach_axis: np.ndarray
) -> int:
    frame_count = len(root_positions)
    search_start = max(1, int(frame_count * 0.10))
    search_end = min(frame_count - 1, int(frame_count * 0.90))
    window = max(3, min(21, frame_count // 20))
    if window % 2 == 0:
        window += 1
    smoothed_height = np.convolve(
        root_positions[:, 2], np.ones(window) / window, mode="same"
    )
    progress = root_positions[:, :2] @ approach_axis
    progress_range = progress[search_end - 1] - progress[search_start]
    if abs(progress_range) < 1.0e-6:
        normalized_progress = np.linspace(0.0, 1.0, frame_count)
    else:
        normalized_progress = (progress - progress[search_start]) / progress_range
    height_slice = smoothed_height[search_start:search_end]
    height_scale = max(float(np.ptp(height_slice)), 1.0e-6)
    height_score = (smoothed_height - float(height_slice.min())) / height_scale
    score = height_score + 0.35 * np.abs(normalized_progress - 0.5)
    return int(search_start + np.argmin(score[search_start:search_end]))


def _save_binding_preview(
    output_path: Path,
    positions: np.ndarray,
    crossing_frame: int,
    panel_height: float,
    hole_radius: float,
    hole_center_z: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle, Rectangle

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=font_path
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    frame_colors = np.arange(len(positions))

    axes[0].add_patch(Rectangle((-0.075, -0.60), 0.15, 1.20, color="0.35", alpha=0.75))
    axes[0].scatter(
        positions[:, 0], positions[:, 1], c=frame_colors, cmap="viridis", s=8
    )
    axes[0].scatter(
        positions[crossing_frame, 0],
        positions[crossing_frame, 1],
        c="red",
        marker="x",
        s=80,
    )
    axes[0].set(xlabel="X / 穿越方向 (m)", ylabel="Y / 板宽方向 (m)", title="俯视图 XY")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)

    axes[1].add_patch(
        Rectangle(
            (-0.075, 0.0), 0.15, hole_center_z - hole_radius, color="0.35", alpha=0.75
        )
    )
    axes[1].add_patch(
        Rectangle(
            (-0.075, hole_center_z + hole_radius),
            0.15,
            panel_height - hole_center_z - hole_radius,
            color="0.35",
            alpha=0.75,
        )
    )
    axes[1].scatter(
        positions[:, 0], positions[:, 2], c=frame_colors, cmap="viridis", s=8
    )
    axes[1].scatter(
        positions[crossing_frame, 0],
        positions[crossing_frame, 2],
        c="red",
        marker="x",
        s=80,
    )
    axes[1].set(xlabel="X / 穿越方向 (m)", ylabel="Z / 高度 (m)", title="侧视图 XZ")
    axes[1].grid(True, alpha=0.25)

    axes[2].add_patch(
        Rectangle((-0.60, 0.0), 1.20, panel_height, color="0.35", alpha=0.75)
    )
    axes[2].add_patch(Circle((0.0, hole_center_z), hole_radius, color="white"))
    axes[2].add_patch(
        Circle(
            (0.0, hole_center_z), hole_radius, fill=False, color="black", linewidth=1.5
        )
    )
    axes[2].scatter(
        positions[crossing_frame, 1],
        positions[crossing_frame, 2],
        c="red",
        marker="x",
        s=80,
    )
    axes[2].set(
        xlabel="Y / 板宽方向 (m)",
        ylabel="Z / 高度 (m)",
        title=f"正视图 YZ（穿洞帧 {crossing_frame}）",
    )
    axes[2].set_xlim(-0.65, 0.65)
    axes[2].set_ylim(0.0, panel_height + 0.05)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.25)

    figure.suptitle("Circular-hole motion binding preview")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def bind_motion(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = _resolve_existing_or_local(args.input)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.parent.exists():
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=True) as data:
        if "qpos" in data.files:
            qpos = np.asarray(data["qpos"])
            if qpos.ndim != 2 or qpos.shape[1] < 8:
                raise ValueError(
                    f"qpos 必须是 (帧数, 7 + 关节数)，实际为 {qpos.shape}。"
                )
            source_robot_xml = None
            if "robot_xml" in data.files:
                source_robot_xml = _resolve_existing_or_local(
                    str(np.asarray(data["robot_xml"]).item())
                )
            if args.qpos_root_conversion == "torso":
                if source_robot_xml is None or not source_robot_xml.exists():
                    raise FileNotFoundError(
                        "qpos root 转换需要 NPZ 中存在可访问的 robot_xml。"
                    )
                root_positions, root_quaternions = _extract_mujoco_body_poses(
                    qpos, source_robot_xml, "torso_link"
                )
                root_position_key = "MuJoCo FK torso_link position from qpos"
                root_quaternion_key = "MuJoCo FK torso_link wxyz quaternion from qpos"
            else:
                root_positions = qpos[:, :3]
                root_quaternions = qpos[:, 3:7]
                root_position_key = "qpos[:, :3]"
                root_quaternion_key = "qpos[:, 3:7]"
            joint_positions = qpos[:, 7:]
            joint_position_key = "qpos[:, 7:]"
        else:
            root_positions, root_position_key = _read_first(
                data, ("base_pos_w", "root_pos", "trans")
            )
            root_quaternions, root_quaternion_key = _read_first(
                data, ("base_quat_w", "root_rot")
            )
            joint_positions, joint_position_key = _read_first(
                data, ("joint_pos", "dof_pos")
            )
        framerate = _read_framerate(data)
        joint_names = _read_joint_names(data, joint_positions.shape[1])

    root_positions = np.asarray(root_positions, dtype=np.float64)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    root_quaternions, resolved_quat_order = _to_wxyz(
        np.asarray(root_quaternions, dtype=np.float64),
        root_quaternion_key,
        args.quat_order,
    )
    if not (len(root_positions) == len(root_quaternions) == len(joint_positions)):
        raise ValueError(
            "base/root position、rotation 和 joint/dof position 的帧数不一致。"
        )
    if root_positions.ndim != 2 or root_positions.shape[1] != 3:
        raise ValueError(f"根位置必须是 (帧数, 3)，实际为 {root_positions.shape}。")

    approach_axis = _estimate_approach_axis(root_positions)
    crossing_frame = (
        _estimate_crossing_frame(root_positions, approach_axis)
        if args.crossing_frame is None
        else int(args.crossing_frame)
    )
    if crossing_frame < 0 or crossing_frame >= len(root_positions):
        raise ValueError(
            f"crossing_frame={crossing_frame} 超出 [0, {len(root_positions) - 1}]。"
        )

    local_window = max(
        2,
        min(
            args.approach_window,
            crossing_frame,
            len(root_positions) - crossing_frame - 1,
        ),
    )
    local_direction = (
        root_positions[crossing_frame + local_window, :2]
        - root_positions[crossing_frame - local_window, :2]
    )
    if np.linalg.norm(local_direction) >= 0.05:
        approach_axis = local_direction / np.linalg.norm(local_direction)

    approach_yaw = math.atan2(float(approach_axis[1]), float(approach_axis[0]))
    panel_origin_xy = (
        root_positions[crossing_frame, :2] + args.panel_x_offset * approach_axis
    )
    delta_xy = root_positions[:, :2] - panel_origin_xy
    cosine = math.cos(approach_yaw)
    sine = math.sin(approach_yaw)
    bound_positions = root_positions.copy()
    bound_positions[:, 0] = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
    bound_positions[:, 1] = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]

    alignment_quaternion = np.zeros_like(root_quaternions)
    alignment_quaternion[:, 0] = math.cos(-approach_yaw / 2.0)
    alignment_quaternion[:, 3] = math.sin(-approach_yaw / 2.0)
    bound_quaternions = _quat_multiply_wxyz(alignment_quaternion, root_quaternions)
    bound_quaternions /= np.linalg.norm(bound_quaternions, axis=1, keepdims=True)

    post_yaw = math.radians(args.alignment_yaw_deg)
    post_cosine = math.cos(post_yaw)
    post_sine = math.sin(post_yaw)
    post_x = post_cosine * bound_positions[:, 0] + post_sine * bound_positions[:, 1]
    post_y = -post_sine * bound_positions[:, 0] + post_cosine * bound_positions[:, 1]
    bound_positions[:, 0] = post_x - args.alignment_x
    bound_positions[:, 1] = post_y - args.alignment_y
    z_profile_center = (
        crossing_frame
        if args.alignment_z_center_frame is None
        else int(args.alignment_z_center_frame)
    )
    z_profile = np.zeros(len(bound_positions), dtype=np.float64)
    if args.alignment_z_peak > 0.0 and args.alignment_z_half_width_frames > 0:
        frame_distance = np.abs(np.arange(len(bound_positions)) - z_profile_center)
        active = frame_distance < args.alignment_z_half_width_frames
        z_profile[active] = (
            args.alignment_z_peak
            * 0.5
            * (
                1.0
                + np.cos(
                    math.pi
                    * frame_distance[active]
                    / args.alignment_z_half_width_frames
                )
            )
        )
    bound_positions[:, 2] += args.alignment_z + z_profile
    post_quaternion = np.zeros_like(bound_quaternions)
    post_quaternion[:, 0] = math.cos(-post_yaw / 2.0)
    post_quaternion[:, 3] = math.sin(-post_yaw / 2.0)
    bound_quaternions = _quat_multiply_wxyz(post_quaternion, bound_quaternions)
    bound_quaternions /= np.linalg.norm(bound_quaternions, axis=1, keepdims=True)

    output_motion = output_dir / f"{input_path.stem}_bound_retargetted.npz"
    np.savez_compressed(
        output_motion,
        framerate=np.asarray(framerate, dtype=np.float32),
        joint_names=np.asarray(joint_names, dtype=object),
        joint_pos=joint_positions,
        base_pos_w=bound_positions.astype(np.float32),
        base_quat_w=bound_quaternions.astype(np.float32),
    )

    binding = {
        "version": 1,
        "source_motion": str(input_path),
        "bound_motion": output_motion.name,
        "source_fields": {
            "base_position": root_position_key,
            "base_quaternion": root_quaternion_key,
            "joint_position": joint_position_key,
            "quaternion_order": resolved_quat_order,
            "qpos_root_conversion": args.qpos_root_conversion,
        },
        "motion": {
            "framerate": framerate,
            "num_frames": len(root_positions),
            "crossing_frame": crossing_frame,
            "crossing_time_s": crossing_frame / framerate,
        },
        "motion_to_hole": {
            "raw_panel_origin_xy": panel_origin_xy.tolist(),
            "raw_approach_yaw_rad": approach_yaw,
            "panel_x_offset": args.panel_x_offset,
            "post_alignment": {
                "yaw_deg": args.alignment_yaw_deg,
                "x": args.alignment_x,
                "y": args.alignment_y,
                "z": args.alignment_z,
                "z_profile": {
                    "type": "raised_cosine",
                    "peak": args.alignment_z_peak,
                    "center_frame": z_profile_center,
                    "half_width_frames": args.alignment_z_half_width_frames,
                },
            },
            "task_frame": "X through-hole, Y panel-width, Z up",
        },
        "obstacle": {
            "position": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "panel_width": 1.20,
            "panel_height": args.panel_height,
            "panel_thickness": 0.15,
            "hole_radius": args.hole_radius,
            "hole_center_z": args.hole_center_z,
        },
        "camera": {
            "parent_frame": "torso_link",
            "convention": "world",
            "translation_parent_camera": [
                0.0487988662332928,
                0.015,
                0.4378029937970051,
            ],
            "rotation_parent_camera_wxyz": [
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ],
            "raw_resolution_hw": [27, 48],
            "raw_fov_deg": [58.0, 87.0],
            "raw_intrinsic_matrix": [
                [25.29072380065918, 0.0, 24.0],
                [0.0, 24.354644775390625, 13.5],
                [0.0, 0.0, 1.0],
            ],
            "crop_top_bottom_left_right": [2, 2, 2, 2],
            "policy_resolution_hw": [18, 32],
            "policy_intrinsic_matrix_after_crop_resize": [
                [18.393253673206676, 0.0, 15.863636363636363],
                [0.0, 19.06015678074049, 8.891304347826088],
                [0.0, 0.0, 1.0],
            ],
            "depth_range_m": [0.0, 2.0],
            "update_period_s": 0.02,
        },
        "training_randomization": {
            "initial": {
                "x": [-0.10, 0.10],
                "y": [-0.05, 0.05],
                "yaw_rad": [-0.087, 0.087],
            },
            "expanded": {
                "x": [-0.25, 0.25],
                "y": [-0.10, 0.10],
                "yaw_rad": [-0.175, 0.175],
            },
        },
        "deployment": {
            "anchor": "detected_hole_pose_world",
            "root_position_world": "hole_position_world + hole_rotation_world * root_position_task",
            "root_rotation_world": "hole_rotation_world * root_rotation_task",
        },
    }
    binding_path = output_dir / "circular_hole_binding.yaml"
    binding_path.write_text(
        yaml.safe_dump(binding, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    selection_path = output_dir / "selected_motion.yaml"
    selection_path.write_text(
        yaml.safe_dump(
            {"selected_files": [output_motion.name], "motion_weights": [1.0]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preview_path = output_dir / "circular_hole_binding_preview.png"
    _save_binding_preview(
        preview_path,
        bound_positions,
        crossing_frame,
        panel_height=args.panel_height,
        hole_radius=args.hole_radius,
        hole_center_z=args.hole_center_z,
    )

    print(f"[INFO] 绑定动作：{output_motion}")
    print(f"[INFO] 绑定元数据：{binding_path}")
    print(f"[INFO] 绑定预览：{preview_path}")
    print(f"[INFO] 自动穿洞帧：{crossing_frame} ({crossing_frame / framerate:.3f}s)")
    print(f"[INFO] 原始穿越方向 yaw：{math.degrees(approach_yaw):.2f}°")
    return output_motion, binding_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将钻洞动作和 CircularHoleObstacle 绑定到统一任务坐标系。"
    )
    parser.add_argument(
        "--input", required=True, help="原始或重定向后的 NPZ 动作文件。"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出绑定动作、选择文件和 binding YAML 的目录。",
    )
    parser.add_argument(
        "--crossing_frame",
        type=int,
        default=None,
        help="穿过板中面的帧；默认根据低姿态自动估计。",
    )
    parser.add_argument(
        "--panel_x_offset",
        type=float,
        default=0.0,
        help="板平面相对所选根节点位置沿穿越方向的偏移。",
    )
    parser.add_argument(
        "--approach_window", type=int, default=10, help="估计穿越方向时使用的前后帧数。"
    )
    parser.add_argument(
        "--quat_order",
        choices=("auto", "wxyz", "xyzw"),
        default="auto",
        help="输入根四元数顺序；auto 对 root_rot 使用 xyzw，其余字段使用 wxyz。",
    )
    parser.add_argument(
        "--qpos_root_conversion",
        choices=("torso", "none"),
        default="none",
        help="圆洞任务使用 pelvis-root G1，因此默认保留 qpos 的 pelvis freejoint 位姿。",
    )
    parser.add_argument(
        "--alignment_yaw_deg",
        type=float,
        default=0.0,
        help="规范化后额外施加的水平旋转。",
    )
    parser.add_argument(
        "--alignment_x", type=float, default=0.0, help="规范化后沿 X 调整动作。"
    )
    parser.add_argument(
        "--alignment_y", type=float, default=0.0, help="规范化后沿 Y 调整动作。"
    )
    parser.add_argument(
        "--alignment_z", type=float, default=0.0, help="规范化后沿 Z 调整动作。"
    )
    parser.add_argument(
        "--alignment_z_peak", type=float, default=0.0, help="穿洞阶段平滑抬升的峰值。"
    )
    parser.add_argument(
        "--alignment_z_center_frame",
        type=int,
        default=None,
        help="平滑抬升中心帧；默认使用 crossing_frame。",
    )
    parser.add_argument(
        "--alignment_z_half_width_frames",
        type=int,
        default=0,
        help="平滑抬升的半窗口帧数。",
    )
    parser.add_argument(
        "--hole_radius", type=float, default=0.40, help="与动作绑定的圆洞半径。"
    )
    parser.add_argument(
        "--hole_center_z", type=float, default=0.70, help="与动作绑定的圆洞中心高度。"
    )
    parser.add_argument(
        "--panel_height", type=float, default=1.20, help="与动作绑定的障碍板高度。"
    )
    bind_motion(parser.parse_args())


if __name__ == "__main__":
    main()
