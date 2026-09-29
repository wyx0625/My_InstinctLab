"""Shadowing task termination functions."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def obstacle_contact_force(
    env: ManagerBasedRLEnv, sensor_names: list[str]
) -> torch.Tensor:
    """Return the maximum obstacle contact force across links and physics substeps."""
    max_force = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        sensor_data = env.scene[sensor_name].data
        force_matrix = sensor_data.force_matrix_w_history
        if force_matrix is None:
            force_matrix = sensor_data.force_matrix_w
        if force_matrix is not None:
            force_norm = torch.linalg.norm(force_matrix, dim=-1)
            reduce_dims = tuple(range(1, force_norm.ndim))
            max_force = torch.maximum(max_force, force_norm.amax(dim=reduce_dims))
    return max_force


def obstacle_contact(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any robot link contacts the circular-hole obstacle."""
    return obstacle_contact_force(env, sensor_names) > threshold
