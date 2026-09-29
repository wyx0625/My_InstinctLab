"""Shadowing task termination functions."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def obstacle_contact(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any robot link contacts the circular-hole obstacle."""
    contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in sensor_names:
        force_matrix = env.scene[sensor_name].data.force_matrix_w
        if force_matrix is not None:
            contact |= (
                torch.linalg.norm(force_matrix, dim=-1).amax(dim=(1, 2)) > threshold
            )
    return contact
