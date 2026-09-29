"""Shadowing task reward functions."""

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .terminations import obstacle_contact_force


def obstacle_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return a binary penalty whenever any robot link contacts the obstacle."""
    return (obstacle_contact_force(env, sensor_names) > threshold).float()
