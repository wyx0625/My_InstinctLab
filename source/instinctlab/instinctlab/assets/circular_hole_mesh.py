"""Geometry generation for a rectangular panel with a circular through-hole."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TriangleMeshData:
    """A triangle mesh represented by vertices and indexed faces."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


def build_circular_hole_mesh(
    panel_width: float,
    panel_height: float,
    panel_thickness: float,
    hole_radius: float,
    hole_center_z: float,
    circle_segments: int = 48,
) -> TriangleMeshData:
    """Build a watertight Z-up panel mesh whose thickness is aligned with X."""
    _validate_dimensions(
        panel_width,
        panel_height,
        panel_thickness,
        hole_radius,
        hole_center_z,
        circle_segments,
    )

    half_width = panel_width / 2.0
    half_thickness = panel_thickness / 2.0
    corner_angles = sorted(
        math.atan2(z - hole_center_z, y) % (2.0 * math.pi)
        for y, z in (
            (half_width, 0.0),
            (half_width, panel_height),
            (-half_width, panel_height),
            (-half_width, 0.0),
        )
    )
    segment_counts = [
        circle_segments // 4 + int(index < circle_segments % 4) for index in range(4)
    ]

    inner_yz: list[tuple[float, float]] = []
    outer_yz: list[tuple[float, float]] = []
    for corner_index, start_angle in enumerate(corner_angles):
        end_angle = corner_angles[(corner_index + 1) % 4]
        if end_angle <= start_angle:
            end_angle += 2.0 * math.pi
        count = segment_counts[corner_index]
        for segment_index in range(count):
            angle = start_angle + (end_angle - start_angle) * segment_index / count
            direction_y = math.cos(angle)
            direction_z = math.sin(angle)
            inner_yz.append(
                (
                    hole_radius * direction_y,
                    hole_center_z + hole_radius * direction_z,
                )
            )
            outer_yz.append(
                _ray_rectangle_intersection(
                    direction_y,
                    direction_z,
                    half_width,
                    panel_height,
                    hole_center_z,
                )
            )

    vertices: list[tuple[float, float, float]] = []
    for x, ring in (
        (half_thickness, outer_yz),
        (half_thickness, inner_yz),
        (-half_thickness, outer_yz),
        (-half_thickness, inner_yz),
    ):
        vertices.extend((x, y, z) for y, z in ring)

    front_outer = 0
    front_inner = circle_segments
    back_outer = 2 * circle_segments
    back_inner = 3 * circle_segments
    faces: list[tuple[int, int, int]] = []

    for index in range(circle_segments):
        next_index = (index + 1) % circle_segments
        front_outer_index = front_outer + index
        front_outer_next = front_outer + next_index
        front_inner_index = front_inner + index
        front_inner_next = front_inner + next_index
        back_outer_index = back_outer + index
        back_outer_next = back_outer + next_index
        back_inner_index = back_inner + index
        back_inner_next = back_inner + next_index

        faces.extend(
            (
                (front_outer_index, front_outer_next, front_inner_next),
                (front_outer_index, front_inner_next, front_inner_index),
                (back_outer_index, back_inner_next, back_outer_next),
                (back_outer_index, back_inner_index, back_inner_next),
                (front_outer_index, back_outer_index, back_outer_next),
                (front_outer_index, back_outer_next, front_outer_next),
                (front_inner_index, front_inner_next, back_inner_next),
                (front_inner_index, back_inner_next, back_inner_index),
            )
        )

    return TriangleMeshData(vertices=tuple(vertices), faces=tuple(faces))


def _ray_rectangle_intersection(
    direction_y: float,
    direction_z: float,
    half_width: float,
    panel_height: float,
    hole_center_z: float,
) -> tuple[float, float]:
    distances: list[float] = []
    if abs(direction_y) > 1.0e-12:
        distances.append(
            (half_width if direction_y > 0.0 else -half_width) / direction_y
        )
    if abs(direction_z) > 1.0e-12:
        boundary_z = panel_height if direction_z > 0.0 else 0.0
        distances.append((boundary_z - hole_center_z) / direction_z)
    distance = min(candidate for candidate in distances if candidate > 0.0)
    return direction_y * distance, hole_center_z + direction_z * distance


def _validate_dimensions(
    panel_width: float,
    panel_height: float,
    panel_thickness: float,
    hole_radius: float,
    hole_center_z: float,
    circle_segments: int,
) -> None:
    if panel_width <= 0.0 or panel_height <= 0.0 or panel_thickness <= 0.0:
        raise ValueError("Panel dimensions must all be positive.")
    if hole_radius <= 0.0:
        raise ValueError("Hole radius must be positive.")
    if 2.0 * hole_radius >= panel_width:
        raise ValueError("The hole diameter must be smaller than the panel width.")
    if (
        hole_center_z - hole_radius <= 0.0
        or hole_center_z + hole_radius >= panel_height
    ):
        raise ValueError("The hole must remain strictly inside the panel height.")
    if circle_segments < 12:
        raise ValueError("circle_segments must be at least 12.")
