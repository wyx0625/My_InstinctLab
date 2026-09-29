"""Static circular-hole obstacle for Isaac Lab scenes."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.utils import (
    bind_physics_material,
    clone,
    create_prim,
    get_current_stage,
)
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom

from .circular_hole_mesh import TriangleMeshData, build_circular_hole_mesh


@clone
def spawn_circular_hole_obstacle(
    prim_path: str,
    cfg: CircularHoleObstacleSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a static visual mesh and an exact triangle-mesh collider."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    mesh_data = build_circular_hole_mesh(
        panel_width=cfg.panel_width,
        panel_height=cfg.panel_height,
        panel_thickness=cfg.panel_thickness,
        hole_radius=cfg.hole_radius,
        hole_center_z=cfg.hole_center_z,
        circle_segments=cfg.circle_segments,
    )

    create_prim(
        prim_path,
        "Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    UsdGeom.Xform.Define(stage, f"{prim_path}/geometry")

    visual_mesh = _define_mesh(stage, f"{prim_path}/geometry/visual", mesh_data)
    visual_mesh.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])

    collision_path = f"{prim_path}/geometry/collision"
    collision_mesh = _define_mesh(stage, collision_path, mesh_data)
    collision_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    schemas.define_collision_properties(
        collision_path, cfg.collision_props, stage=stage
    )
    schemas.define_mesh_collision_properties(
        collision_path, cfg.mesh_collision_props, stage=stage
    )
    material_path = f"{prim_path}/physicsMaterial"
    cfg.physics_material.func(material_path, cfg.physics_material)
    bind_physics_material(collision_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


def _define_mesh(
    stage: Usd.Stage, prim_path: str, mesh_data: TriangleMeshData
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    sim_utils.standardize_xform_ops(mesh.GetPrim())
    points = [Gf.Vec3f(*vertex) for vertex in mesh_data.vertices]
    min_point = Gf.Vec3f(
        *(min(vertex[axis] for vertex in mesh_data.vertices) for axis in range(3))
    )
    max_point = Gf.Vec3f(
        *(max(vertex[axis] for vertex in mesh_data.vertices) for axis in range(3))
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(mesh_data.faces))
    mesh.CreateFaceVertexIndicesAttr(
        [index for face in mesh_data.faces for index in face]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateExtentAttr([min_point, max_point])
    return mesh


@configclass
class CircularHoleObstacleSpawnerCfg(sim_utils.SpawnerCfg):
    """Low-level spawner configuration for the circular-hole mesh."""

    func = spawn_circular_hole_obstacle

    panel_width: float = 1.20
    panel_height: float = 1.20
    panel_thickness: float = 0.15
    hole_radius: float = 0.40
    hole_center_z: float = 0.70
    circle_segments: int = 48
    color: tuple[float, float, float] = (0.18, 0.42, 0.72)
    collision_props: schemas_cfg.CollisionPropertiesCfg = (
        schemas_cfg.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        )
    )
    mesh_collision_props: schemas_cfg.TriangleMeshPropertiesCfg = (
        schemas_cfg.TriangleMeshPropertiesCfg(usd_func=None)
    )
    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.8,
        restitution=0.0,
    )


@configclass
class CircularHoleObstacleCfg(AssetBaseCfg):
    """Scene asset configuration for a static panel with a circular through-hole.

    The asset origin is centered in Y, placed on the panel bottom in Z, and centered
    through the panel thickness in X. ``position`` therefore places the panel bottom.
    Quaternion rotations use Isaac Lab's ``(w, x, y, z)`` convention.
    """

    panel_width: float = 1.20
    panel_height: float = 1.20
    panel_thickness: float = 0.15
    hole_radius: float = 0.40
    hole_center_z: float = 0.70
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    circle_segments: int = 48
    color: tuple[float, float, float] = (0.18, 0.42, 0.72)

    def __post_init__(self) -> None:
        build_circular_hole_mesh(
            panel_width=self.panel_width,
            panel_height=self.panel_height,
            panel_thickness=self.panel_thickness,
            hole_radius=self.hole_radius,
            hole_center_z=self.hole_center_z,
            circle_segments=self.circle_segments,
        )
        self.init_state = AssetBaseCfg.InitialStateCfg(
            pos=self.position, rot=self.rotation
        )
        self.spawn = CircularHoleObstacleSpawnerCfg(
            panel_width=self.panel_width,
            panel_height=self.panel_height,
            panel_thickness=self.panel_thickness,
            hole_radius=self.hole_radius,
            hole_center_z=self.hole_center_z,
            circle_segments=self.circle_segments,
            color=self.color,
        )
