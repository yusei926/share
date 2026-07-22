#!/usr/bin/env python3
"""Create a PhysX-valid assembled-table scene before the simulator starts.

Scene02.usd contains an obsolete fixed joint whose body relationships point at
pre-renamed prims. Runtime edits happen too late for PhysX to register a new
joint reliably, so this tool fixes a sibling USD before env_server initializes.
The source USD is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


TABLE_BODY = "/World/Table001/Table001_01"
WORKBENCH_BODY = "/World/Table278/Table278"
LEG_BODIES = (
    "/World/Leg001/Leg001",
    "/World/Leg001_01/Leg001",
    "/World/Leg001_03/Leg001",
    "/World/Leg001_06/Leg001",
)
JOINT_PREFIX = "FlipTableEvalFixedJoint_"
WHITE_CONTACT_MATERIAL = "/World/Looks/flip_table_contact_white"
WORKBENCH_CONTACT_MATERIAL = "/World/Looks/flip_table_contact_workbench"
WORKBENCH_TOP_COLLIDER = f"{WORKBENCH_BODY}/Collisions/Table278_Collider5"
ASSEMBLED_RIGID_BODIES = (TABLE_BODY, *LEG_BODIES, WORKBENCH_BODY)
LEG_SHAFT_COLLIDER_SUFFIX = "/Collisions/Leg001_Collider118"
LEG_EXTERNAL_COLLIDER_SUFFIXES = (
    "/Collisions/Leg001_Collider116",  # exposed end cap
    "/Collisions/Leg001_Collider117",  # registration-end block
    LEG_SHAFT_COLLIDER_SUFFIX,
)
LEG_VISUAL_MESH_SUFFIX = "/Visuals/Leg001"
LEG_VISUAL_MESH_NAME = "Leg001_visual"

# Scene02's leg mesh is authored with its long axis along local +Z. The
# registration end is attached to the tabletop, so the pre-flip pose uses a
# 180-degree X rotation to make the leg extend upward from the workbench.
LEG_ASSEMBLED_QUAT = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
DEX1_FORCE_CALIBRATION_CENTERS_M = {
    "left": (-1.105897, 2.140316, 0.880589),
    "right": (-1.098306, 2.422868, 0.867408),
}
DEX1_FORCE_CALIBRATION_ORIENTATIONS_WXYZ = {
    "left": (0.000769534, -0.022965045, 0.002735808, 0.999732229),
    "right": (-0.377091366, -0.088136697, 0.031728409, 0.921426792),
}
DEX1_FORCE_CALIBRATION_SIZE_M = (0.160, 0.040, 0.300)


def _add_dex1_force_calibration_blockers(stage: Usd.Stage) -> dict[str, object]:
    """Add static, leg-thickness fixtures for diagnostic force calibration."""

    root_path = "/World/FlipTableDex1ForceCalibration"
    UsdGeom.Xform.Define(stage, root_path)
    paths = {}
    for side, center in DEX1_FORCE_CALIBRATION_CENTERS_M.items():
        path = f"{root_path}/{side}_blocker"
        blocker = UsdGeom.Cube.Define(stage, path)
        blocker.CreateSizeAttr(1.0)
        blocker.AddTranslateOp().Set(Gf.Vec3d(*center))
        real, x, y, z = DEX1_FORCE_CALIBRATION_ORIENTATIONS_WXYZ[side]
        blocker.AddOrientOp().Set(Gf.Quatf(real, Gf.Vec3f(x, y, z)))
        blocker.AddScaleOp().Set(Gf.Vec3d(*DEX1_FORCE_CALIBRATION_SIZE_M))
        blocker.CreateDisplayColorAttr(
            [
                Gf.Vec3f(0.9, 0.1, 0.1)
                if side == "left"
                else Gf.Vec3f(0.1, 0.3, 0.9)
            ]
        )
        UsdPhysics.CollisionAPI.Apply(blocker.GetPrim()).CreateCollisionEnabledAttr(True)
        paths[side] = path
    return {
        "diagnostic_only": True,
        "leg_thickness_m": 0.040,
        "fixture_size_m": DEX1_FORCE_CALIBRATION_SIZE_M,
        "paths": paths,
        "centers_m": DEX1_FORCE_CALIBRATION_CENTERS_M,
        "orientations_wxyz": DEX1_FORCE_CALIBRATION_ORIENTATIONS_WXYZ,
    }


def _remove_nested_physics_scenes(stage: Usd.Stage) -> list[str]:
    """Remove asset-local scenes; Isaac Lab owns the single world PhysicsScene."""

    paths = [
        prim.GetPath()
        for prim in stage.TraverseAll()
        if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith("/World/")
    ]
    removed = []
    for path in paths:
        removed.append(str(path))
        stage.RemovePrim(path)
        # The prim may originate in a referenced layer. Keep an explicit
        # inactive override so it cannot reappear when this stage is exported.
        stage.OverridePrim(path).SetActive(False)
    return removed


def _define_contact_material(
    stage: Usd.Stage,
    path: str,
    *,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)
    # The standalone USD Python does not load the PhysX schema plugin. Author
    # its applied schema and attributes directly so they are active before
    # Isaac Sim creates any tensor views.
    material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    material.GetPrim().CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("average")
    material.GetPrim().CreateAttribute(
        "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("average")
    return material


def _prepare_contact_material_bindings(stage: Usd.Stage) -> dict[str, object]:
    # Midpoints of the configured pair distributions, converted to per-surface
    # coefficients for PhysX's average combine rule.
    white_material = _define_contact_material(
        stage,
        WHITE_CONTACT_MATERIAL,
        static_friction=0.525,
        dynamic_friction=0.405,
        restitution=0.03,
    )
    workbench_material = _define_contact_material(
        stage,
        WORKBENCH_CONTACT_MATERIAL,
        static_friction=0.425,
        dynamic_friction=0.325,
        restitution=0.03,
    )
    bound = []
    for path in (TABLE_BODY, *LEG_BODIES):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"missing white-table contact prim: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            white_material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        bound.append(path)
    workbench_top = stage.GetPrimAtPath(WORKBENCH_TOP_COLLIDER)
    if not workbench_top.IsValid() or not workbench_top.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError(f"missing workbench top collider: {WORKBENCH_TOP_COLLIDER}")
    UsdShade.MaterialBindingAPI.Apply(workbench_top).Bind(
        workbench_material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )
    bound.append(WORKBENCH_TOP_COLLIDER)
    return {
        "white_material": WHITE_CONTACT_MATERIAL,
        "workbench_material": WORKBENCH_CONTACT_MATERIAL,
        "bound": bound,
    }


def _world_transform(cache: UsdGeom.XformCache, stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"missing scene prim: {path}")
    return cache.GetLocalToWorldTransform(prim)


def _set_world_pose(prim, position: Gf.Vec3d, rotation: Gf.Quatd) -> None:
    """Write a rigid body's world pose while preserving its parent hierarchy."""
    parent = prim.GetParent()
    if parent and parent.IsValid():
        parent_transform = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
        local_transform = parent_transform.GetInverse() * Gf.Matrix4d().SetRotate(rotation)
        local_transform.SetTranslateOnly(
            parent_transform.GetInverse().Transform(position)
        )
    else:
        local_transform = Gf.Matrix4d().SetRotate(rotation)
        local_transform.SetTranslateOnly(position)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()
    scale_op = xformable.AddScaleOp()
    local_translation = local_transform.ExtractTranslation()
    if str(translate_op.GetAttr().GetTypeName()) == "float3":
        translate_op.Set(
            Gf.Vec3f(
                float(local_translation[0]),
                float(local_translation[1]),
                float(local_translation[2]),
            )
        )
    else:
        translate_op.Set(Gf.Vec3d(local_translation))
    local_rotation = local_transform.ExtractRotationQuat()
    if str(orient_op.GetAttr().GetTypeName()) == "quatf":
        orient_op.Set(
            Gf.Quatf(
                float(local_rotation.GetReal()),
                Gf.Vec3f(
                    float(local_rotation.GetImaginary()[0]),
                    float(local_rotation.GetImaginary()[1]),
                    float(local_rotation.GetImaginary()[2]),
                ),
            )
        )
    else:
        orient_op.Set(local_rotation)
    if str(scale_op.GetAttr().GetTypeName()) == "float3":
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    else:
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _zero_rigid_body_velocities(stage: Usd.Stage) -> list[str]:
    """Remove source-scene motion before PhysX builds the fixed assembly.

    Scene02 is a captured organizer state rather than a clean construction
    pose. Its four loose legs carry large authored angular velocities. Leaving
    those values in the derived scene lets PhysX apply constraint impulses when
    the fixed joints are first instantiated, before the task reset can clear
    tensor state.
    """

    zero = Gf.Vec3f(0.0, 0.0, 0.0)
    zeroed = []
    for path in ASSEMBLED_RIGID_BODIES:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"missing rigid body while clearing velocity: {path}")
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_body:
            raise ValueError(f"prim is not a rigid body while clearing velocity: {path}")
        rigid_body.CreateVelocityAttr().Set(zero)
        rigid_body.CreateAngularVelocityAttr().Set(zero)
        zeroed.append(path)
    return zeroed


def _activate_leg_contact_reporting(stage: Usd.Stage) -> dict[str, object]:
    """Enable body-level reports while preserving organizer collision geometry.

    Each organizer leg contains many collision shapes. PhysX GPU contact
    filtering cannot target one shape, so grasp attribution is measured in the
    supported direction: one sensor per leg body, filtered against the four
    single-shape Dex1 finger bodies.
    """

    records = []
    for path in LEG_BODIES:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"missing rigid leg body for contact reporting: {path}")

        if "PhysxRigidBodyAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxContactReportAPI")
        prim.CreateAttribute(
            "physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float
        ).Set(0.0)
        prim.CreateAttribute(
            "physxContactReport:threshold", Sdf.ValueTypeNames.Float
        ).Set(0.0)

        enabled_colliders = []
        for descendant in Usd.PrimRange(prim):
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                enabled_colliders.append(str(descendant.GetPath()))
        shaft_path = f"{path}{LEG_SHAFT_COLLIDER_SUFFIX}"
        shaft_prim = stage.GetPrimAtPath(shaft_path)
        if shaft_path not in enabled_colliders or not shaft_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"missing enabled leg shaft collider: {shaft_path}")
        extent = UsdGeom.Mesh(shaft_prim).GetExtentAttr().Get()
        if extent is None or len(extent) != 2:
            raise ValueError(f"missing leg shaft extent: {shaft_path}")
        dimensions = [float(extent[1][axis] - extent[0][axis]) for axis in range(3)]
        if not (
            0.038 <= dimensions[0] <= 0.050
            and 0.038 <= dimensions[1] <= 0.050
            and 0.36 <= dimensions[2] <= 0.40
        ):
            raise ValueError(
                f"unexpected leg shaft dimensions at {shaft_path}: {dimensions}"
            )
        records.append(
            {
                "body": path,
                "enabled_collision_shape_count": len(enabled_colliders),
                "shaft_collider": shaft_path,
                "shaft_dimensions_m": dimensions,
                "contact_report_threshold_n": 0.0,
            }
        )
    return {
        "strategy": "one-leg-body sensor filtered against four Dex1 finger bodies",
        "source_detailed_collision_geometry_available": True,
        "legs": records,
    }


def _disambiguate_leg_contact_reporter_names(stage: Usd.Stage) -> list[dict[str, str]]:
    """Give each leg visual a name distinct from its rigid body.

    Isaac Lab's PhysX contact sensor resolves the configured parent and then
    searches its descendants by leaf name. The organizer assets call both the
    rigid body and its visual mesh ``Leg001``. With cloned environments this
    makes the sensor construct two body patterns per leg and the filtered
    contact view fails to initialize. Renaming only the visual prim preserves
    all rendered and collision geometry while making body resolution unique.
    """

    editor = Usd.NamespaceEditor(stage)
    renamed = []
    for body_path in LEG_BODIES:
        source_path = f"{body_path}{LEG_VISUAL_MESH_SUFFIX}"
        target_path = f"{body_path}/Visuals/{LEG_VISUAL_MESH_NAME}"
        source = stage.GetPrimAtPath(source_path)
        if not source.IsValid() or not source.IsA(UsdGeom.Mesh):
            raise ValueError(f"missing leg visual mesh: {source_path}")
        if stage.GetPrimAtPath(target_path).IsValid():
            raise ValueError(f"leg visual target already exists: {target_path}")
        if not editor.RenamePrim(source, LEG_VISUAL_MESH_NAME):
            raise RuntimeError(f"could not queue leg visual rename: {source_path}")
        if not editor.CanApplyEdits() or not editor.ApplyEdits():
            raise RuntimeError(f"could not rename leg visual: {source_path}")
        if stage.GetPrimAtPath(source_path).IsValid() or not stage.GetPrimAtPath(
            target_path
        ).IsValid():
            raise RuntimeError(f"leg visual rename did not take effect: {source_path}")
        renamed.append({"source": source_path, "target": target_path})
    return renamed


def _simplify_white_table_collision(stage: Usd.Stage) -> dict[str, object]:
    """Keep task-relevant outer geometry while removing threaded internals."""

    per_body = []
    for body_path in (TABLE_BODY, *LEG_BODIES):
        body = stage.GetPrimAtPath(body_path)
        if not body.IsValid():
            raise ValueError(f"missing white-table body: {body_path}")
        enabled_before = 0
        disabled = 0
        enabled_after = 0
        for prim in Usd.PrimRange(body):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision = UsdPhysics.CollisionAPI(prim)
            enabled = collision.GetCollisionEnabledAttr().Get() is not False
            enabled_before += int(enabled)
            keep_leg_external = (
                body_path in LEG_BODIES
                and any(
                    str(prim.GetPath()) == f"{body_path}{suffix}"
                    for suffix in LEG_EXTERNAL_COLLIDER_SUFFIXES
                )
            )
            if enabled and not keep_leg_external:
                collision.CreateCollisionEnabledAttr(False)
                disabled += 1
                enabled = False
            enabled_after += int(enabled)

        if body_path == TABLE_BODY:
            proxy = UsdGeom.Cube.Define(
                stage,
                f"{TABLE_BODY}/SimplifiedCollisions/TabletopBox",
            )
            proxy.CreateSizeAttr(1.0)
            proxy.AddScaleOp().Set(Gf.Vec3f(0.579987, 0.419991, 0.040322))
            proxy.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr(True)
            white_material = UsdShade.Material.Get(stage, WHITE_CONTACT_MATERIAL)
            if not white_material:
                raise RuntimeError("white contact material is missing for collision proxy")
            UsdShade.MaterialBindingAPI.Apply(proxy.GetPrim()).Bind(
                white_material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
            enabled_after += 1

        expected_after = 1 if body_path == TABLE_BODY else len(LEG_EXTERNAL_COLLIDER_SUFFIXES)
        if enabled_after != expected_after or disabled < 1:
            raise RuntimeError(
                f"white-table collision simplification failed for {body_path}: "
                f"before={enabled_before}, disabled={disabled}, after={enabled_after}"
            )
        per_body.append(
            {
                "body": body_path,
                "enabled_before": enabled_before,
                "disabled_original_colliders": disabled,
                "enabled_after": enabled_after,
            }
        )
    return {
        "enabled_before": sum(item["enabled_before"] for item in per_body),
        "disabled_original_colliders": sum(
            item["disabled_original_colliders"] for item in per_body
        ),
        "enabled_after": sum(item["enabled_after"] for item in per_body),
        "visual_geometry_changed": False,
        "preserved_leg_external_colliders": list(LEG_EXTERNAL_COLLIDER_SUFFIXES),
        "bodies": per_body,
    }


def _scene_geometry_report(stage: Usd.Stage) -> dict[str, object]:
    """Measure visible task geometry in world coordinates from the organizer USD."""

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )

    def bounds(path: str) -> dict[str, object]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"missing geometry for bounds audit: {path}")
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = box.GetMin()
        maximum = box.GetMax()
        return {
            "path": path,
            "minimum_m": [float(minimum[index]) for index in range(3)],
            "maximum_m": [float(maximum[index]) for index in range(3)],
            "size_m": [float(maximum[index] - minimum[index]) for index in range(3)],
        }

    return {
        "workbench": bounds(WORKBENCH_BODY),
        "white_tabletop": bounds(TABLE_BODY),
    }


def prepare_scene(
    source: Path,
    output: Path,
    *,
    simplify_white_collision: bool = False,
    dex1_force_calibration: bool = False,
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open USD scene: {source}")

    table_prim = stage.GetPrimAtPath(TABLE_BODY)
    if not table_prim.IsValid():
        raise ValueError(f"missing table body: {TABLE_BODY}")

    # The organizer scene authors the 29 kg assembly workbench as a dynamic
    # rigid body. In the real task it is a stationary fixture; allowing it to
    # recoil changes the contact problem and lets the white table slide with
    # the bench instead of rotating. Keep its collision geometry but make the
    # rigid body kinematic before PhysX creates the scene.
    workbench_prim = stage.GetPrimAtPath(WORKBENCH_BODY)
    if not workbench_prim.IsValid():
        raise ValueError(f"missing workbench body: {WORKBENCH_BODY}")
    workbench_rigid_body = UsdPhysics.RigidBodyAPI(workbench_prim)
    if not workbench_rigid_body:
        raise ValueError(f"workbench is not a rigid body: {WORKBENCH_BODY}")
    workbench_rigid_body.CreateKinematicEnabledAttr(True)
    contact_materials = _prepare_contact_material_bindings(stage)
    renamed_leg_visuals = _disambiguate_leg_contact_reporter_names(stage)
    leg_contact_reporting = _activate_leg_contact_reporting(stage)
    removed_physics_scenes = _remove_nested_physics_scenes(stage)

    removed = []
    for body_prim in (table_prim, workbench_prim):
        for child in list(body_prim.GetChildren()):
            if child.GetTypeName() == "PhysicsFixedJoint" or child.GetName() == "FixedJoint":
                removed.append(str(child.GetPath()))
                stage.RemovePrim(child.GetPath())
                # Scene02.usd may author this prim through a referenced layer;
                # an explicit inactive override makes the deletion survive export.
                stage.OverridePrim(child.GetPath()).SetActive(False)

    cache = UsdGeom.XformCache()
    table_transform = _world_transform(cache, stage, TABLE_BODY)
    table_position = table_transform.ExtractTranslation()
    table_rotation = table_transform.ExtractRotationQuat()
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    created = []

    # Move each leg registration site onto the corresponding tabletop site
    # before creating the fixed joints. This makes the pre-start PhysX scene
    # agree with the reset pose used by the evaluation task.
    table_site_paths = (
        "/World/Table001/Table001_01/Sites/reg_int1",
        "/World/Table001/Table001_01/Sites/reg_int2",
        "/World/Table001/Table001_01/Sites/reg_int3",
        "/World/Table001/Table001_01/Sites/reg_int4",
    )
    leg_site_paths = tuple(f"{path}/Sites/reg_int1" for path in LEG_BODIES)
    for table_site_path, leg_path, leg_site_path in zip(table_site_paths, LEG_BODIES, leg_site_paths):
        leg_prim = stage.GetPrimAtPath(leg_path)
        leg_site = stage.GetPrimAtPath(leg_site_path)
        table_site = stage.GetPrimAtPath(table_site_path)
        if not leg_prim.IsValid() or not leg_site.IsValid() or not table_site.IsValid():
            raise ValueError(f"missing registration site for {leg_path}")
        leg_transform = _world_transform(cache, stage, leg_path)
        leg_site_transform = _world_transform(cache, stage, leg_site_path)
        leg_site_offset = leg_transform.GetInverse().Transform(
            leg_site_transform.ExtractTranslation()
        )
        target_site = _world_transform(cache, stage, table_site_path).ExtractTranslation()
        leg_position = target_site - Gf.Matrix4d().SetRotate(LEG_ASSEMBLED_QUAT).TransformDir(
            leg_site_offset
        )
        _set_world_pose(leg_prim, leg_position, LEG_ASSEMBLED_QUAT)

    cache = UsdGeom.XformCache()
    for index, leg_path in enumerate(LEG_BODIES):
        leg_prim = stage.GetPrimAtPath(leg_path)
        if not leg_prim.IsValid():
            raise ValueError(f"missing leg body: {leg_path}")
        leg_transform = _world_transform(cache, stage, leg_path)
        leg_inverse = leg_transform.GetInverse()
        relative_position = leg_inverse.Transform(table_position)
        relative_rotation = leg_transform.ExtractRotationQuat().GetInverse() * table_rotation

        joint_path = f"{TABLE_BODY}/{JOINT_PREFIX}{index}"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([table_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([leg_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(float(relative_position[0]), float(relative_position[1]), float(relative_position[2]))
        )
        joint.CreateLocalRot0Attr().Set(identity)
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(
                float(relative_rotation.GetReal()),
                Gf.Vec3f(
                    float(relative_rotation.GetImaginary()[0]),
                    float(relative_rotation.GetImaginary()[1]),
                    float(relative_rotation.GetImaginary()[2]),
                ),
            )
        )
        created.append(joint_path)

    zeroed_rigid_body_velocities = _zero_rigid_body_velocities(stage)
    collision_simplification = (
        _simplify_white_table_collision(stage)
        if simplify_white_collision
        else None
    )
    collision_contract = {
        "mode": (
            "task_external_proxy"
            if collision_simplification is not None
            else "organizer_detailed"
        ),
        "visual_geometry_changed": False,
        "internal_thread_colliders_enabled": collision_simplification is None,
        "active_white_colliders": (
            collision_simplification["enabled_after"]
            if collision_simplification is not None
            else None
        ),
    }
    scene_geometry = _scene_geometry_report(stage)
    dex1_force_blockers = (
        _add_dex1_force_calibration_blockers(stage)
        if dex1_force_calibration
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.Export(str(output)):
        raise RuntimeError(f"failed to export patched USD: {output}")
    return {
        "source": str(source),
        "output": str(output),
        "removed": removed,
        "created": created,
        "zeroed_rigid_body_velocities": zeroed_rigid_body_velocities,
        "workbench_kinematic": True,
        "contact_materials": contact_materials,
        "renamed_leg_visuals": renamed_leg_visuals,
        "leg_contact_reporting": leg_contact_reporting,
        "removed_physics_scenes": removed_physics_scenes,
        "collision_simplification": collision_simplification,
        "collision_contract": collision_contract,
        "scene_geometry": scene_geometry,
        "dex1_force_calibration": dex1_force_blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplify-white-collision", action="store_true")
    parser.add_argument("--dex1-force-calibration", action="store_true")
    args = parser.parse_args()
    report = prepare_scene(
        args.source,
        args.output,
        simplify_white_collision=args.simplify_white_collision,
        dex1_force_calibration=args.dex1_force_calibration,
    )
    workbench = report["scene_geometry"]["workbench"]
    print(
        f"[flip_table] prepared assembled scene: output={report['output']}, "
        f"removed={len(report['removed'])}, fixed_joints={len(report['created'])}, "
        f"workbench_kinematic={report['workbench_kinematic']}, "
        f"nested_physics_scenes_removed={len(report['removed_physics_scenes'])}, "
        f"rigid_body_velocities_zeroed={len(report['zeroed_rigid_body_velocities'])}, "
        f"contact_bindings={len(report['contact_materials']['bound'])}, "
        f"leg_visuals_renamed={len(report['renamed_leg_visuals'])}, "
        f"leg_contact_reporters={len(report['leg_contact_reporting']['legs'])}, "
        f"white_collision_mode={report['collision_contract']['mode']}, "
        f"active_white_colliders="
        f"{report['collision_contract']['active_white_colliders'] or 'detailed'}, "
        f"dex1_force_calibration={report['dex1_force_calibration'] is not None}, "
        f"workbench_size_m={[round(value, 6) for value in workbench['size_m']]}, "
        f"workbench_top_z_m={workbench['maximum_m'][2]:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
