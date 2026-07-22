"""Read back episode-fixed physical randomization values from the V1 runtime."""

from __future__ import annotations

from typing import Any

import torch


TABLE_ENTITIES = (
    "Table001_Table001_01",
    "Leg001_Leg001",
    "Leg001_01_Leg001",
    "Leg001_03_Leg001",
    "Leg001_06_Leg001",
)
CONTACT_MATERIAL_SUFFIXES = {
    "hand": "Robot/Looks/flip_table_contact_hand",
    "white_table": "Scene/Looks/flip_table_contact_white",
    "workbench": "Scene/Looks/flip_table_contact_workbench",
}


def snapshot_physical_randomization(env, env_id: int = 0) -> dict[str, Any]:
    arena = getattr(env.cfg, "isaaclab_arena_env", None)
    task = getattr(arena, "task", None)
    if task is None or not hasattr(task, "_find_prim_by_suffix"):
        raise RuntimeError("V1 task cannot expose physical randomization provenance")

    masses = {}
    for entity_name in TABLE_ENTITIES:
        entity = env.scene.rigid_objects.get(entity_name)
        if entity is None:
            raise RuntimeError(f"randomized table entity is missing: {entity_name}")
        body_mass = torch.as_tensor(entity.data.body_mass).reshape(env.num_envs, -1)
        values = body_mass[env_id].detach().cpu().to(dtype=torch.float64).tolist()
        if not values or any(value <= 0.0 for value in values):
            raise RuntimeError(f"invalid randomized mass for {entity_name}: {values}")
        masses[entity_name] = values

    materials = {}
    for surface_name, suffix in CONTACT_MATERIAL_SUFFIXES.items():
        prim = task._find_prim_by_suffix(env, suffix, env_id=env_id)
        if prim is None:
            raise RuntimeError(f"contact material is missing: {suffix}")
        values = {}
        for output_name, attribute_name in (
            ("static_friction", "physics:staticFriction"),
            ("dynamic_friction", "physics:dynamicFriction"),
            ("restitution", "physics:restitution"),
        ):
            attribute = prim.GetAttribute(attribute_name)
            if not attribute or not attribute.HasAuthoredValueOpinion():
                raise RuntimeError(f"{suffix} lacks {attribute_name}")
            values[output_name] = float(attribute.Get())
        if values["static_friction"] < values["dynamic_friction"]:
            raise RuntimeError(f"contact material violates static >= dynamic: {surface_name}")
        if not 0.0 <= values["restitution"] <= 0.2:
            raise RuntimeError(f"contact material restitution is unrealistic: {surface_name}")
        materials[surface_name] = values

    return {
        "table_body_masses_kg": masses,
        "contact_surface_materials": materials,
        "contact_combine_mode": "average",
        "robot_root_locked": True,
        "lower_body_locked": True,
    }
