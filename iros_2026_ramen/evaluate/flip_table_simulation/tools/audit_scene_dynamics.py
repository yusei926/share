#!/usr/bin/env python3
"""Audit rigid bodies and joints that can prevent the white table from moving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdPhysics


BODY_PATHS = (
    "/World/Table001/Table001_01",
    "/World/Leg001/Leg001",
    "/World/Leg001_01/Leg001",
    "/World/Leg001_03/Leg001",
    "/World/Leg001_06/Leg001",
    "/World/Table278/Table278",
)


def _attribute_value(prim: Usd.Prim, name: str):
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute and attribute.HasAuthoredValue() else None


def _body_report(stage: Usd.Stage, path: str) -> dict[str, object]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return {"path": path, "valid": False}
    rigid = UsdPhysics.RigidBodyAPI(prim)
    mass = UsdPhysics.MassAPI(prim)
    ancestors = []
    ancestor = prim.GetParent()
    while ancestor and ancestor.IsValid() and str(ancestor.GetPath()) != "/":
        ancestor_rigid = UsdPhysics.RigidBodyAPI(ancestor)
        if ancestor_rigid or ancestor.HasAPI(UsdPhysics.ArticulationRootAPI):
            ancestors.append(
                {
                    "path": str(ancestor.GetPath()),
                    "rigid_body": bool(ancestor_rigid),
                    "rigid_body_enabled": _attribute_value(
                        ancestor, "physics:rigidBodyEnabled"
                    ),
                    "kinematic_enabled": _attribute_value(
                        ancestor, "physics:kinematicEnabled"
                    ),
                    "articulation_root": ancestor.HasAPI(
                        UsdPhysics.ArticulationRootAPI
                    ),
                }
            )
        ancestor = ancestor.GetParent()
    return {
        "path": path,
        "valid": True,
        "type": prim.GetTypeName(),
        "active": prim.IsActive(),
        "instanceable": prim.IsInstanceable(),
        "rigid_body": bool(rigid),
        "rigid_body_enabled": _attribute_value(prim, "physics:rigidBodyEnabled"),
        "kinematic_enabled": _attribute_value(prim, "physics:kinematicEnabled"),
        "mass_kg": mass.GetMassAttr().Get() if mass else None,
        "articulation_root": prim.HasAPI(UsdPhysics.ArticulationRootAPI),
        "ancestors_with_physics": ancestors,
    }


def audit_scene(path: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open USD scene: {path}")

    joints = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(target) for target in joint.GetBody0Rel().GetTargets()]
        body1 = [str(target) for target in joint.GetBody1Rel().GetTargets()]
        joints.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "enabled": _attribute_value(prim, "physics:jointEnabled"),
                "exclude_from_articulation": _attribute_value(
                    prim, "physics:excludeFromArticulation"
                ),
                "body0": body0,
                "body1": body1,
                "world_fixed": not body0 or not body1,
                "involves_white_table": any(
                    target.startswith("/World/Table001")
                    or target.startswith("/World/Leg001")
                    for target in (*body0, *body1)
                ),
            }
        )
    return {
        "scene": str(path),
        "bodies": [_body_report(stage, body_path) for body_path in BODY_PATHS],
        "joints": joints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_scene(args.scene)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
