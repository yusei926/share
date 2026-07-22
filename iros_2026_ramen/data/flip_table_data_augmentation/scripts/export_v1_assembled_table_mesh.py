#!/usr/bin/env python3
"""Export the exact assembled V1 white table as a body-frame OBJ mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from pxr import Usd, UsdGeom

from data.flip_table_data_augmentation.io_utils import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


SCHEMA_VERSION = "team_ramen_v1_assembled_table_mesh/v1"
TABLE_BODY_PATH = "/World/Table001/Table001_01"
VISUAL_MESH_PATHS = (
    "/World/Table001/Table001_01/Visuals/Table001",
    "/World/Leg001/Leg001/Visuals/Leg001_visual",
    "/World/Leg001_01/Leg001/Visuals/Leg001_visual",
    "/World/Leg001_03/Leg001/Visuals/Leg001_visual",
    "/World/Leg001_06/Leg001/Visuals/Leg001_visual",
)
EXPECTED_MIN_M = np.asarray((-0.289993405, -0.209995389, -0.4073812))
EXPECTED_MAX_M = np.asarray((0.289993405, 0.209995389, 0.020159841))


def _triangles(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles = []
    cursor = 0
    for count in counts:
        count = int(count)
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            raise ValueError("USD mesh contains a face with fewer than three vertices")
        triangles.extend((int(face[0]), int(face[index]), int(face[index + 1])) for index in range(1, count - 1))
    if cursor != len(indices):
        raise ValueError("USD face counts do not consume the face vertex index array")
    return np.asarray(triangles, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled-scene", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    scene_path = args.assembled_scene.expanduser().resolve()
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise FileNotFoundError(f"could not open V1 assembled scene: {scene_path}")
    table_body = stage.GetPrimAtPath(TABLE_BODY_PATH)
    if not table_body.IsValid():
        raise ValueError(f"assembled scene lacks table body {TABLE_BODY_PATH}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body_world_inverse = cache.GetLocalToWorldTransform(table_body).GetInverse()

    all_vertices = []
    all_triangles = []
    mesh_records = []
    vertex_offset = 0
    for path in VISUAL_MESH_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"assembled scene lacks visual mesh {path}")
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not points or counts.size == 0 or indices.size == 0:
            raise ValueError(f"visual mesh is empty: {path}")
        mesh_to_body = cache.GetLocalToWorldTransform(prim) * body_world_inverse
        if float(mesh_to_body.GetDeterminant3()) <= 0.0:
            raise ValueError(f"visual mesh transform reflects or collapses geometry: {path}")
        vertices = np.asarray([mesh_to_body.Transform(point) for point in points], dtype=np.float64)
        triangles = _triangles(counts, indices) + vertex_offset
        all_vertices.append(vertices)
        all_triangles.append(triangles)
        mesh_records.append(
            {
                "prim_path": path,
                "vertex_count": int(len(vertices)),
                "triangle_count": int(len(triangles)),
                "bounds_min_m": vertices.min(axis=0).tolist(),
                "bounds_max_m": vertices.max(axis=0).tolist(),
            }
        )
        vertex_offset += len(vertices)

    vertices = np.concatenate(all_vertices, axis=0)
    triangles = np.concatenate(all_triangles, axis=0)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    if not np.allclose(bounds_min, EXPECTED_MIN_M, atol=2.0e-6) or not np.allclose(
        bounds_max, EXPECTED_MAX_M, atol=2.0e-6
    ):
        raise ValueError(
            "assembled table bounds differ from the pinned V1 geometry: "
            f"min={bounds_min.tolist()} max={bounds_max.tolist()}"
        )

    lines = [
        "# RoboFinals-IKEA-V1 assembled Table001 in /World/Table001/Table001_01 body frame",
        "o Table001_assembled",
    ]
    lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    atomic_write_text(args.output_obj.expanduser().resolve(), "\n".join(lines) + "\n")
    obj_path = args.output_obj.expanduser().resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_assembled_scene": str(scene_path),
        "source_assembled_scene_sha256": sha256_file(scene_path),
        "table_body_frame": TABLE_BODY_PATH,
        "visual_meshes": mesh_records,
        "output_obj": str(obj_path),
        "output_obj_sha256": sha256_file(obj_path),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "bounds_min_m": bounds_min.tolist(),
        "bounds_max_m": bounds_max.tolist(),
        "dimensions_m": (bounds_max - bounds_min).tolist(),
    }
    atomic_write_json(args.manifest.expanduser().resolve(), manifest)
    print(json.dumps({key: manifest[key] for key in ("output_obj_sha256", "vertex_count", "triangle_count", "dimensions_m")}, sort_keys=True))


if __name__ == "__main__":
    main()
