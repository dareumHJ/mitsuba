#!/usr/bin/env python3
"""
Prepare deformed COMSOL surface meshes for Mitsuba 3.

Inputs:
  core_boundary.stl
  cladding_outer.stl
  silicone_surface.stl
  detector.stl

Outputs:
  core_boundary.ply
  cladding_air.ply
  cladding_contact.ply
  detector.ply

The contact patch is estimated by proximity of cladding face centers to the
deformed silicone surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: str, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path}: expected one triangle mesh, got {type(mesh)}")
    mesh = mesh.copy()
    if scale != 1.0:
        mesh.vertices *= scale
    if len(mesh.faces) == 0:
        raise ValueError(f"{path}: mesh has no faces")
    # STL stores three private vertices per triangle. Weld them: besides
    # saving memory, Mitsuba/OptiX silently drops geometry from meshes with
    # more than 2**24 vertices (observed with Mitsuba 3.9.1 cuda_ad_rgb).
    mesh.merge_vertices()
    if len(mesh.vertices) >= 2**24:
        print(f"[warning] {path}: {len(mesh.vertices)} vertices >= 2**24; "
              "Mitsuba may lose triangles. Coarsen or split this mesh.")
    return mesh


def distance_to_surface(points: np.ndarray, surface: trimesh.Trimesh) -> np.ndarray:
    """
    Preferred: exact nearest-surface query through trimesh.
    Fallback: nearest silicone vertex, which is less accurate but useful for
    coarse prototyping if an optional proximity dependency is unavailable.
    """
    try:
        _, dist, _ = trimesh.proximity.closest_point(surface, points)
        return np.asarray(dist)
    except Exception as exc:
        print(f"[warning] exact trimesh proximity failed: {exc}")
        print("[warning] falling back to nearest silicone vertex (approximate)")
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(surface.vertices))
        dist, _ = tree.query(points, k=1)
        return np.asarray(dist)


def export_submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray, path: Path) -> int:
    ids = np.flatnonzero(face_mask)
    if len(ids) == 0:
        # Mitsuba still needs a valid file; create a tiny off-scene triangle.
        # Normally you should instead omit the corresponding shape from the
        # Mitsuba scene. We keep this explicit so an empty contact patch is
        # obvious to the user.
        print(f"[warning] {path.name}: zero selected faces; not writing file")
        return 0

    sub = mesh.submesh([ids], append=True, repair=False)
    sub.export(path)
    return len(sub.faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--cladding", required=True)
    ap.add_argument("--silicone", default=None,
                    help="Silicone surface. Omit for a no-contact case "
                         "(the whole outer cladding is then cladding_air).")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--contact-tol", type=float, default=None,
                    help="Contact distance threshold in mesh coordinate units.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply all coordinates by this factor.")
    ap.add_argument("--out", default="prepared")
    args = ap.parse_args()
    if args.silicone is not None and args.contact_tol is None:
        ap.error("--contact-tol is required when --silicone is given")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    core = load_mesh(args.core, args.scale)
    clad = load_mesh(args.cladding, args.scale)
    silicone = load_mesh(args.silicone, args.scale) if args.silicone else None
    detector = load_mesh(args.detector, args.scale)

    core.export(out / "core_boundary.ply")
    detector.export(out / "detector.ply")

    centers = np.asarray(clad.triangles_center)
    if silicone is not None:
        dist = distance_to_surface(centers, silicone)
    else:
        dist = np.full(len(centers), np.inf)

    # Remove any stale contact patch from a previous run in the same folder.
    for name in ["cladding_contact.ply", "cladding_air.ply"]:
        (out / name).unlink(missing_ok=True)

    is_contact = dist <= (args.contact_tol if silicone is not None else -np.inf)
    is_air = ~is_contact

    n_contact = export_submesh(clad, is_contact, out / "cladding_contact.ply")
    n_air = export_submesh(clad, is_air, out / "cladding_air.ply")

    print("\n=== Mesh summary ===")
    print(f"core faces      : {len(core.faces)}")
    print(f"outer faces     : {len(clad.faces)}")
    print(f"contact faces   : {n_contact}")
    print(f"air faces       : {n_air}")
    print(f"contact fraction: {is_contact.mean():.6f}")
    print(f"min distance    : {dist.min():.6g}")
    print(f"median distance : {np.median(dist):.6g}")
    print(f"max distance    : {dist.max():.6g}")

    for name, mesh in [
        ("core", core), ("cladding", clad),
        ("silicone", silicone), ("detector", detector)
    ]:
        if mesh is None:
            continue
        b = mesh.bounds
        print(f"{name:10s} bounds: min={b[0]}, max={b[1]}")

    print(f"\nPrepared meshes written to: {out.resolve()}")


if __name__ == "__main__":
    main()
