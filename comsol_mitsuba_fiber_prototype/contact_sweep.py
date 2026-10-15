#!/usr/bin/env python3
"""
Silicone contact on the apex of the U-bend: sweep the overlap depth.

A rectangular silicone block (n = ior.silicone) is lowered along -z onto the
outer apex of the 180 deg bend so that it overlaps the fiber by `d`. The fiber
is not deformed; the silicone is assumed to conform to it (perfect optical
contact) wherever they overlap:

  * cladding faces whose centroid lies above the block bottom (z > z_p)
    become cladding <-> silicone (cladding_contact)
  * the block's surface bordering air (top, sides, and the bottom face with
    the fiber's exact cross-section cut out) is silicone <-> air
    (silicone_body), so light that enters the silicone is traced further

    python contact_sweep.py --config test_bent_fiber/config.yaml \\
        --depths 0 0.05 0.10 0.15 0.20 0.25 --thetas 15 30

Lengths in mm. Writes contact_sweep/<case>/ meshes and contact_sweep/results.csv.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path

import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import trimesh
import yaml
from trimesh.creation import triangulate_polygon

from make_test_fiber import build_fiber


def fiber_section(z_p, L1, R, b, n=2000):
    """Exact cross-section of the bent cladding (torus part) with the plane
    z = z_p (> L1). The bend lies in the xz-plane around (R, 0, L1)."""
    h = z_p - L1
    y_max = b if h <= R else math.sqrt(max(b * b - (h - R) ** 2, 0.0))
    if y_max <= 0:
        return None
    t = np.linspace(0.0, math.pi, n)
    y = -y_max * np.cos(t)                      # dense near the ends
    s = np.sqrt(np.maximum(b * b - y * y, 0.0))
    xo = np.sqrt(np.maximum((R + s) ** 2 - h * h, 0.0))
    xi = np.sqrt(np.maximum((R - s) ** 2 - h * h, 0.0))
    right = sg.Polygon(np.r_[np.c_[R + xo, y], np.c_[R + xi, y][::-1]]).buffer(0)
    left = sg.Polygon(np.r_[np.c_[R - xo, y], np.c_[R - xi, y][::-1]]).buffer(0)
    return so.unary_union([right, left])


def silicone_block(z_p, L1, R, b, size_xy, height):
    """Silicone surface bordering air, normals pointing out of the silicone."""
    x0, x1 = R - size_xy / 2, R + size_xy / 2
    y0, y1 = -size_xy / 2, size_xy / 2
    box = trimesh.creation.box(bounds=[[x0, y0, z_p], [x1, y1, z_p + height]])
    keep = box.face_normals[:, 2] > -0.5          # drop the plain bottom face
    parts = [trimesh.Trimesh(box.vertices, box.faces[keep], process=False)]

    bottom = sg.box(x0, y0, x1, y1)
    hole = fiber_section(z_p, L1, R, b)
    if hole is not None and not hole.is_empty:
        bottom = bottom.difference(hole)
    polys = list(bottom.geoms) if hasattr(bottom, "geoms") else [bottom]
    for poly in polys:
        v2, f = triangulate_polygon(poly, engine="earcut")
        v3 = np.c_[v2, np.full(len(v2), z_p)]
        m = trimesh.Trimesh(v3, f, process=False)
        if m.face_normals[:, 2].mean() > 0:       # bottom must face -z
            m.invert()
        parts.append(m)
    return trimesh.util.concatenate(parts)


def build_case(out, d, cfg_template, geo, R, b, size_xy, height):
    L1 = geo["L1"]
    z_top = L1 + R + b
    z_p = z_top - d
    out.mkdir(parents=True, exist_ok=True)

    cv, cf = geo["cladding_outer"]
    clad = trimesh.Trimesh(cv, cf, process=False)
    contact = clad.triangles_center[:, 2] > z_p
    names = {
        "core_boundary": trimesh.Trimesh(*geo["core_boundary"], process=False),
        "cladding_air": clad.submesh([np.flatnonzero(~contact)], append=True),
        "detector": trimesh.Trimesh(*geo["detector"], process=False),
        "silicone_body": silicone_block(z_p, L1, R, b, size_xy, height),
    }
    if contact.any():
        names["cladding_contact"] = clad.submesh([np.flatnonzero(contact)], append=True)
    for k, m in names.items():
        m.export(out / f"{k}.ply")

    cfg = copy.deepcopy(cfg_template)
    cfg["meshes"] = {k: f"{k}.ply" for k in
                     ["core_boundary", "cladding_air", "cladding_contact",
                      "detector", "silicone_body"]}
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    area = float(clad.area_faces[contact].sum()) if contact.any() else 0.0
    block_box = (R - size_xy / 2, R + size_xy / 2, -size_xy / 2, size_xy / 2,
                 z_p, z_p + height)
    return out / "config.yaml", area, block_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="test_bent_fiber/config.yaml",
                    help="template (ior, simulation, detector settings)")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ap.add_argument("--thetas", type=float, nargs="+", default=[15.0, 30.0])
    ap.add_argument("--rays", type=int, default=1_000_000)
    ap.add_argument("--length", type=float, default=50.0)
    ap.add_argument("--bend-radius", type=float, default=0.5)
    ap.add_argument("--core-diameter", type=float, default=0.240)
    ap.add_argument("--clad-diameter", type=float, default=0.250)
    ap.add_argument("--block-size", type=float, default=3.0, help="block x/y size")
    ap.add_argument("--block-height", type=float, default=1.0)
    ap.add_argument("--out", default="contact_sweep")
    args = ap.parse_args()

    import mitsuba as mi
    import trace_mitsuba as tm

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_t = yaml.safe_load(f)
    mi.set_variant(cfg_t["simulation"].get("variant", "cuda_ad_rgb"))
    off = float(cfg_t["simulation"].get("ray_offset_rel", 2.0**-20))

    R, b = args.bend_radius, 0.5 * args.clad_diameter
    geo = build_fiber(args.length, R, 0.5 * args.core_diameter, b)
    out_root = Path(args.out)
    rows = []
    for d in args.depths:
        case = out_root / f"depth_{round(d * 1000):03d}um"
        cfg_path, area, bb = build_case(case, d, cfg_t, geo, R, b,
                                        args.block_size, args.block_height)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        scene = tm.load_scene(cfg, case)
        det = tm.detector_settings(cfg, scene)
        for th in args.thetas:
            cfg["source"]["theta_max_deg"] = th
            o, dv, _ = tm.generate_launch_rays(cfg, cfg["ior"]["core"], args.rays,
                                               int(cfg["simulation"].get("seed", 1)))
            r = tm.trace(scene, o, dv, int(cfg["simulation"]["max_bounces"]),
                         int(cfg["simulation"].get("seed", 1)) + 1000, det=det,
                         offset_rel=off, return_escapes=True)
            eo = r["escape_origin"][r["escaped_mask"]]
            tol = 1e-3   # escape origins sit ~offset outside the surface
            in_block = ((eo[:, 0] > bb[0] - tol) & (eo[:, 0] < bb[1] + tol) &
                        (eo[:, 1] > bb[2] - tol) & (eo[:, 1] < bb[3] + tol) &
                        (eo[:, 2] > bb[4] - tol) & (eo[:, 2] < bb[5] + tol))
            n = r["launched"]
            row = {
                "depth_um": round(d * 1000),
                "theta_deg": th,
                "contact_area_mm2": area,
                "reach_end": r["transmission"],
                "reach_core": r["reach_end_core"],
                "reach_cladding": r["reach_end_cladding"],
                "pout_pin": r["exit_total"],
                "lost_via_silicone": in_block.sum() / n,
                "lost_elsewhere": (~in_block).sum() / n,
                "trapped_max_bounces": r["terminated_at_max_bounces"] / n,
            }
            rows.append(row)
            print(f"d={row['depth_um']:3d} um  theta={th:4.1f}  area={area:.4f} mm2  "
                  f"Pout/Pin={row['pout_pin']:.4f}  reach={row['reach_end']:.4f}  "
                  f"lost(silicone)={row['lost_via_silicone']:.4f}  "
                  f"lost(other)={row['lost_elsewhere']:.4f}  "
                  f"trapped={row['trapped_max_bounces']:.4f}", flush=True)

    with open(out_root / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nresults written to {(out_root / 'results.csv').resolve()}")


if __name__ == "__main__":
    main()
