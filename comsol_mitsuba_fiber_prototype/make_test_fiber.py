#!/usr/bin/env python3
"""
Generate a synthetic U-bent step-index fiber as COMSOL-like STL exports.

Use this to check the prepare_meshes.py -> trace_mitsuba.py pipeline without
COMSOL. Geometry (all lengths in mm by default):

    leg 1 (straight, +z)  ->  180 deg arc (centerline radius R)  ->  leg 2 (-z)

The input end face is at z = 0 centered on the origin. The output end face is
at (2R, 0, L1). The arc lies in the xz-plane.

Outputs (same names prepare_meshes.py expects):
  core_boundary.stl   lateral core/cladding interface, normals out of the core
  cladding_outer.stl  lateral outer cladding surface, normals outward
  detector.stl        output end face (disk covering core + cladding)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import trimesh


def centerline(s, L1, R):
    """Centerline point c, normal N (toward bend center) and binormal B at
    arclength s. Returns arrays of shape (len(s), 3)."""
    s = np.asarray(s, dtype=np.float64)
    if R is None:
        R, L1 = 1.0, np.inf
    arc = math.pi * R
    c = np.zeros((len(s), 3))
    N = np.zeros((len(s), 3))
    B = np.tile([0.0, 1.0, 0.0], (len(s), 1))

    leg1 = s <= L1
    bend = (s > L1) & (s < L1 + arc)
    leg2 = s >= L1 + arc

    c[leg1] = np.stack([0 * s[leg1], 0 * s[leg1], s[leg1]], axis=1)
    N[leg1] = [1.0, 0.0, 0.0]

    th = (s[bend] - L1) / R
    c[bend] = np.stack([R - R * np.cos(th), 0 * th, L1 + R * np.sin(th)], axis=1)
    N[bend] = np.stack([np.cos(th), 0 * th, -np.sin(th)], axis=1)

    s2 = s[leg2] - L1 - arc
    c[leg2] = np.stack([0 * s2 + 2 * R, 0 * s2, L1 - s2], axis=1)
    N[leg2] = [-1.0, 0.0, 0.0]
    return c, N, B


def stations(length, R, straight_step, bend_segments):
    """Arclength samples: coarse on straight legs, fine in the bend.
    R = None gives a straight fiber (leg 1 only)."""
    if R is None:
        n = max(1, math.ceil(length / straight_step))
        return np.linspace(0.0, length, n + 1), length
    arc = math.pi * R
    L1 = 0.5 * (length - arc)
    if L1 <= 0:
        raise ValueError(f"length {length} too short for a 180 deg bend of radius {R}")
    n_leg = max(1, math.ceil(L1 / straight_step))
    s_leg1 = np.linspace(0.0, L1, n_leg + 1)
    s_bend = np.linspace(L1, L1 + arc, bend_segments + 1)[1:-1]
    s_leg2 = np.linspace(L1 + arc, length, n_leg + 1)
    return np.concatenate([s_leg1, s_bend, s_leg2]), L1


def tube(s, L1, R, radius, n_phi):
    """Swept circular tube without end caps, outward-facing winding."""
    c, N, B = centerline(s, L1, R)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    ring = np.cos(phi)[None, :, None] * N[:, None, :] + np.sin(phi)[None, :, None] * B[:, None, :]
    verts = (c[:, None, :] + radius * ring).reshape(-1, 3)

    i = np.arange(len(s) - 1)[:, None]
    j = np.arange(n_phi)[None, :]
    jn = (j + 1) % n_phi
    v00 = i * n_phi + j
    v01 = i * n_phi + jn
    v10 = (i + 1) * n_phi + j
    v11 = (i + 1) * n_phi + jn
    # (v00, v01, v10): normal = B x T = +N at phi = 0, i.e. outward.
    faces = np.concatenate([
        np.stack([v00, v01, v10], axis=-1).reshape(-1, 3),
        np.stack([v01, v11, v10], axis=-1).reshape(-1, 3),
    ])
    return verts, faces


def end_disk(s_end, L1, R, radius, n_phi):
    """Triangle-fan disk sharing the cladding tube's end-ring vertices."""
    c, N, B = centerline(np.array([s_end]), L1, R)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    ring = c[0] + radius * (np.cos(phi)[:, None] * N[0] + np.sin(phi)[:, None] * B[0])
    verts = np.vstack([c, ring])
    j = np.arange(n_phi)
    faces = np.stack([np.zeros(n_phi, dtype=np.int64), 1 + j, 1 + (j + 1) % n_phi], axis=1)
    return verts, faces


def build_fiber(length, bend_radius, core_radius, clad_radius,
                n_phi=720, bend_segments=360, straight_step=1.0):
    s, L1 = stations(length, bend_radius, straight_step, bend_segments)
    core = tube(s, L1, bend_radius, core_radius, n_phi)
    clad = tube(s, L1, bend_radius, clad_radius, n_phi)
    det = end_disk(length, L1, bend_radius, clad_radius, n_phi)
    out_center, _, _ = centerline(np.array([length]), L1, bend_radius)
    return {
        "core_boundary": core,
        "cladding_outer": clad,
        "detector": det,
        "L1": L1,
        "output_center": out_center[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=50.0, help="total centerline length")
    ap.add_argument("--bend-radius", type=float, default=0.5, help="centerline bend radius")
    ap.add_argument("--core-diameter", type=float, default=0.240)
    ap.add_argument("--clad-diameter", type=float, default=0.250)
    ap.add_argument("--straight", action="store_true", help="no bend (validation case)")
    ap.add_argument("--n-phi", type=int, default=720, help="segments around the circumference")
    ap.add_argument("--bend-segments", type=int, default=360, help="segments along the 180 deg bend")
    ap.add_argument("--straight-step", type=float, default=1.0, help="ring spacing on straight legs")
    ap.add_argument("--out", default="test_fiber_stl")
    args = ap.parse_args()

    geo = build_fiber(args.length, None if args.straight else args.bend_radius,
                      0.5 * args.core_diameter, 0.5 * args.clad_diameter,
                      args.n_phi, args.bend_segments, args.straight_step)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["core_boundary", "cladding_outer", "detector"]:
        v, f = geo[name]
        trimesh.Trimesh(v, f, process=False).export(out / f"{name}.stl")
        print(f"{name:15s}: {len(f):8d} faces")

    print(f"straight leg length : {geo['L1']:.6f}")
    print(f"input  face center  : [0, 0, 0], axis +z")
    print(f"output face center  : {geo['output_center'].tolist()}")
    print(f"STL written to      : {out.resolve()}")


if __name__ == "__main__":
    main()
