#!/usr/bin/env python3
"""
Experimental forward ray tracer for deformed fiber meshes using Mitsuba 3.

This is an MVP intended for validation against COMSOL Ray Optics.

The scene uses three dielectric interfaces:
  core -> cladding
  cladding -> air
  cladding -> silicone (contact patch)

The detector is an area-emitting PLY surface only so that the tracing loop can
identify detector intersections. No ordinary light rendering is performed.

The estimator uses Mitsuba's stochastic dielectric BSDF sampling. A ray is
counted as transmitted when it reaches the detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import yaml

import mitsuba as mi
import drjit as dr


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def source_basis(axis):
    z = unit(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, z)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x = unit(np.cross(helper, z))
    y = unit(np.cross(z, x))
    return x, y, z


def generate_launch_rays(cfg, n_core, n_rays, seed):
    src = cfg["source"]
    center = np.asarray(src["center"], dtype=np.float64)
    x, y, z = source_basis(src["axis"])
    radius = float(src["radius"])

    if "theta_max_deg" in src and src["theta_max_deg"] is not None:
        theta_max = math.radians(float(src["theta_max_deg"]))
    else:
        na = float(src["launch_na"])
        s = na / float(n_core)
        if not 0 <= s <= 1:
            raise ValueError(f"launch_na/n_core={s} must lie in [0, 1]")
        theta_max = math.asin(s)

    rng = np.random.default_rng(seed)

    # Uniform point distribution over the circular input aperture.
    u = rng.random(n_rays)
    phi_p = 2 * np.pi * rng.random(n_rays)
    rr = radius * np.sqrt(u)
    origins = (
        center[None, :]
        + (rr * np.cos(phi_p))[:, None] * x[None, :]
        + (rr * np.sin(phi_p))[:, None] * y[None, :]
    )

    # Uniform distribution over the internal acceptance solid angle.
    cos_max = math.cos(theta_max)
    cos_t = 1.0 - rng.random(n_rays) * (1.0 - cos_max)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi_d = 2 * np.pi * rng.random(n_rays)

    dirs = (
        (sin_t * np.cos(phi_d))[:, None] * x[None, :]
        + (sin_t * np.sin(phi_d))[:, None] * y[None, :]
        + cos_t[:, None] * z[None, :]
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    return origins.astype(np.float32), dirs.astype(np.float32), theta_max


def p3(a):
    return mi.Point3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def v3(a):
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def dielectric(int_ior, ext_ior):
    return {
        "type": "dielectric",
        "int_ior": float(int_ior),
        "ext_ior": float(ext_ior),
    }


def load_scene(cfg, base_dir="."):
    m = cfg["meshes"]
    n = cfg["ior"]
    flip = cfg.get("flip_normals", {})
    base_dir = Path(base_dir)

    def ply(key, bsdf, optional=False):
        path = m.get(key)
        if path is not None:
            path = (base_dir / path).resolve()
        if path is None or not path.exists():
            if optional:
                print(f"[info] {key}: no mesh ({path}); skipped")
                return None
            raise FileNotFoundError(f"{key}: mesh not found: {path}")
        return {
            "type": "ply",
            "filename": str(path),
            "face_normals": True,
            "flip_normals": bool(flip.get(key, False)),
            "bsdf": bsdf,
        }

    scene = {
        "type": "scene",

        # Normals should point out of core into cladding.
        "core_boundary": ply("core_boundary", dielectric(n["core"], n["cladding"])),

        # Normals should point out of cladding. Either patch may be empty
        # (no contact, or full contact), so both are optional.
        "cladding_air": ply("cladding_air", dielectric(n["cladding"], n["air"]),
                            optional=True),
        "cladding_contact": ply("cladding_contact",
                                dielectric(n["cladding"], n["silicone"]),
                                optional=True),

        # Optional silicone body surface where it borders air (normals out
        # of the silicone). The part bordering the fiber is cladding_contact.
        "silicone_body": ply("silicone_body", dielectric(n["silicone"], n["air"]),
                             optional=True),

        # The area emitter is just an intersection tag for our custom
        # forward tracer. It is not used as a physical source.
        "detector": ply("detector", {"type": "null"}),
    }
    scene["detector"]["emitter"] = {
        "type": "area",
        "radiance": {"type": "rgb", "value": 1.0},
    }
    scene = {k: v for k, v in scene.items() if v is not None}
    return mi.load_dict(scene)


def detector_settings(cfg, scene):
    """Detector post-processing: core/cladding split and exit Fresnel loss."""
    d = cfg.get("detector", {}) or {}
    shape = next(sh for sh in scene.shapes() if sh.id() == "detector")
    center = d.get("center")
    if center is None:
        center = list(shape.bbox().center())
    return {
        "center": [float(c) for c in center],
        "core_radius": d.get("core_radius"),
        "exit_fresnel": bool(d.get("exit_fresnel", False)),
        "n_core": float(cfg["ior"]["core"]),
        "n_cladding": float(cfg["ior"]["cladding"]),
        "n_out": float(d.get("exit_ior", cfg["ior"]["air"])),
    }


def trace(scene, origins, directions, max_bounces, seed, det=None,
          record_paths=False, escape_len=0.3, offset_rel=2.0**-20,
          return_escapes=False):
    """Stochastic Fresnel-branching forward trace.

    If ``det`` is given (see detector_settings), detector hits are split into
    core / cladding by radial distance from the detector center, and each hit
    is weighted by the Fresnel transmittance from glass into ``n_out``.

    Secondary rays are offset from the surface by
    ``(1 + max|p|) * offset_rel`` along the normal. Mitsuba's own
    spawn_ray() uses ~1500 float32 ulps (about 4.5 um at 50 mm from the
    origin), which is comparable to a 5 um cladding and makes rays skip it.

    With ``record_paths``, every interaction point is copied to the host
    (slow; use only for a few hundred rays) for visualization.
    """
    n_rays = len(origins)

    ray = mi.Ray3f(p3(origins), v3(directions))

    sampler = mi.load_dict({
        "type": "independent",
        "sample_count": 1,
        "seed": int(seed),
    })
    sampler.seed(int(seed), n_rays)

    active = mi.Bool(np.ones(n_rays, dtype=np.bool_))
    detected = mi.Bool(np.zeros(n_rays, dtype=np.bool_))
    escaped = mi.Bool(np.zeros(n_rays, dtype=np.bool_))
    in_core_hit = mi.Bool(np.zeros(n_rays, dtype=np.bool_))
    exit_weight = mi.Float(np.zeros(n_rays, dtype=np.float32))
    bounces = mi.UInt32(np.zeros(n_rays, dtype=np.uint32))
    escape_o = mi.Point3f(0.0)
    if return_escapes:
        escape_o = dr.zeros(mi.Point3f, n_rays)
    paths = [np.asarray(origins, dtype=np.float32)] if record_paths else None
    dirs = [] if record_paths else None
    if det is not None:
        det_c = mi.Point3f(*det["center"])
        core_r = det["core_radius"]

    ctx = mi.BSDFContext()

    for depth in range(int(max_bounces)):
        if not bool(dr.any(active)):
            break

        si = scene.ray_intersect(
            ray,
            ray_flags=mi.RayFlags.All,
            coherent=(depth == 0),
            active=active,
        )

        valid = active & si.is_valid()
        escaped |= active & ~si.is_valid()
        if return_escapes:
            # Origin of the escaping ray = last interaction point.
            escape_o = dr.select(active & ~si.is_valid(), ray.o, escape_o)

        # Detector is the only emitter-tagged shape in the scene. Test the
        # shape itself rather than emitter.eval(): area emitters are
        # one-sided, so eval() would miss hits on the back face.
        hit_detector = valid & si.shape.is_emitter()

        detected |= hit_detector
        active_next = valid & ~hit_detector
        bounces += mi.UInt32(dr.select(active_next, 1, 0))

        if det is not None:
            if core_r is not None:
                rel = si.p - det_c
                rel = rel - dr.dot(rel, si.n) * si.n
                core_here = dr.norm(rel) < core_r
            else:
                core_here = mi.Bool(True)
            in_core_hit |= hit_detector & core_here
            if det["exit_fresnel"]:
                n_in = dr.select(core_here, det["n_core"], det["n_cladding"])
                cos_i = dr.abs(dr.dot(ray.d, si.n))
                F, _, _, _ = mi.fresnel(cos_i, det["n_out"] / n_in)
                t_exit = 1.0 - F
            else:
                t_exit = mi.Float(1.0)
            exit_weight = dr.select(hit_detector, t_exit, exit_weight)

        if record_paths:
            pt = dr.select(valid, si.p, ray.o + escape_len * ray.d)
            pt = np.array(pt, dtype=np.float32).T
            pt[~np.array(active, dtype=bool)] = np.nan
            paths.append(pt)
            dv = np.array(ray.d, dtype=np.float32).T   # direction arriving here
            dv[~np.array(active, dtype=bool)] = np.nan
            dirs.append(dv)

        if not bool(dr.any(active_next)):
            active = active_next
            break

        bsdf = si.bsdf(ray)
        bs, weight = bsdf.sample(
            ctx,
            si,
            sampler.next_1d(active_next),
            sampler.next_2d(active_next),
            active_next,
        )

        # If the sampled BSDF branch has zero weight, terminate that path.
        nonzero = dr.any(mi.unpolarized_spectrum(weight) != 0.0)
        active_next &= nonzero

        wo = si.to_world(bs.wo)
        offset = (1.0 + dr.max(dr.abs(si.p))) * offset_rel
        ray = mi.Ray3f(si.p + dr.mulsign(si.n, dr.dot(si.n, wo)) * offset, wo)
        active = active_next

        # Evaluate the full loop state every iteration. Evaluating only
        # `active` (via dr.any) leaves the rest lazy, so the graph grows and
        # earlier intersections get recomputed; with everything evaluated,
        # each iteration reuses one cached kernel.
        sampler.schedule_state()
        dr.eval(ray, active, detected, escaped,
                in_core_hit, exit_weight, bounces, escape_o)

    dr.eval(detected, escaped, active, in_core_hit, exit_weight, bounces)
    dr.sync_thread()

    det_np = np.array(detected, dtype=bool)
    core_np = np.array(in_core_hit, dtype=bool) & det_np
    clad_np = det_np & ~core_np
    w = np.array(exit_weight, dtype=np.float64)
    escaped_count = int(np.count_nonzero(np.array(escaped, dtype=bool)))
    still_active_count = int(np.count_nonzero(np.array(active, dtype=bool)))

    result = {
        "launched": n_rays,
        "detected": int(det_np.sum()),
        "escaped": escaped_count,
        "terminated_at_max_bounces": still_active_count,
        "transmission": det_np.sum() / n_rays,
    }
    if det is not None:
        result.update({
            "reach_end_core": core_np.sum() / n_rays,
            "reach_end_cladding": clad_np.sum() / n_rays,
            "exit_core": w[core_np].sum() / n_rays,
            "exit_cladding": w[clad_np].sum() / n_rays,
            "exit_total": w[det_np].sum() / n_rays,
            "mean_interactions_detected": float(np.array(bounces)[det_np].mean()) if det_np.any() else 0.0,
        })
    if return_escapes:
        result["escaped_mask"] = np.array(escaped, dtype=bool)
        result["escape_origin"] = np.array(escape_o, dtype=np.float64).T
        result["active_mask"] = np.array(active, dtype=bool)
    if record_paths:
        status = np.zeros(n_rays, dtype=np.int8)       # 0 = lost
        status[core_np] = 1                            # 1 = output, core
        status[clad_np] = 2                            # 2 = output, cladding
        status[np.array(active, dtype=bool)] = 3       # 3 = max bounces
        result["paths"] = np.stack(paths, axis=1)      # (n_rays, n_steps, 3)
        result["dirs"] = np.stack(dirs, axis=1)        # (n_rays, n_steps-1, 3)
        result["status"] = status
    return result


def ray_stderr(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rays", type=int, default=None, help="override simulation.rays")
    ap.add_argument("--seed", type=int, default=None, help="override simulation.seed")
    ap.add_argument("--theta-max-deg", type=float, default=None,
                    help="override source.theta_max_deg")
    ap.add_argument("--save-paths", type=int, default=0, metavar="N",
                    help="also trace N rays with full path recording and save "
                         "them to paths.npz next to the config (for viewing)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = cfg_path.parent

    sim = cfg["simulation"]
    if args.rays is not None:
        sim["rays"] = args.rays
    if args.seed is not None:
        sim["seed"] = args.seed
    if args.theta_max_deg is not None:
        cfg["source"]["theta_max_deg"] = args.theta_max_deg

    variant = sim.get("variant", "cuda_ad_rgb")
    mi.set_variant(variant)

    print(f"Mitsuba version : {mi.MI_VERSION}")
    print(f"variant         : {mi.variant()}")
    print(f"CUDA backend    : {mi.variant().startswith("cuda")}")

    n_rays = int(sim["rays"])
    seed = int(sim.get("seed", 1))

    origins, directions, theta_max = generate_launch_rays(
        cfg, cfg["ior"]["core"], n_rays, seed
    )
    print(f"launch theta max: {math.degrees(theta_max):.4f} deg")

    scene = load_scene(cfg, base_dir)
    det = detector_settings(cfg, scene)
    print(f"detector center : {det['center']}")

    import time
    t0 = time.perf_counter()
    result = trace(
        scene,
        origins,
        directions,
        max_bounces=int(sim["max_bounces"]),
        seed=seed + 1000,
        det=det,
        offset_rel=float(sim.get("ray_offset_rel", 2.0**-20)),
    )
    elapsed = time.perf_counter() - t0

    print("\n=== Result ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:26s}: {v:.6f}")
        else:
            print(f"{k:26s}: {v}")
    print(f"{'stderr(transmission)':26s}: {ray_stderr(result['transmission'], n_rays):.6f}")
    print(f"{'trace time [s]':26s}: {elapsed:.3f}")

    if args.save_paths > 0:
        o, d, _ = generate_launch_rays(cfg, cfg["ior"]["core"], args.save_paths, seed + 7)
        pr = trace(scene, o, d, int(sim["max_bounces"]), seed + 2000,
                   det=det, record_paths=True,
                   offset_rel=float(sim.get("ray_offset_rel", 2.0**-20)))
        out = base_dir / "paths.npz"
        np.savez_compressed(out, paths=pr["paths"], status=pr["status"])
        print(f"\npaths for {args.save_paths} rays saved to {out}")


if __name__ == "__main__":
    main()
