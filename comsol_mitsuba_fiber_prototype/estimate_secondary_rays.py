#!/usr/bin/env python3
"""
Estimate how many secondary rays a COMSOL Ray Optics run of this geometry
needs, by emulating COMSOL's deterministic ray splitting on the Mitsuba scene.

At every partially reflecting interface the transmitted ray continues and the
reflected ray (power R * P) is released as a new secondary ray if its power
exceeds the threshold; otherwise that power is dropped. Total internal
reflection does not split. The output end face is a freeze wall (case A in
the comparison notes), the input end is open, as in trace_mitsuba.py.

    python estimate_secondary_rays.py --config test_bent_fiber/config.yaml

Thresholds are relative to the power of one primary ray (Pin / N).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
import mitsuba as mi
import drjit as dr


def run(tm, scene, det, cfg, n_primary, threshold, max_steps, offset_rel, seed):
    o, d, _ = tm.generate_launch_rays(cfg, cfg["ior"]["core"], n_primary, seed)
    p = np.ones(n_primary)
    ctx = mi.BSDFContext()
    core_r = det["core_radius"]
    c = np.asarray(det["center"])

    secondaries = 0
    peak_live = n_primary
    acc = {"core": 0.0, "clad": 0.0, "escaped": 0.0, "dropped": 0.0, "unfinished": 0.0}

    for _ in range(max_steps):
        if len(p) == 0:
            break
        peak_live = max(peak_live, len(p))
        ray = mi.Ray3f(tm.p3(o), tm.v3(d))
        si = scene.ray_intersect(ray)
        valid = np.array(si.is_valid(), dtype=bool)
        is_det = np.array(si.shape.is_emitter(), dtype=bool) & valid
        pos = np.array(si.p, dtype=np.float64).T
        nrm = np.array(si.n, dtype=np.float64).T

        acc["escaped"] += p[~valid].sum()
        if is_det.any():
            rel = pos[is_det] - c
            nn = nrm[is_det]
            rel -= (rel * nn).sum(1, keepdims=True) * nn
            in_core = np.linalg.norm(rel, axis=1) < core_r if core_r is not None else np.ones(len(rel), bool)
            acc["core"] += p[is_det][in_core].sum()
            acc["clad"] += p[is_det][~in_core].sum()

        hit = valid & ~is_det
        if not hit.any():
            p = p[:0]
            break

        # Evaluate both lobes of the smooth dielectric: sample1 = 0 always
        # picks reflection (pdf = R), sample1 -> 1 picks transmission
        # (pdf = 1 - R) unless the interface reflects totally.
        act = mi.Bool(hit)
        bsdf = si.bsdf()
        bs_r, _ = bsdf.sample(ctx, si, mi.Float(0.0), mi.Point2f(0.5, 0.5), act)
        bs_t, _ = bsdf.sample(ctx, si, mi.Float(1.0 - 1e-7), mi.Point2f(0.5, 0.5), act)
        wr = np.array(si.to_world(bs_r.wo), dtype=np.float64).T
        wt = np.array(si.to_world(bs_t.wo), dtype=np.float64).T
        R = np.array(bs_r.pdf, dtype=np.float64)
        tir = (R >= 1.0 - 1e-6) | ~np.array(
            mi.has_flag(bs_t.sampled_type, mi.BSDFFlags.Transmission), dtype=bool)
        R[tir] = 1.0

        def spawn(w):
            off = (1.0 + np.abs(pos).max(1, keepdims=True)) * offset_rel
            side = np.sign((nrm * w).sum(1, keepdims=True))
            return pos + side * nrm * off

        new_o, new_d, new_p = [], [], []
        # Continuation: reflected ray under TIR, otherwise the refracted ray.
        cont_w = np.where(tir[:, None], wr, wt)
        cont_p = np.where(tir, p, p * (1.0 - R))
        new_o.append(spawn(cont_w)[hit]); new_d.append(cont_w[hit]); new_p.append(cont_p[hit])
        # Secondary reflected rays.
        refl_p = p * R
        part = hit & ~tir & (R > 0)
        rel_ok = part & (refl_p > threshold)
        acc["dropped"] += refl_p[part & ~rel_ok].sum()
        secondaries += int(rel_ok.sum())
        new_o.append(spawn(wr)[rel_ok]); new_d.append(wr[rel_ok]); new_p.append(refl_p[rel_ok])

        o = np.concatenate(new_o).astype(np.float32)
        d = np.concatenate(new_d).astype(np.float32)
        p = np.concatenate(new_p)
    acc["unfinished"] = p.sum()
    return secondaries, peak_live, {k: v / n_primary for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="test_bent_fiber/config.yaml")
    ap.add_argument("--primary", type=int, default=2000)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args()

    import trace_mitsuba as tm
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    mi.set_variant(cfg["simulation"].get("variant", "cuda_ad_rgb"))
    scene = tm.load_scene(cfg, cfg_path.parent)
    det = tm.detector_settings(cfg, scene)
    off = float(cfg["simulation"].get("ray_offset_rel", 2.0**-20))

    print(f"{'threshold':>10} {'sec/primary':>12} {'peak live/primary':>18} "
          f"{'reach core':>11} {'reach clad':>11} {'reach total':>12} {'lost side':>10} {'dropped':>9}")
    for thr in args.thresholds:
        sec, peak, a = run(tm, scene, det, cfg, args.primary, thr, args.max_steps, off, 1)
        print(f"{thr:10.0e} {sec / args.primary:12.1f} {peak / args.primary:18.1f} "
              f"{a['core']:11.4f} {a['clad']:11.4f} {a['core'] + a['clad']:12.4f} "
              f"{a['escaped']:10.4f} {a['dropped']:9.5f}")


if __name__ == "__main__":
    main()
