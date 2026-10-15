#!/usr/bin/env python3
"""
Independent reference for the straight-fiber test case (no Mitsuba).

A point source on the axis of a straight step-index fiber launches only
meridional rays, which stay in a plane through the axis. The problem is then
exactly 2D: a slab |x| < a (core) inside |x| < b (cladding) surrounded by air.
This script runs the same stochastic, unpolarized Fresnel branching as the
Mitsuba tracer, but analytically, so the two can be compared.
"""

import argparse
import math
import numpy as np


def fresnel_unpolarized(cos_i, n1, n2):
    """Reflectance going from n1 into n2; returns (R, cos_t)."""
    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    if sin_t2 >= 1.0:
        return 1.0, 0.0
    cos_t = math.sqrt(1.0 - sin_t2)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return 0.5 * (rs * rs + rp * rp), cos_t


def run(n_rays, theta_max_deg, length, a, b, n_core, n_clad, n_air, seed,
        theta_min_deg=0.0):
    rng = np.random.default_rng(seed)
    cos_max = math.cos(math.radians(theta_max_deg))
    cos_min = math.cos(math.radians(theta_min_deg))
    reach = {"core": 0, "clad": 0}
    exit_w = {"core": 0.0, "clad": 0.0}
    lost = 0
    for _ in range(n_rays):
        # Uniform over the 3D cone solid angle; a meridional ray only needs
        # its polar angle theta (azimuth just picks the plane).
        ct = cos_min - rng.random() * (cos_min - cos_max)
        st = math.sqrt(1.0 - ct * ct)
        x, z = 0.0, 0.0
        dx, dz = st, ct            # dx sign is irrelevant by symmetry
        n_here = n_core
        while True:
            if dz <= 0:            # travelling back toward the input end
                lost += 1
                break
            # Next interface along x for the current region.
            if n_here == n_core:
                xb = a if dx > 0 else -a
            else:
                # In the cladding: inner (|x|=a) or outer (|x|=b) wall.
                if (x > 0) == (dx > 0):
                    xb = math.copysign(b, x)
                else:
                    xb = math.copysign(a, x)
            t = (xb - x) / dx if dx != 0 else math.inf
            if z + t * dz >= length:
                region = "core" if n_here == n_core else "clad"
                reach[region] += 1
                R, _ = fresnel_unpolarized(dz, n_here, n_air)
                exit_w[region] += 1.0 - R
                break
            x, z = xb, z + t * dz
            cos_i = abs(dx)
            if abs(xb) == b:
                n_other = n_air
            else:
                n_other = n_clad if n_here == n_core else n_core
            R, cos_t = fresnel_unpolarized(cos_i, n_here, n_other)
            if rng.random() < R:
                dx = -dx
            else:
                if n_other == n_air:
                    lost += 1
                    break
                # Snell: tangential component scales by n_here / n_other.
                dz = dz * n_here / n_other
                dx = math.copysign(cos_t, dx)
                n_here = n_other
    return reach, exit_w, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="test_straight_fiber/config.yaml",
                    help="indices (ior.core/cladding/air) are read from here")
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--theta-max-deg", type=float, default=30.0)
    ap.add_argument("--theta-min-deg", type=float, default=0.0)
    ap.add_argument("--length", type=float, default=50.0)
    ap.add_argument("--core-radius", type=float, default=0.120)
    ap.add_argument("--clad-radius", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        n = yaml.safe_load(f)["ior"]
    reach, exit_w, lost = run(args.rays, args.theta_max_deg, args.length,
                              args.core_radius, args.clad_radius,
                              n["core"], n["cladding"], n["air"], args.seed,
                              args.theta_min_deg)
    n = args.rays
    print(f"reach_end_core     : {reach['core'] / n:.6f}")
    print(f"reach_end_cladding : {reach['clad'] / n:.6f}")
    print(f"exit_core          : {exit_w['core'] / n:.6f}")
    print(f"exit_cladding      : {exit_w['clad'] / n:.6f}")
    print(f"exit_total         : {(exit_w['core'] + exit_w['clad']) / n:.6f}")
    print(f"lost               : {lost / n:.6f}")


if __name__ == "__main__":
    main()
