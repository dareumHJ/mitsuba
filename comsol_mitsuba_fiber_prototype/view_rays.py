#!/usr/bin/env python3
"""
Interactive 3D viewer for the fiber ray tracer (polyscope + Mitsuba GPU).

    python view_rays.py --config test_bent_fiber/config.yaml
    python view_rays.py --config test_bent_fiber/config.yaml --live-bend

Shows the fiber surfaces and a subset of traced ray paths, colored by fate:
  green  = reached the output face inside the core
  orange = reached the output face inside the cladding
  red    = lost (left the fiber / back out of the input end)

The side panel re-traces on the GPU when parameters change and reports the
transmission estimated from a larger batch of rays. With --live-bend the
geometry is the synthetic U-bent fiber from make_test_fiber.py, rebuilt
in memory whenever the bend radius slider changes.

--screenshot out.png renders one frame without opening a window.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import yaml

import mitsuba as mi

STATUS = {
    1: ("output (core)", (0.10, 0.75, 0.30)),
    2: ("output (cladding)", (1.00, 0.55, 0.05)),
    0: ("lost", (0.90, 0.15, 0.15)),
    3: ("max bounces", (0.5, 0.5, 0.5)),
}


class Viewer:
    def __init__(self, cfg_path, live_bend):
        import trace_mitsuba as tm
        self.tm = tm
        self.cfg_path = Path(cfg_path).resolve()
        self.base_dir = self.cfg_path.parent
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        mi.set_variant(self.cfg["simulation"].get("variant", "cuda_ad_rgb"))

        src = self.cfg["source"]
        if src.get("theta_max_deg") is not None:
            self.theta = float(src["theta_max_deg"])
        else:
            self.theta = math.degrees(math.asin(float(src["launch_na"]) / float(self.cfg["ior"]["core"])))
        self.n_show = 200
        self.n_stats = 100_000
        self.seed = int(self.cfg["simulation"].get("seed", 1))
        self.live_bend = live_bend
        self.bend_radius = 0.5
        self.tmp = tempfile.TemporaryDirectory(prefix="fiber_view_") if live_bend else None

        self.paths = None
        self.status = None
        self.stats = {}
        self.step_shown = 0
        self.playing = False
        self.play_speed = 4
        self.dirty_geometry = True
        self.dirty_rays = True

    # ---- geometry -------------------------------------------------------
    def mesh_paths(self):
        if not self.live_bend:
            return self.cfg, self.base_dir
        from make_test_fiber import build_fiber
        t0 = time.perf_counter()
        geo = build_fiber(50.0, self.bend_radius, 0.120, 0.125, 720, 360, 1.0)
        out = Path(self.tmp.name)
        names = {"core_boundary": "core_boundary", "cladding_outer": "cladding_air",
                 "detector": "detector"}
        for src, dst in names.items():
            v, f = geo[src]
            trimesh.Trimesh(v, f, process=False).export(out / f"{dst}.ply")
        cfg = copy.deepcopy(self.cfg)
        cfg["meshes"] = {k: f"{k}.ply" for k in ["core_boundary", "cladding_air", "detector"]}
        self.t_geo = time.perf_counter() - t0
        return cfg, out

    def load_geometry(self):
        cfg, base = self.mesh_paths()
        self.run_cfg, self.run_base = cfg, base
        self.scene = self.tm.load_scene(cfg, base)
        self.det = self.tm.detector_settings(cfg, self.scene)
        ps.remove_all_structures()
        for key, color, alpha in [("core_boundary", (0.3, 0.5, 1.0), 0.15),
                                  ("cladding_air", (0.7, 0.7, 0.8), 0.10),
                                  ("cladding_contact", (0.9, 0.3, 0.9), 0.35),
                                  ("silicone_body", (0.3, 0.8, 0.9), 0.12),
                                  ("detector", (1.0, 1.0, 0.2), 0.6)]:
            path = cfg["meshes"].get(key)
            if path is None or not (base / path).exists():
                continue
            m = trimesh.load(base / path, process=False)
            sm = ps.register_surface_mesh(key, np.asarray(m.vertices), np.asarray(m.faces),
                                          color=color, transparency=alpha)
            sm.set_back_face_policy("identical")
        self.dirty_rays = True

    # ---- tracing --------------------------------------------------------
    def retrace(self):
        cfg = copy.deepcopy(self.run_cfg)
        cfg["source"]["theta_max_deg"] = self.theta
        sim = cfg["simulation"]
        n_core = cfg["ior"]["core"]
        off = float(sim.get("ray_offset_rel", 2.0**-20))

        t0 = time.perf_counter()
        o, d, _ = self.tm.generate_launch_rays(cfg, n_core, self.n_stats, self.seed)
        r = self.tm.trace(self.scene, o, d, int(sim["max_bounces"]), self.seed + 1000,
                          det=self.det, offset_rel=off)
        self.t_stats = time.perf_counter() - t0
        self.stats = r

        t0 = time.perf_counter()
        o, d, _ = self.tm.generate_launch_rays(cfg, n_core, self.n_show, self.seed + 7)
        r = self.tm.trace(self.scene, o, d, int(sim["max_bounces"]), self.seed + 2000,
                          det=self.det, record_paths=True, offset_rel=off)
        self.t_paths = time.perf_counter() - t0
        self.paths, self.status = r["paths"], r["status"]
        self.step_shown = self.paths.shape[1]
        self.draw_paths()

    def draw_paths(self):
        if self.paths is None:
            return
        upto = max(2, min(self.step_shown, self.paths.shape[1]))
        for code, (name, color) in STATUS.items():
            sel = np.flatnonzero(self.status == code)
            nodes, edges = [], []
            base = 0
            for i in sel:
                p = self.paths[i, :upto]
                p = p[~np.isnan(p[:, 0])]
                if len(p) < 2:
                    continue
                nodes.append(p)
                k = np.arange(len(p) - 1) + base
                edges.append(np.stack([k, k + 1], axis=1))
                base += len(p)
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)
            if nodes:
                cn = ps.register_curve_network(name, np.concatenate(nodes),
                                               np.concatenate(edges), color=color)
                cn.set_radius(0.0006, relative=False)

    # ---- UI -------------------------------------------------------------
    def update(self):
        if self.dirty_geometry:
            self.load_geometry()
            self.dirty_geometry = False
        if self.dirty_rays:
            self.retrace()
            self.dirty_rays = False

    def ui(self):
        self.update()

        psim.TextUnformatted("Launch (point source, isotropic cone)")
        ch, self.theta = psim.SliderFloat("half-angle in core [deg]", self.theta, 1.0, 60.0)
        if ch:
            self.dirty_rays = True
        ch, self.n_show = psim.SliderInt("rays drawn", self.n_show, 10, 1000)
        if ch:
            self.dirty_rays = True
        ch, v = psim.InputInt("rays for statistics", self.n_stats, 10000, 100000)
        if ch and v > 0:
            self.n_stats, self.dirty_rays = v, True

        if self.live_bend:
            psim.Separator()
            psim.TextUnformatted("Geometry (synthetic U-bend, 50 mm, 240/250 um)")
            ch, self.bend_radius = psim.SliderFloat("bend radius [mm]", self.bend_radius, 0.2, 5.0)
            if psim.IsItemDeactivatedAfterEdit():
                self.dirty_geometry = True

        psim.Separator()
        psim.TextUnformatted("Animation")
        if psim.Button("Pause" if self.playing else "Play"):
            self.playing = not self.playing
            if self.playing and self.step_shown >= self.paths.shape[1]:
                self.step_shown = 2
        psim.SameLine()
        _, self.play_speed = psim.SliderInt("steps/frame", self.play_speed, 1, 20)
        ch, self.step_shown = psim.SliderInt("interactions shown", self.step_shown, 2,
                                             self.paths.shape[1])
        if ch:
            self.draw_paths()
        if self.playing:
            self.step_shown = min(self.step_shown + self.play_speed, self.paths.shape[1])
            self.draw_paths()
            if self.step_shown >= self.paths.shape[1]:
                self.playing = False

        psim.Separator()
        s = self.stats
        n = s["launched"]
        psim.TextUnformatted(f"Results ({n} rays, {self.t_stats:.2f} s on GPU)")
        psim.TextUnformatted(f"  reach output face : {s['transmission']:.4f}")
        psim.TextUnformatted(f"    in core         : {s['reach_end_core']:.4f}")
        psim.TextUnformatted(f"    in cladding     : {s['reach_end_cladding']:.4f}")
        psim.TextUnformatted(f"  exit into air     : {s['exit_total']:.4f}  (Pout/Pin)")
        psim.TextUnformatted(f"  lost              : {s['escaped'] / n:.4f}")
        psim.TextUnformatted(f"  paths traced in {self.t_paths:.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--live-bend", action="store_true",
                    help="use the synthetic fiber and expose a bend-radius slider")
    ap.add_argument("--screenshot", default=None, help="render one frame to this PNG and exit")
    args = ap.parse_args()

    if args.screenshot:
        ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_program_name("Fiber ray viewer")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_transparency_mode("pretty")

    v = Viewer(args.config, args.live_bend)
    ps.set_user_callback(v.ui)

    if args.screenshot:
        v.update()
        ps.look_at((0.5, 0.9, 25.6), (0.5, 0.0, 24.4))
        ps.screenshot(args.screenshot, transparent_bg=False)
        return
    ps.show()


if __name__ == "__main__":
    main()
