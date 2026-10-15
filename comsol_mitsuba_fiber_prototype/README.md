# COMSOL FEM → Mitsuba 3 GPU ray-tracing prototype

This starter project is for a workflow like:

    COMSOL Solid Mechanics
      hyperelastic silicone + contact + fiber deformation
                ↓
      deformed surface meshes (.stl)
                ↓
      prepare_meshes.py
                ↓
       Mitsuba-compatible .ply
                ↓
       trace_mitsuba.py
                ↓
        RTX / OptiX ray tracing
                ↓
              Pout/Pin

The intent is **not** to replace COMSOL mechanics. COMSOL remains the
high-confidence FEM/contact solver; only the expensive ray-optics stage is
moved to a GPU ray tracer.

## 0. Important model assumptions

This MVP assumes a step-index fiber with three optical interfaces:

1. `core_boundary`: core ↔ cladding
2. `cladding_air`: outer cladding ↔ air
3. `cladding_contact`: outer cladding ↔ silicone at the contact patch

The output end face is exported separately as `detector`.

The input/output end caps must **not** be included in the lateral
core/cladding or cladding/external meshes, otherwise rays will hit a dielectric
surface before the detector/source.

The ray tracer uses stochastic Fresnel branching through Mitsuba's smooth
dielectric BSDF. Therefore `Pout/Pin` is estimated from the fraction of launched
rays that reach the detector. Validate this against several COMSOL Ray Optics
cases before using it for a large parameter sweep.

## 1. COMSOL: what to export

Solve only the mechanics/contact problem first.

Recommended physics:
- Solid Mechanics
- Hyperelastic material for silicone
- Geometric nonlinearity
- Contact between silicone and the optical-fiber outer surface

For one solved load case, create separate Result/Solution datasets (or copies)
scoped to these boundary selections:

- **core_boundary.stl**
  - only the lateral core ↔ cladding interface
  - exclude input and output end faces

- **cladding_outer.stl**
  - only the lateral outer cladding surface
  - exclude input/output end faces

- **silicone_surface.stl**
  - silicone surface that can contact the fiber
  - used only to classify which outer-cladding triangles are in contact

- **detector.stl**
  - output end face
  - preferably the fiber output aperture only

In COMSOL:
`Results -> Export -> Mesh`

Choose the Solution dataset corresponding to the deformed configuration and
export STL.

For a Parametric Sweep, COMSOL's Mesh Export API supports `outersolnum`, so
each load case can be exported automatically later.

### Normal orientation

Mitsuba's dielectric convention in this starter expects:

- `core_boundary`: normals point **out of the core and into cladding**
- `cladding_*`: normals point **out of cladding toward the external medium**

If transmission looks nonsensical, the first thing to test is flipping normals.
The YAML config exposes `flip_normals` flags.

## 2. Install Python dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Mitsuba's pip distribution provides CUDA variants. On an NVIDIA machine,
`cuda_ad_rgb` uses the CUDA backend and Mitsuba can use OptiX for ray
intersection. A sufficiently recent NVIDIA driver is required.

Check:

```bash
python -c "import mitsuba as mi; print(mi.MI_VERSION); print(mi.variants())"
```

You should see `cuda_ad_rgb`.

## 3. Convert/split the COMSOL meshes

Example, if COMSOL geometry coordinates are already in the units you want to
use:

```bash
python prepare_meshes.py \
  --core core_boundary.stl \
  --cladding cladding_outer.stl \
  --silicone silicone_surface.stl \
  --detector detector.stl \
  --contact-tol 0.01 \
  --out prepared
```

`--contact-tol` is in the **same coordinate unit as the STL meshes**.
For example, if your geometry coordinates are in mm, `0.01` means 0.01 mm.

The script creates:

- `prepared/core_boundary.ply`
- `prepared/cladding_air.ply`
- `prepared/cladding_contact.ply`
- `prepared/detector.ply`

### Contact-patch classification

For each outer-cladding triangle, the script finds its center and estimates the
distance to the deformed silicone surface.

    distance < contact_tol  → silicone contact
    otherwise               → air

This is deliberately simple. For final quantitative results, compare this
classification against COMSOL's actual contact status/contact-pressure field.
The threshold should be included in a convergence/sensitivity test.

## 4. Configure the optical model

Copy:

```bash
copy config.example.yaml config.yaml
# Linux: cp config.example.yaml config.yaml
```

Edit:

```yaml
ior:
  core: 1.49
  cladding: 1.40
  air: 1.000277
  silicone: 1.41
```

and the launch geometry:

```yaml
source:
  center: [0.0, 0.0, 0.05]
  axis: [0.0, 0.0, 1.0]
  radius: 0.45
  launch_na: 0.5
```

All spatial quantities must use the same coordinates/units as the PLY mesh.

The source center must be slightly **inside** the core at the input end, not
exactly on the mesh surface.

For a step-index fiber, when `launch_na` is specified as the usual NA in air,
the starter converts it to an internal axial half-angle using

    theta_internal = asin(NA / n_core)

You can instead specify `theta_max_deg` directly.

## 5. Run GPU ray tracing

```bash
python trace_mitsuba.py --config config.yaml
```

Output:

```text
launched rays : ...
detected rays : ...
Pout/Pin      : ...
```

The first CUDA/OptiX run can include JIT compilation overhead. Compare timing
after a warm-up run.

## 6. Validation plan

Do **not** start with a huge sweep. First compare COMSOL and Mitsuba for a few
fixed deformations, e.g.:

- no silicone contact
- small contact patch
- large contact patch

and perhaps 2–3 bend radii.

For each case, run a ray-count convergence study:

    1e4 -> 3e4 -> 1e5 -> 3e5 -> 1e6

Compare:

    T = Pout / Pin

and the relative discrepancy between Mitsuba and COMSOL.

A practical first milestone is not "perfect agreement"; it is:
- same trend versus force/contact
- stable ray-count convergence
- small enough error for your intended sensor-design sweep

## 7. Why the contact patch matters

The mechanics-to-optics interface is not the complete silicone stress field.
For CMFTIR-like behavior, the most important exported information is usually:

- deformed fiber geometry / local curvature
- which part of the outer cladding is in silicone contact
- refractive index assigned to that contact region

That is why the outer-cladding surface is split into `air` and `contact`
meshes before ray tracing.

## 7a. Synthetic test fiber and viewer (no COMSOL needed)

```bash
# U-bent 240/250 um fiber, 50 mm long, 0.5 mm bend radius (units: mm)
python make_test_fiber.py --out test_bent_fiber/stl
python prepare_meshes.py --core test_bent_fiber/stl/core_boundary.stl \
  --cladding test_bent_fiber/stl/cladding_outer.stl \
  --detector test_bent_fiber/stl/detector.stl --out test_bent_fiber/prepared
python trace_mitsuba.py --config test_bent_fiber/config.yaml

# Straight-fiber validation against an independent 2D calculation
python check_straight_2d.py

# Interactive 3D viewer (re-traces on the GPU when sliders change)
python view_rays.py --config test_bent_fiber/config.yaml --live-bend
```

`--silicone`/`--contact-tol` are optional in `prepare_meshes.py`; without
them the whole outer cladding is treated as cladding/air.

### Numerical pitfalls found while validating

- **Ray offset.** Mitsuba's `spawn_ray()` offsets new rays by about
  `(1 + max|p|) * 9e-5`, i.e. ~4.5 um at 50 mm from the origin. That is as
  thick as the cladding, so rays skipped it far from the origin. The tracer
  now uses its own offset (`simulation.ray_offset_rel`, default 2**-20).
- **Vertex count.** Meshes with more than 2**24 vertices lost triangles in
  Mitsuba 3.9.1 / OptiX (rays passed through). STL stores 3 vertices per
  triangle, so `prepare_meshes.py` now welds vertices and warns above 2**24.

## 8. Next upgrades after the MVP

1. Replace geometric distance-based contact tagging with COMSOL contact status
   or contact pressure exported at surface elements.
2. Reuse one mesh topology and update only vertex positions for every load
   case. Mitsuba supports updating mesh position tensors and refreshes dependent
   acceleration data.
3. Batch all force/load cases.
4. Record loss location and bounce statistics.
5. Add wavelength-dependent refractive indices.
6. Add polarization if experiments show it matters.
7. If Mitsuba's stochastic dielectric branching converges too slowly, replace
   it with a dedicated OptiX deterministic/split-ray implementation.

## Known caveat

The scripts are a research starter, not a certified optical solver. In
particular, the `trace_mitsuba.py` loop uses Mitsuba's renderer-oriented
dielectric sampling to estimate energy transmission statistically. It should
be treated as a fast surrogate only after validation against your existing
COMSOL Ray Optics model.