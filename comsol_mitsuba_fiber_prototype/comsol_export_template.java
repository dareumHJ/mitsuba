// Generic COMSOL Java API fragment for exporting a solved deformed mesh.
// Tags such as "dset1" and output paths must be replaced with those in your model.
//
// The official Results Mesh Export API supports:
//   data / dataset
//   filename
//   outersolnum  (parametric sweep index)
//   solnum
//   stlformat
//
// Create separate datasets with Selection attributes for:
//   1) lateral core-cladding interface
//   2) lateral outer-cladding surface
//   3) silicone contact-capable surface
//   4) output detector face
//
// Then point one export node at each selected dataset.

String base = "C:/tmp/fiber_case_";

for (int outer = 1; outer <= 10; ++outer) {
    // Example for one selected dataset "dsetCore".
    if (!model.result().export().has("expCore")) {
        model.result().export().create("expCore", "dsetCore", "Mesh");
    }

    model.result().export("expCore").set(
        "filename",
        base + String.format("%03d_core_boundary.stl", outer)
    );
    model.result().export("expCore").set("outersolnum", outer);
    model.result().export("expCore").set("stlformat", "binary");
    model.result().export("expCore").run();

    // Repeat the same pattern for:
    // dsetClad -> *_cladding_outer.stl
    // dsetSil  -> *_silicone_surface.stl
    // dsetDet  -> *_detector.stl
}
