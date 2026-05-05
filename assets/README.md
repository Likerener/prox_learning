# `assets/` — robot model + reference renders

```
assets/
├── urdf/               URDF (FR3 + skin) — source of truth for sensor poses
├── mjcf/               MuJoCo XML — built from URDF by `pla.sim.build_mjcf`
└── reference_images/   PNG dumps used for sanity comparisons in viz/
```

## URDF -> MJCF flow

1. `urdf/fr3_full_skin_fixed.urdf` — the corrected URDF after Blender
   sensor placement + post-processing (sensor +Z aligned outward).
2. `python -m pla.sim.build_mjcf --urdf assets/urdf/fr3_full_skin_fixed.urdf
   --out assets/mjcf/fr3_skin_fixed.xml` — build MJCF.
3. `python scripts/verify_skin.py --mjcf assets/mjcf/fr3_skin_fixed.xml`
   — confirm <5 self-hitting sensors.

## What's *expected* to be here vs not

* `urdf/` — checked into git; small.
* `mjcf/` — checked into git; built artifact but cheap to track.
* `reference_images/` — checked into git; tiny PNGs used for asserting
  visual regressions in the viz pipeline.
* **NOT** here: STL / mesh files. They live in the upstream
  `gentact_ros_tools` package; `build_mjcf` resolves them at build time.

## `reference_images/annotated/`

The originals (`franka_skin_*.png`, `franka_skin_in_kitchen*.png`) show
the FR3 mesh but the 4 mm sensor sites are too small to see at the
render resolution. The `annotated/` subfolder contains color-coded
overlays produced by `pla.viz.sensor_overlay` that mark every one of
the 29 VL53L5CX patches with a labeled disk:

  * red = link2 (7 sensors), orange = link3 (8), green = link5 (6),
    blue = link6 (8) — total 29.
  * label inside each disk = MJCF index within that link.
  * back-of-link sensors are culled per-view (mujoco depth doesn't draw
    them through the mesh) so each viewpoint only shows what's
    physically visible.

Files:

| file                                    | what                                               |
|-----------------------------------------|----------------------------------------------------|
| `skin_overlay_az<DEG>_el<DEG>.png`      | one orbit camera, sensors over the FR3 mesh        |
| `skin_overlay_legend.png`               | 4-up grid (-180 / -90 / 0 / +90) with legend       |
| `sensor_layout_table.csv`               | mjcf_index, name, link, world XYZ at home pose     |
| `_fr3_skin_patched.xml`                 | the patched MJCF (mesh paths repointed to caches)  |

Regenerate:

```bash
MUJOCO_GL=egl python -m pla.viz.sensor_overlay
# custom orbit + framing:
MUJOCO_GL=egl python -m pla.viz.sensor_overlay \
    --azimuths -180,-135,-90,-45,0,45,90,135 \
    --elevation -25 --distance 1.4 --width 800 --height 600
```
