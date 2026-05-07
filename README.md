# Proximity Learning Architecture (PLA)

Sim-and-real pipeline for training a **Proximity Learning Architecture** — a
manipulation policy that fuses VLM-derived task grounding with low-resolution,
high-rate, body-distributed time-of-flight depth from a sensor skin on the
Franka FR3.

The headline experiment for the CoRL 2026 deadline is **PLA vs VLM-only ACT**:
the same VLM front-end and ACT policy backbone, with and without the proximity
skin stream, on a sweep of Franka manipulation tasks across iTHOR /
ProcTHOR-derived household scenes.

This top-level repo is a thin shell that pins:

- `assets/` — robot MJCF and shared scene/object assets.
- `submodules/molmospaces/` — the data-generation, simulation, and rendering
  stack (forked / extended).
- `submodules/MolmoBot/` — the policy / molmo VLM integration (training,
  inference).
- `synthetic_verify/`, `proximity_inspect/` — verification artifacts produced
  while validating the proximity sensor pipeline (kept in-tree for repro).

The remainder of this README documents the **proximity sensor stack** end to
end: model, simulator integration, verification protocol, and the substantive
fixes that landed during validation.

---

## 1. The franka_skin robot

`assets/robots/franka_skin/model.xml` is a Franka FR3 with:

- The standard FR3 collision and visual meshes (link0 .. link7).
- A Robotiq 2F-85 gripper (`robotiq_2f85_v4/2f85.xml`).
- A **sensor skin** wrapping link2, link3, link5, link6 — defined as
  decorative `class="skin"` geoms (group=2, contype=0, conaffinity=0,
  mass=0). The skin is purely visual; it does not collide and has no inertia.
- **29 SPAD-style proximity sensors** distributed across the four skinned
  links:

| Link | # sensors |
|------|-----------|
| link2 | 7 |
| link3 | 8 |
| link5 | 6 |
| link6 | 8 |
| **Total** | **29** |

Each sensor is realized in MJCF as a body whose pose places it just above the
collision mesh at the desired skin location, with two children:

- A `<site class="skin_sensor_site">` (red sphere, group=2) — visual marker
  for the sensor location.
- A `<camera mode="fixed" pos="0 0 0" quat="0 0 1 0" fovy="45.0"
  resolution="8 8"/>` — the 8x8 proximity sensor itself.

The `quat="0 0 1 0"` is a 180° rotation about the body's local Y axis, which
preserves the outward viewing direction (along the body's +Z) while orienting
the rendered image so "up" in the image corresponds to the conventional ToF
top edge. (See §5 for why this changed from `0 1 0 0`.)

Sensor body positions and orientations were imported from the original Isaac
Sim USD placements; the per-sensor `pos` / `quat` on each `<body>` is what
defines the geometry of the array.

---

## 2. SPAD proximity sensor model

A single proximity sensor models an array-style time-of-flight depth chip
(loosely a VL53L8CX class part):

| Property | Value |
|----------|-------|
| Resolution | 8 × 8 |
| Field of view | 45° (HFOV = VFOV, square pixel grid) |
| Range | 0.05 – 4.0 m |
| Sample rate | 60 Hz (sub-stepped within the policy step) |
| Output channel | depth only (no RGB at training time) |

Because all 29 sensors share these parameters and only differ in body pose,
they are generated programmatically in
`molmo_spaces/configs/camera_configs.py::_skin_sensor_camera_specs()` (29
`MjcfCameraConfig(..., record_depth=True, is_proximity_sensor=True)` entries).

`is_proximity_sensor=True` is the flag the env layer uses to:

- skip RGB rendering for that camera,
- record depth at `proximity_sensor_period_ms` (default 16.67 ms ≈ 60 Hz),
  sub-stepping multiple frames per policy step,
- route through the dedicated 8x8 depth renderer rather than the global RGB
  renderer.

---

## 3. How proximity depth is rendered

Implemented in
`submodules/molmospaces/molmo_spaces/env/env.py::record_proximity_depths`.

Two non-obvious design decisions, both load-bearing:

### 3a. Dedicated 8×8 renderer, not the global one

The global Mujoco renderer is sized at the RGB camera resolution (e.g.
624 × 352, non-square). If we asked it to render an `fovy=45°` proximity
camera, the *vertical* FOV is 45° but the *horizontal* FOV is determined by
the aspect ratio — we'd get HFOV ≈ 72°, ~1.6× wider than the SPAD spec, and
the depth pattern would be wrong.

The fix is a separate `mujoco.Renderer(model, height=8, width=8)` used only
for proximity sensors. With a square 8×8 viewport, HFOV = VFOV = 45° as
intended.

### 3b. Hide the skin during the proximity render

The skin is a decorative shell (group=2) sitting *outside* the collision
mesh; the proximity sensor bodies are positioned *inside* that shell (above
the collision mesh, embedded in the skin volume). With the skin visible,
each sensor sees its own skin at near-zero distance and reports the wrong
thing.

The fix:

```python
self._proximity_scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(self._proximity_scene_option)
self._proximity_scene_option.geomgroup[2] = 0  # hide skin
```

passed via `update_scene(..., scene_option=self._proximity_scene_option)`.
With group 2 hidden the renderer only considers the visual+collision links
of the robot and the surrounding scene — exactly what a real ToF chip
mounted on the skin would see.

### 3c. Sub-step buffer

Inside one policy control step, `record_proximity_depths(camera_names)` is
called multiple times (the substep loop), each call appending one 8×8
depth frame per active sensor to `self._proximity_depth_frames[camera]`.
`reset_proximity_depth_buffer()` clears the buffer at the start of each
policy step. `ProximityDepthBufferSensor` consumes the per-step list and
emits the trajectory time-series.

---

## 4. Verification protocol

We were not willing to start large-scale data collection without a
quantitative *and* visual sign-off that the 29 sensors see the world
correctly. The verification suite lives in
`submodules/molmospaces/scripts/datagen/`:

### 4a. Empty-room and flat-plane geometry tests

`verify_synthetic_scenes.py` and `verify_proximity_gt.py` place the
franka_skin robot in synthetic rooms (empty box, flat ground plane) and
compare the rendered 8×8 depth against ground-truth `mj_ray` casts (which
ignore back-face culling and rasterization quantization).

Per-sensor outputs:

- `synthetic_verify/empty_room/` — point clouds, error histograms, per-axis
  bias, GT-vs-rendered overlays.
- `synthetic_verify/flat_plane/` — quantization analysis as a function of
  sensor resolution; documents the **−44.6 mm floor bias at 8×8**
  (rasterization-driven, scales as 1/H, expected, not a bug).

### 4b. mj_ray independent classifier

A pure ray-cast diagnostic that bypasses the renderer and reports, per
sensor, whether the *body name* hit by the central ray is:

- the parent link's collision mesh (sensor pointing inward — placement bug),
- another robot link (sensor pointing across the arm — usually OK),
- the world / scene (sensor pointing outward — correct),
- nothing within range (free space).

Result: 24 / 29 sensors are correctly placed and oriented. 5 sensors
(link2_1, link2_2, link2_4, link3_0, link3_7) have a nominal viewing
direction that points toward their parent link's collision geom. With the
skin culled (§3b) and back-face culling on the collision mesh, the renderer
recovers an environment-derived depth for those rays anyway, so they are
usable in the production pipeline. The placement bug is documented and can
be fixed in a future skin revision without changing any code.

### 4c. Hi-res RGB inspection (one-off)

To go beyond histograms — an actual visual check of what every sensor sees
during a real episode — we ran a one-shot inspection mode that rendered an
extra 256×256 RGB frame from each proximity camera at every substep,
matching the depth render exactly (same camera pose, same skin-hidden scene
option). Outputs were 29 per-sensor MP4s plus a 29-tile grid MP4 over the
full episode in `assets/datagen/pick_planner_v1/inspect_prox_*/proximity_rgb/`.

The inspection MP4s confirmed:

- No sensor sees the gripper, its own link, or its own skin in normal
  motion.
- Sensor frames track the iTHOR house_1 kitchen geometry as the arm moves
  (cabinets, countertop, plants, windows visible from the appropriate
  link6 sensors).
- Image orientation was correct after the §5 quat fix.

The inspection-mode code was a verification-only path. **It has been
removed from `env.py` for production** so the env runs the canonical 8×8
depth path and nothing else (see §6).

---

## 5. Substantive fixes that landed during validation

Documented here so they don't get lost in git archaeology.

### 5a. Skin self-occlusion (the big one)

**Symptom:** in initial verification runs, ~33 % of rendered depths in an
empty room came back as `~0`, classified as self-hits.

**Root cause:** the skin shell (group=2) was visible to the proximity
renderer, and sensor bodies sit *inside* the skin volume. Every sensor saw
its own skin at distance ≈ 0.

**Fix:** hide group 2 via `MjvOption.geomgroup[2] = 0` on the proximity
renderer's scene option (see §3b). Empty-room self-hit rate dropped to
near zero for the 24 well-placed sensors.

### 5b. Failed `cx` shift, then reverted

**Symptom:** a residual −44.6 mm floor bias at 8×8 against a flat ground
plane.

**Hypothesis tried:** off-center principal point — moved `cx` from
`(W-1)/2 = 3.5` to `3.0` to "compensate" for what looked like a half-pixel
asymmetry.

**Result:** self-hit classification *increased* from 33.3 % → 36.8 %. The
shift introduced a lateral misalignment in the back-projected point cloud
without fixing the bias.

**Resolution:** reverted to `cx = (W-1)/2 = 3.5`. The −44.6 mm floor bias is
**rasterization quantization** that scales as `1/H` (verified by re-running
flat-plane tests at multiple resolutions). It is not a calibration bug; it
is a property of finite-resolution rendering and is small relative to the
SPAD's depth noise floor.

### 5c. Image orientation flip

**Symptom:** in the hi-res RGB inspection MP4s, scene content appeared
upside-down relative to the wrist / exo cameras.

**Root cause:** all 29 cameras had `quat="0 1 0 0"` — a 180° rotation about
local X. That preserves the viewing direction (still down +Z out of the
body) but flips the image vertically.

**Fix:** changed all 29 camera quats to `quat="0 0 1 0"` — a 180° rotation
about local Y. Same viewing direction, image right-side-up (and
horizontally mirrored vs the buggy state, which is the correct ToF
convention). Backup of the pre-fix model lives at
`assets/robots/franka_skin/model.xml.bak_before_orientation_fix`.

### 5d. Site-packages vs in-tree `molmo_spaces` resolution

**Symptom:** edits to `submodules/molmospaces/molmo_spaces/env/env.py`
weren't visible to the data-generation pipeline at runtime.

**Root cause:** the MolmoBot venv had both an old site-packages copy of
`molmo_spaces` (from a prior install) *and* the in-tree submodule. Python
was importing the site-packages copy.

**Fix at runtime:** invoke the pipeline with the in-tree path on
`PYTHONPATH` and the in-tree dir as cwd, e.g.

```bash
cd submodules/molmospaces \
  && PYTHONPATH=. \
       /home/jaydv/code/prox_learning/submodules/MolmoBot/MolmoBot/.venv/bin/python \
       scripts/datagen/run_pipeline.py [args...]
```

This is the canonical invocation pattern; the pipeline itself was not the
problem.

### 5e. Production cleanup

After visual sign-off, the inspection-mode patches were removed from
`env.py`. The fields `_proximity_inspect_resolution`,
`_proximity_inspect_renderer`, and `_proximity_rgb_frames`, the method
`enable_proximity_rgb_inspection`, and the conditional inspection branch
inside `record_proximity_depths` are all gone. `record_proximity_depths`
now does exactly one thing: render 8×8 depth from each proximity camera
with the skin hidden, append to the substep buffer, return.

The standalone helper scripts that were only useful during validation —
`scripts/datagen/inspect_proximity_rgb.py` and the failed
`scripts/datagen/fix_sensor_orientations.py` — have been deleted. The
analytical verifiers (`verify_synthetic_scenes.py`,
`verify_proximity_gt.py`, `visualize_proximity.py`,
`analyze_sample_episode.py`) remain.

---

## 6. Production data collection

To kick off a data-collection run with the franka_skin robot:

```bash
cd submodules/molmospaces
PYTHONPATH=. \
  /home/jaydv/code/prox_learning/submodules/MolmoBot/MolmoBot/.venv/bin/python \
  scripts/datagen/run_pipeline.py \
    --robot skin \
    --policy planner \
    --task_type pick \
    --scene_dataset ithor \
    --data_split train \
    --house_inds <list> \
    --samples_per_house <N> \
    --seed <S> \
    --run_name_prefix <name>
```

The output trajectory bundles will contain, per timestep:

- The standard exo / wrist RGB streams (as before).
- For each of the 29 proximity sensors, a list of 8×8 float32 depth frames
  recorded at 60 Hz across the policy step
  (`ProximityDepthBufferSensor`).
- The usual proprioception / action / language / object channels.

Trajectories are converted to a single training H5 with
`scripts/datagen/combine_trajs_into_h5.py`.

---

## 7. Repo layout (top level)

```
prox_learning/
├── assets/
│   ├── robots/franka_skin/        # franka_skin MJCF + skin meshes (the model)
│   ├── robots/franka_fr3/         # baseline FR3 (no skin)
│   ├── robots/...                 # other robots (yam, rby1, ...)
│   └── datagen/                   # rollout outputs + inspection runs
├── synthetic_verify/              # quantitative GT-vs-render verification
│   ├── empty_room/
│   ├── flat_plane/
│   └── summary.md
├── proximity_inspect/             # ad hoc proximity inspection artifacts
├── pointcloud.ipynb               # interactive proximity point-cloud notebook
├── submodules/
│   ├── molmospaces/               # data-gen + sim + rendering stack (extended)
│   └── MolmoBot/                  # VLM + policy training
├── pyproject.toml
└── README.md                      # this file
```

---

## 8. Status

- [x] Skin on FR3 (29 SPAD-style 8×8 proximity sensors, link2/3/5/6).
- [x] 8×8 depth render path with correct HFOV/VFOV (dedicated 8×8 renderer).
- [x] Skin self-occlusion fix (`geomgroup[2]=0`).
- [x] Camera image orientation correct (`quat="0 0 1 0"`).
- [x] Quantitative GT verification (mj_ray, empty room, flat plane).
- [x] Visual verification (per-sensor hi-res RGB inspection of a full
      house_1 pick episode).
- [x] Production cleanup (inspection-mode code removed from `env.py`).
- [ ] Large-scale data collection across iTHOR / ProcTHOR scenes.
- [ ] VLM-only ACT baseline.
- [ ] Train PLA policy on the same scene/task split.
- [ ] Real-robot transfer evaluation.
