# `pla/viz/` — figures, videos, dataset plots

## Purpose

Everything the paper needs as visual output. The four paper figures land
here:

  1. **System overview** with tensor shapes (`composite.system_overview`)
  2. **ToF heatmap sequence** — far / mid / near / pre-grasp
     (`heatmap.tof_sequence`)
  3. **Sensor importance heatmap on FR3 body** (`heatmap.sensor_importance_heatmap`)
  4. **Composite trajectory video** with proximity readings overlaid
     (`composite.trajectory_video`)

## Files

| file                  | what's in it                                                      |
|-----------------------|-------------------------------------------------------------------|
| `dataset_audit.py`    | **pre/post-collection visual audit suite (7 plots)**              |
| `heatmap.py`          | ToF heatmap sequences + sensor-importance heatmaps                |
| `composite.py`        | composite trajectory videos (RGB + ToF overlay)                   |
| `pointcloud.py`       | dataset-driven sensor pointcloud reconstruction (3D)              |
| `pointcloud_core.py`  | **pure-NumPy projection core (camera + legacy frames)**           |
| `pointcloud_tests.py` | **6-test rigorous pointcloud reconstruction suite**               |
| `sensor_overlay.py`   | **color-coded sensor-position overlays for the 29 ToF patches**   |
| `dataset_plots.py`    | depth histograms, coverage stats, per-sensor stats (legacy)       |
| `cvae_plots.py`       | (legacy) CVAE pretrain plots                                      |

## Visual audit suite (`dataset_audit.py`)

The single most important thing to look at before launching a long
collection. Generates 7 plots from any HDF5 dataset directory; each
catches a specific pre-launch failure mode you can't catch from
metrics alone.

| plot                              | catches                                                    |
|-----------------------------------|------------------------------------------------------------|
| `01_tof_montage.png`              | sensors all-saturated (skin orientation flipped); peak frame missing  |
| `02_per_sensor_dist.png`          | dead sensors (single-spike); stuck-at-saturation; uneven coverage    |
| `03_sensor_coverage.png`          | dead sensor min > 1500 mm; stuck σ < 0.5 mm                |
| `04_episode_traces.png`           | depth-min vs time / action-norm vs time; check trajectories make sense |
| `05_rgb_strip.png`                | blank cameras; frozen RGB; garbage RGB                     |
| `06_length_distribution.png`      | truncation; planner timeout; success-by-length pattern     |
| `07_action_distribution.png`      | action explosion (tail near ±1 cap); zero-action policy    |

```bash
# Run on any dataset directory:
python -m pla.viz.dataset_audit \
    --data-dir data/raw/near_contact_pilot \
    --out reports/checks/audit_pilot

# Then eyeball:
cat reports/checks/audit_pilot/INDEX.md
```

The preflight script (`scripts/preflight.sh`) calls this automatically
on the 50-episode pilot dataset and refuses to declare success until
you've inspected the plots.

## Outputs land in...

  * `reports/figures/` — paper-bound PDF figures
  * `reports/videos/` — composite trajectory MP4s for supplementary
  * `reports/checks/` — diagnostic plots from sanity checks

## Run

```bash
# 1. Paper figures (Day 12-13).
python -m pla.viz.heatmap --tof-h5 data/raw/near_contact/episode_000005.h5 \
    --out reports/figures/tof_sequence.pdf
python -m pla.viz.heatmap --importance-json reports/tables/sensor_importance.json \
    --out reports/figures/sensor_importance.pdf

# 2. Trajectory video (supplementary).
python -m pla.viz.composite --episode data/raw/near_contact/episode_000005.h5 \
    --out reports/videos/episode_5.mp4

# 3. Dataset coverage plots.
python -m pla.viz.dataset_plots --data-dir data/raw/near_contact \
    --out reports/figures/dataset_coverage.pdf
```

## Style requirements (paper)

  * minimum font size 8 pt at final embed scale
  * colorblind-safe palette (default: cmocean ``thermal`` or matplotlib ``viridis``)
  * self-contained captions — every axis labeled with units (mm for depth)
  * no embedded raster bitmaps for vector content (heatmaps may be raster
    inside a PDF wrapper)

## Sensor-position overlays (`sensor_overlay.py`)

The base `franka_skin_*.png` renders in `assets/reference_images/`
show the FR3 + GenTact mesh, but the 4 mm ToF sites in the MJCF are
**too small to see** at typical render resolution. `sensor_overlay`
re-renders the FR3 from configurable orbit cameras, projects every one
of the 29 sensor body positions into image pixels using the camera's
own `cam_xpos / cam_xmat`, and draws a labelled disk per sensor with
back-face culling.

Color legend:

| color  | link  | sensors | indices       |
|--------|-------|---------|---------------|
| red    | link2 | 7       | 0–6           |
| orange | link3 | 8       | 0–7           |
| green  | link5 | 6       | 0–5           |
| blue   | link6 | 8       | 0–7           |

```bash
MUJOCO_GL=egl python -m pla.viz.sensor_overlay
# → assets/reference_images/annotated/skin_overlay_*.png
# → assets/reference_images/annotated/sensor_layout_table.csv (29 rows)
# → assets/reference_images/annotated/skin_overlay_legend.png  (4-up grid)
```

## Pointcloud reconstruction tests (`pointcloud_tests.py`)

Six tests validate the ToF-depth → world-point pipeline against
mujoco-rendered ground truth, with diagnostic plots in
`reports/checks/pointcloud_tests/`:

| test | what it checks                                                                            | tolerance |
|------|--------------------------------------------------------------------------------------------|-----------|
| T1   | pinhole intrinsics: ray directions span the configured FOV, both frames                    | 1e-6      |
| T2   | synthetic flat wall unprojects to coplanar points at correct distance, side length         | 1e-9      |
| T3   | mujoco depth on a real wall: single-sensor reconstruction in world frame                   | 1 mm rms  |
| T4   | 29-sensor coverage: every reconstructed point lies in the look-direction half-space        | 0 ghosts  |
| T5   | static wall, two arm poses: same world plane reconstructed                                 | 5 mm rms  |
| T6   | legacy `pla.viz.pointcloud` y-flip vs corrected `+v` body-frame formula (regression alarm) | report    |

T6 surfaces an existing bug: the legacy `pla/viz/pointcloud.py`
formula `(u·half·d, -v·half·d, d)` × `R_body` has a residual y-flip
relative to the camera-frame ground truth (88 mm rms / 140 mm worst
case at 0.18 m wall distance). The corrected formula is
`(u·half·d, +v·half·d, d)` × `R_body`, or equivalently the
camera-frame path `pla.viz.pointcloud_core.reconstruct_world_pts(...,
frame="camera")` using `data.cam_xpos` / `data.cam_xmat`.

```bash
MUJOCO_GL=egl python -m pla.viz.pointcloud_tests
# exit 0 if all 6 pass; PNG plots + results.json land in
# reports/checks/pointcloud_tests/
```

## Sanity-check checklist (Day 13)

- [ ] All four headline figures render without `[?]` or missing fonts
- [ ] No emoji or non-ASCII characters appear in figure captions
- [ ] Compressed PDF output < 4 MB total
- [ ] Sensor importance heatmap colors match the colorbar legend
      (failure mode: the palette inverts when min > max)
