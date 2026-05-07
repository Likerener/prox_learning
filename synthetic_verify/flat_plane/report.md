# Flat-plane verification

- Scene: only a horizontal floor at z=0 + the robot
- Poses: 36 arm configurations (12 yaws x 3 shoulder reaches)
- Total reconstructed points: 19,862

## Classification (5cm tolerance for 'on floor')
- on floor (|z| <= 5 cm): **7,556** (38.0%)
- robot self-hit (|z|>5cm, within 1.5m of base): 1,324
- other: 10,982

## Floor-plane accuracy (on-floor points only)
- Mean |z residual|: **23.32 mm**
- p99 |z|: **49.21 mm**
- Signed bias: **-18.83 mm** (nonzero indicates a systematic offset of the reconstruction along the floor normal)
- std(signed): 19.12 mm

## Outputs
- `scene_views.png` — rendered MuJoCo scene (3rd-person + top-down)
- `error_breakdown.png` — signed and absolute residual histograms; top-down scatter of reconstruction
- `recon.html` — interactive Plotly with GT floor surface (semi-transparent), robot skeleton, on-floor pts colored by link, self-hits in light gray
- `scene.ply` / `full.ply` — binary point clouds