# Empty-room verification

- Scene: empty box, walls at +/-4.0 m, ceiling at 3.0 m
- Robot: franka_skin (FR3 + Robotiq + 29 SPAD-style proximity sensors)
- Poses: 108 arm configurations sweeping joint 1 (yaw, 12 angles), joint 2 (shoulder, 3 values), joint 4 (elbow, 3 values)
- Total reconstructed points: **185,658**

## Classification
| category | count | %  |
|---|---|---|
| on a wall plane (within 10 cm) | **86,852** | 46.8% |
| robot self-hits (in-room, off-wall) | 68,384 | 36.8% |
| out-of-room / other | 30,422 | 16.4% |

## Per-wall accuracy (on-wall points only)
| wall | n | mean&nbsp;\|err\| | p99 | signed bias | std |
|---|---|---|---|---|---|
| floor | 38,570 | 39.51 mm | 97.69 mm | -11.89 mm | 47.03 mm |
| wall_n | 14,035 | 42.70 mm | 98.89 mm | +1.05 mm | 52.01 mm |
| wall_s | 14,035 | 42.70 mm | 98.89 mm | +1.05 mm | 52.01 mm |
| wall_e | 10,106 | 39.90 mm | 98.66 mm | +2.04 mm | 49.36 mm |
| wall_w | 10,106 | 39.90 mm | 98.66 mm | +2.04 mm | 49.36 mm |

- **Mean |err|** = average distance of an on-wall point to the wall plane (small = good).
- **Signed bias** = average of (point - plane) along the plane normal. Nonzero indicates a systematic offset of reconstruction relative to the geometry (e.g. depth values systematically too long/short).
- **std(signed)** = noise level (sensor + sub-pixel quantization + back-projection error).

## Per-sensor coverage
See `sensor_coverage.png`. Each row = one of the 29 SPAD cameras; each column = where its returns landed (which wall, robot self, out-of-range).
Sensors that consistently hit the robot itself are persistent self-hit candidates and should be masked when training a downstream proximity model.

## Per-pose breakdown
`pose_grid.png` shows 4 representative poses. For each: (left) the rendered MuJoCo scene from a third-person camera so you can see what the robot is looking at, (middle) a top-down 2D scatter of the reconstructed cloud (robot at the cross), (right) histogram of |dist to nearest wall|.

## Outputs
- `scene_views.png` — third-person + top-down rendered MuJoCo scene
- `pose_grid.png` — 4 representative poses with rendered RGB + cloud + residual histogram
- `error_breakdown.png` — error histograms (signed + abs), per-wall bias/std/count bars
- `sensor_coverage.png` — per-sensor coverage heatmap
- `recon.html` — interactive Plotly with GT wall surfaces (semi-transparent), robot skeleton (black line), reconstructed cloud colored by link, robot self-hits in light gray
- `scene.ply` / `full.ply` — binary point clouds (load in MeshLab/CloudCompare)