# Sample episode analysis

- Source: `/home/jaydv/code/prox_learning/assets/datagen/pick_and_place_skin_pilot_smoke_v1/FrankaSkinPickAndPlacePilotSmokeConfig/20260510_124831/house_2/trajectories_batch_1_of_1.h5`
- Timesteps: **278**
- Proximity sensors: **29**
- Per-sensor shape: `(278, 4, 8, 8)` (T, n_substeps, H, W)
- Episode success: **None**

## Verification

### Q1. Are the readings physically plausible?

- Total samples: 2,063,872, nonzero: 2,056,448 (99.6%)
- Depth range (nonzero): **[0.209, 2079.525] m**, median 1.896 m
- Fraction within SPAD spec range [0.05, 4.0]: **87.196%**
- Any NaN/Inf: NaN=False, Inf=False
- Note: depth values slightly above 4.0 m come from the renderer's zfar (10 m); consumers should clip to [0.05, 4.0] m for SPAD-faithful readings.
- **PASS**: False

### Q2. Do readings change over time (temporal structure)?

- Mean temporal variance per sensor: **0.50409 m²**
- Max temporal variance per sensor:  **3.69426 m²**
- Mean step-to-step Δ depth: **0.03339 m**
- **PASS**: True (variance > 1e-6, step-Δ > 0.1mm)

### Q3. Do readings correlate with task phase?

- Pearson r(Σ|Δ depth|, ‖q̇‖): **-0.054**  (per-step proximity activity vs arm joint-velocity magnitude)
- Phase-mean variance / total variance: **1.016**  (mean-level shift between phases)
- Max-within-phase std / min-within-phase std: **56059.4x**  (within-phase variability differs by this factor across phases)

| phase id | n_steps | mean prox depth (m) | within-phase std (m) | mean ‖q̇‖ (rad/s) |
|----------|---------|---------------------|----------------------|--------------------|
| 0 | 33 | 2.237 | 0.000 | 0.000 |
| 1 | 5 | 1.597 | 0.000 | 0.003 |
| 2 | 29 | 2.233 | 0.425 | 0.621 |
| 3 | 16 | 2.422 | 0.011 | 0.286 |
| 4 | 9 | 2.408 | 0.000 | 0.004 |
| 5 | 39 | 2.406 | 0.058 | 0.264 |
| 6 | 35 | 2.224 | 0.163 | 0.449 |
| 7 | 29 | 1.686 | 0.068 | 0.219 |
| 8 | 13 | 1.643 | 0.029 | 0.224 |
| 9 | 70 | 2.146 | 0.247 | 0.240 |

- **PASS**: True  (phase-mean shift > 5% of total variance OR within-phase std varies > 5x across phases)

### Q4. Is data saved in the right place with the right schema?

Expected schema:
- `obs/proximity/link{N}_sensor_{i}` (29 datasets, shape (T, 4, 8, 8) float32)
- `obs/sensor_param/<cam>/{intrinsic_cv, extrinsic_cv, cam2world_gl}`
- `obs/agent/qpos`, `obs/agent/qvel` as JSON-encoded uint8
- `actions/{ee_pose, ee_twist, joint_pos, joint_pos_rel, commanded_action}`
- Companion MP4s for `wrist_camera`, `exo_camera_1` (RGB + `_depth`)

- Found 29 proximity datasets ✓ (29 expected)
- Found 31 sensor_param entries ✓ (≥31 expected)
- RGB videos found: 2/2
- Depth videos found: 2/2
- **PASS**: True

## Point cloud reconstruction

- **1,793,141 points** emitted — per-pixel back-projection of every (sensor, substep, time, u, v) reading with depth in [0.05, 4.0]m.
- Theoretical maximum: 29 sensors x 278 steps x 4 substeps x 64 px = 2,063,872 pts
- World x range: [-3.52, 3.90] m
- World y range: [-3.02, 4.32] m
- World z range: [-2.17, 4.95] m
- Open `pointcloud.ply` in MeshLab/CloudCompare to inspect alongside the scene.

### Per-sensor diagnostics

| sensor | n valid pts | mean depth (m) | min depth | max depth | frac saturated (>=4.0m) |
|--------|-------------|----------------|-----------|-----------|--------------------------|
| link2_sensor_0 | 70,912/71,168 | 2.114 | 0.278 | 3.796 | 0.0% |
| link2_sensor_1 | 53,189/71,168 | 2.326 | 0.979 | 4.000 | 24.9% |
| link2_sensor_2 | 70,912/71,168 | 2.305 | 0.758 | 3.798 | 0.0% |
| link2_sensor_3 | 69,746/71,168 | 1.787 | 0.807 | 4.000 | 1.6% |
| link2_sensor_4 | 50,508/71,168 | 2.723 | 1.510 | 4.000 | 28.7% |
| link2_sensor_5 | 63,115/71,168 | 2.612 | 1.466 | 4.000 | 11.0% |
| link2_sensor_6 | 70,912/71,168 | 1.442 | 0.263 | 3.952 | 0.0% |
| link3_sensor_0 | 52,903/71,168 | 2.341 | 0.950 | 4.000 | 25.3% |
| link3_sensor_1 | 47,625/71,168 | 2.067 | 0.420 | 4.000 | 32.7% |
| link3_sensor_2 | 70,912/71,168 | 1.640 | 0.558 | 3.698 | 0.0% |
| link3_sensor_3 | 58,502/71,168 | 2.180 | 0.748 | 4.000 | 17.4% |
| link3_sensor_4 | 65,049/71,168 | 2.287 | 1.111 | 4.000 | 8.2% |
| link3_sensor_5 | 59,313/71,168 | 2.405 | 0.448 | 4.000 | 16.3% |
| link3_sensor_6 | 64,657/71,168 | 2.515 | 0.459 | 4.000 | 8.8% |
| link3_sensor_7 | 70,912/71,168 | 1.305 | 0.228 | 2.610 | 0.0% |
| link5_sensor_0 | 54,496/71,168 | 2.417 | 0.437 | 4.000 | 23.1% |
| link5_sensor_1 | 55,083/71,168 | 2.417 | 0.653 | 4.000 | 22.2% |
| link5_sensor_2 | 55,292/71,168 | 2.246 | 0.666 | 4.000 | 21.9% |
| link5_sensor_3 | 56,196/71,168 | 1.523 | 0.443 | 4.000 | 20.7% |
| link5_sensor_4 | 51,142/71,168 | 2.407 | 0.386 | 4.000 | 27.8% |
| link5_sensor_5 | 49,129/71,168 | 2.305 | 0.629 | 4.000 | 30.6% |
| link6_sensor_0 | 70,912/71,168 | 0.783 | 0.483 | 1.187 | 0.0% |
| link6_sensor_1 | 70,912/71,168 | 0.603 | 0.432 | 1.022 | 0.0% |
| link6_sensor_2 | 70,912/71,168 | 0.575 | 0.435 | 0.914 | 0.0% |
| link6_sensor_3 | 60,993/71,168 | 1.241 | 0.209 | 4.000 | 13.9% |
| link6_sensor_4 | 70,912/71,168 | 0.649 | 0.444 | 1.282 | 0.0% |
| link6_sensor_5 | 70,912/71,168 | 0.565 | 0.437 | 0.866 | 0.0% |
| link6_sensor_6 | 70,912/71,168 | 0.634 | 0.430 | 1.032 | 0.0% |
| link6_sensor_7 | 46,171/71,168 | 2.263 | 0.453 | 4.000 | 34.8% |
