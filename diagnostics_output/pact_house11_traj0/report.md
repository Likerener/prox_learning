# PACT trajectory diagnostics — `traj_0`

**Source:** `/home/jaydv/code/prox_learning/assets/datagen/house_7_test/548/PACT/20260604_222101/house_11/trajectories_batch_1_of_1.h5`  
**Task:** Pick up the brown pinecone keychain and place it in or on the bright insect design bowl  
**Steps:** 270 · **Proximity sensors:** 29 · **Final success:** True · terminated: True

## Collision
- **Environment-collision probability** (collision_metrics, excludes held object): **1.1%** (3/270 steps; collisions at steps [0, 1, 2])
- **Any-contact probability** (task_info.robot_contact, includes the grasped object): **31.5%**
- Nearest surface seen by the skin over the episode: **0.050 m** (at step 8)

## Proximity health
- nonzero-pixel fraction: 0.9963 (≈1.0 = sensors recording every step)
- per-sensor mean depth range: 0.22–3.57 m (clipped to 4 m for display)

## Phase durations (steps)
- unknown: 33
- gripper-open: 6
- pregrasp: 48
- grasp: 15
- gripper-close: 9
- lift: 34
- preplace: 15
- place: 27
- retreat: 13
- go_home: 70

## Figures
- `01_qpos_arm.png`
- `02_qvel_arm.png`
- `03_gripper.png`
- `04_action_vs_qpos.png`
- `05_tcp_trajectory.png`
- `06_tcp_distances.png`
- `07_phases.png`
- `08_proximity_mean_heatmap.png`
- `09_proximity_closest_heatmap.png`
- `10_proximity_montage_closest.png`
- `11_collision_metric.png`
- `12_collision_prob_by_phase.png`
- `13_reward_success.png`