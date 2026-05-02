## Dataset statistics — `skin_pick_fixed_v1`

- Trajectories: **10** across 2 HDF5 files
- Total timesteps: **715**
- Trajectory length range: 57–92
- Trajectory mean length: 71.5

### Storage
- HDF5 files: 9.9 MB across 2 files
- RGB MP4 files: 2.9 MB across 20 files (20 × 2 per episode: exo + wrist)
- Total on disk: 12.7 MB

### Per-trajectory table
| idx | path | timesteps |
|---|---|---|
| 0 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_0` | 92 |
| 1 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_1` | 67 |
| 2 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_2` | 75 |
| 3 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_3` | 72 |
| 4 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_4` | 70 |
| 5 | `FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5::traj_5` | 72 |
| 6 | `FrankaSkinPickConfig/20260420_225721/house_1/trajectories_batch_1_of_1.h5::traj_0` | 71 |
| 7 | `FrankaSkinPickConfig/20260420_225721/house_1/trajectories_batch_1_of_1.h5::traj_1` | 64 |
| 8 | `FrankaSkinPickConfig/20260420_225721/house_1/trajectories_batch_1_of_1.h5::traj_2` | 75 |
| 9 | `FrankaSkinPickConfig/20260420_225721/house_1/trajectories_batch_1_of_1.h5::traj_3` | 57 |

### Phase coverage
| phase | name | timesteps | % |
|---|---|---|---|
| 1 | approach | 8 | 1.1% |
| 2 | pre-grasp | 315 | 44.1% |
| 3 | grasp | 102 | 14.3% |
| 4 | lift | 99 | 13.8% |
| 5 | place | 191 | 26.7% |

### Arm joint positions (panda[:, 0:7])
| joint | min | max | mean | std |
|---|---|---|---|---|
| fr3_joint1 | -0.379 | +0.414 | -0.025 | 0.165 |
| fr3_joint2 | -0.903 | +0.564 | -0.172 | 0.432 |
| fr3_joint3 | -0.404 | +1.219 | +0.291 | 0.388 |
| fr3_joint4 | -2.941 | -1.065 | -2.165 | 0.489 |
| fr3_joint5 | -0.352 | +0.578 | +0.027 | 0.211 |
| fr3_joint6 | +1.360 | +3.129 | +2.037 | 0.428 |
| fr3_joint7 | -0.940 | +1.929 | +0.410 | 0.676 |

### Gripper (driver + 5 mimic joints saved as the same value)
| finger | min | max | mean | std |
|---|---|---|---|---|
| open-close (rad) | -0.0005 | 0.8240 | 0.1912 | 0.2573 |

### TCP pose (robot base frame) — obs/extra/tcp_pose
| component | min | max | mean | std |
|---|---|---|---|---|
| x (m) | +0.264 | +0.633 | +0.433 | 0.107 |
| y (m) | -0.194 | +0.425 | +0.134 | 0.189 |
| z (m) | +0.591 | +1.070 | +0.875 | 0.112 |
| qx | -0.097 | +0.111 | -0.004 | 0.038 |
| qy | +0.866 | +0.999 | +0.968 | 0.037 |
| qz | -0.499 | +0.329 | -0.087 | 0.220 |
| qw | -0.111 | +0.144 | +0.021 | 0.067 |

### Proximity depth (m) — 29 patches × 8×8 = 1856 per step × 715 steps = 1,327,040 values
- Overall min/max: 0.020 / 4.000
- Overall mean: 1.717
- % of values at zfar (no hit): 8.12%
- % of values < 0.30 m (near contact): 13.39%
- % of values < 0.10 m (very close): 7.84%

### Per-patch median depth (m)
| patch | median | fracNear | σ_t |
|---|---|---|---|
| link6_s0 | 0.381 | 0.371 | 0.241 |
| link6_s1 | 0.675 | 0.029 | 0.868 |
| link6_s2 | 1.205 | 0.013 | 1.235 |
| link6_s3 | 2.820 | 0.057 | 0.918 |
| link6_s4 | 3.157 | 0.000 | 0.976 |
| link6_s5 | 1.361 | 0.009 | 1.225 |
| link6_s6 | 0.587 | 0.038 | 0.572 |
| link6_s7 | 1.536 | 0.004 | 0.769 |
| link5_s0 | 0.187 | 0.529 | 0.598 |
| link5_s1 | 0.592 | 0.189 | 0.980 |
| link5_s2 | 0.488 | 0.385 | 0.619 |
| link5_s3 | 0.580 | 0.051 | 0.853 |
| link5_s4 | 0.640 | 0.088 | 1.058 |
| link5_s5 | 0.403 | 0.460 | 0.723 |
| link3_s0 | 1.834 | 0.117 | 0.936 |
| link3_s1 | 1.888 | 0.130 | 0.967 |
| link3_s2 | 1.541 | 0.015 | 0.873 |
| link3_s3 | 2.606 | 0.009 | 1.039 |
| link3_s4 | 1.831 | 0.004 | 0.907 |
| link3_s5 | 1.752 | 0.091 | 0.926 |
| link3_s6 | 2.288 | 0.000 | 0.985 |
| link3_s7 | 2.042 | 0.055 | 0.986 |
| link2_s0 | 1.882 | 0.032 | 1.114 |
| link2_s1 | 2.125 | 0.176 | 1.041 |
| link2_s2 | 1.566 | 0.247 | 1.094 |
| link2_s3 | 2.633 | 0.006 | 1.013 |
| link2_s4 | 1.444 | 0.410 | 0.843 |
| link2_s5 | 1.729 | 0.276 | 0.897 |
| link2_s6 | 2.254 | 0.091 | 1.032 |
