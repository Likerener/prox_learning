# Proximity activation audit — partial medium dataset

- Frames analyzed: **89029** (343 trajectories, 82 houses)
- Activation threshold: < 0.5 m
- Near-contact threshold: < 5 cm

## Q1 — Per-link activation rate (any sensor on link < 0.50 m)

| Link | Activation rate |
|---|---|
| link2 | 0.997 |
| link3 | 0.792 |
| link5 | 0.432 |
| link6 | 0.629 |

### Threshold sweep

| Threshold | link2 | link3 | link5 | link6 |
|---|---|---|---|---|
| <2.00m | 1.000 | 1.000 | 0.987 | 1.000 |
| <1.00m | 1.000 | 0.999 | 0.788 | 0.928 |
| <0.50m | 0.997 | 0.792 | 0.432 | 0.629 |
| <0.20m | 0.218 | 0.067 | 0.050 | 0.171 |
| <0.10m | 0.044 | 0.014 | 0.011 | 0.057 |

## Q2 — Activation by phase

Approach (pregrasp + preplace) frames: 18587  
Retreat (retreat + go_home) frames: 27801

| Link | Approach | Retreat | Δ |
|---|---|---|---|
| link2 | 0.997 | 0.997 | -0.001 |
| link3 | 0.715 | 0.759 | -0.043 |
| link5 | 0.423 | 0.347 | +0.076 |
| link6 | 0.630 | 0.568 | +0.062 |

## Q3 — Per-house clutter correlation

Clutter proxy = -mean(per-frame min over all 29 sensors), aggregated per house (higher = more clutter nearby on average). Sample size = 82 houses.

| Link | Pearson r (clutter ↔ activation) |
|---|---|
| link2 | -0.060 |
| link3 | +0.381 |
| link5 | +0.263 |
| link6 | +0.424 |

## Q4 — Collision audit

Global `task_info.robot_contact` rate: **0.341** of frames

Per-link near-contact (any sensor < 5 cm), proxy for actual physical contact:

| Link | Near-contact rate |
|---|---|
| link2 | 0.0089 |
| link3 | 0.0035 |
| link5 | 0.0048 |
| link6 | 0.0133 |

## Self-sensing diagnostic

If a link's min sensor reading barely changes over a trajectory (std < 5 cm), the sensors are most likely pointed at the robot's own body at a fixed offset rather than at external clutter. Such links are EXCLUDED from the body-activation count in the path decision.

| Link | Median within-traj std (m) | Median frame-to-frame |Δ| (m) | Flag |
|---|---|---|---|
| link2 | 0.0227 | 0.00380 | **SELF-SENSING** |
| link3 | 0.1106 | 0.01432 | ok |
| link5 | 0.1651 | 0.02266 | ok |
| link6 | 0.1337 | 0.01583 | ok |

## Per-link reading distribution (meters)

| Link | Mean | Std | P05 | P50 | P95 |
|---|---|---|---|---|---|
| link2 | 0.240 | 0.066 | 0.104 | 0.258 | 0.306 |
| link3 | 0.359 | 0.137 | 0.177 | 0.338 | 0.549 |
| link5 | 0.706 | 0.461 | 0.199 | 0.553 | 1.706 |
| link6 | 0.481 | 0.317 | 0.094 | 0.420 | 1.103 |

## Recommendation

**Path B** — EE-centric: real-body links ['link3'] activate on <10% of frames (6.7%) at 0.2 m, while EE-area links ['link5', 'link6'] activate on 17.1%.  (self-sensing flagged: ['link2'])  Reframe paper around end-effector proximity sensing through wrist-camera occlusion.
