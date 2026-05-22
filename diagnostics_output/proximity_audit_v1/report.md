# Proximity activation audit — partial medium dataset

- Frames analyzed: **72651** (280 trajectories, 66 houses)
- Activation threshold: < 0.5 m
- Near-contact threshold: < 5 cm

## Q1 — Per-link activation rate (any sensor on link < 0.50 m)

| Link | Activation rate |
|---|---|
| link2 | 0.997 |
| link3 | 0.805 |
| link5 | 0.431 |
| link6 | 0.627 |

### Threshold sweep

| Threshold | link2 | link3 | link5 | link6 |
|---|---|---|---|---|
| <2.00m | 1.000 | 1.000 | 0.992 | 1.000 |
| <1.00m | 1.000 | 1.000 | 0.789 | 0.926 |
| <0.50m | 0.997 | 0.805 | 0.431 | 0.627 |
| <0.20m | 0.219 | 0.067 | 0.055 | 0.181 |
| <0.10m | 0.046 | 0.014 | 0.013 | 0.060 |

## Q2 — Activation by phase

Approach (pregrasp + preplace) frames: 15166  
Retreat (retreat + go_home) frames: 22766

| Link | Approach | Retreat | Δ |
|---|---|---|---|
| link2 | 0.997 | 0.997 | -0.001 |
| link3 | 0.730 | 0.769 | -0.039 |
| link5 | 0.423 | 0.344 | +0.079 |
| link6 | 0.629 | 0.570 | +0.059 |

## Q3 — Per-house clutter correlation

Clutter proxy = -mean(per-frame min over all 29 sensors), aggregated per house (higher = more clutter nearby on average). Sample size = 66 houses.

| Link | Pearson r (clutter ↔ activation) |
|---|---|
| link2 | +0.248 |
| link3 | +0.610 |
| link5 | +0.271 |
| link6 | +0.457 |

## Q4 — Collision audit

Global `task_info.robot_contact` rate: **0.340** of frames

Per-link near-contact (any sensor < 5 cm), proxy for actual physical contact:

| Link | Near-contact rate |
|---|---|
| link2 | 0.0105 |
| link3 | 0.0038 |
| link5 | 0.0056 |
| link6 | 0.0133 |

## Self-sensing diagnostic

If a link's min sensor reading barely changes over a trajectory (std < 5 cm), the sensors are most likely pointed at the robot's own body at a fixed offset rather than at external clutter. Such links are EXCLUDED from the body-activation count in the path decision.

| Link | Median within-traj std (m) | Median frame-to-frame |Δ| (m) | Flag |
|---|---|---|---|
| link2 | 0.0227 | 0.00382 | **SELF-SENSING** |
| link3 | 0.1108 | 0.01415 | ok |
| link5 | 0.1670 | 0.02183 | ok |
| link6 | 0.1329 | 0.01574 | ok |

## Per-link reading distribution (meters)

| Link | Mean | Std | P05 | P50 | P95 |
|---|---|---|---|---|---|
| link2 | 0.239 | 0.065 | 0.103 | 0.257 | 0.306 |
| link3 | 0.356 | 0.133 | 0.178 | 0.332 | 0.548 |
| link5 | 0.699 | 0.449 | 0.192 | 0.554 | 1.674 |
| link6 | 0.481 | 0.323 | 0.092 | 0.417 | 1.123 |

## Recommendation

**Path B** — EE-centric: real-body links ['link3'] activate on <10% of frames (6.7%) at 0.2 m, while EE-area links ['link5', 'link6'] activate on 18.1%.  (self-sensing flagged: ['link2'])  Reframe paper around end-effector proximity sensing through wrist-camera occlusion.
