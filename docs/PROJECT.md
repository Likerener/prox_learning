# PLA: Peripersonal Language-Action Policies via Whole-Body Time-of-Flight Proximity Sensing

**Researcher:** Jay Vakil
**Advisor:** Alessandro Roncone
**Lab:** HIRO Lab, University of Colorado Boulder
**Target Venue:** Conference on Robot Learning (CoRL) 2026
**Submission Deadline:** May 28, 2026
**Start Date:** May 2, 2026
**Duration:** 26 days
**Code:** [github.com/Jdvakil/prox_learning](https://github.com/Jdvakil/prox_learning)

-----

## 1. Project Overview

PLA (Peripersonal Language-Action) is a learned robot manipulation policy that
integrates whole-body distributed time-of-flight proximity sensing as a
first-class input alongside vision and language. The core hypothesis is that
vision-language-action (VLA) policies have a fundamental geometric blind spot
— the peripersonal zone around the robot arm body — and that filling this
blind spot with pre-contact proximity sensing measurably improves manipulation
performance in cluttered environments.

This is a do-or-die paper submission. The work must be submitted by May 28, 2026.

-----

## 2. The Problem

### 2.1 The Peripersonal Blind Spot

Every state-of-the-art VLA policy — MolmoBot, OpenVLA, ACT, π₀ — conditions
exclusively on RGB observations from head-mounted and wrist-mounted cameras.
This creates a fundamental and underappreciated limitation: the robot cannot
see its own body.

During a reach into a cluttered scene, the arm occludes itself from the head
camera. The wrist camera faces forward — it cannot see what the arm links are
passing near. The 10–50 cm zone immediately surrounding the arm body —
precisely where collision risk is highest, near-contact events occur, and
grasp precision is most critical — provides zero visual signal until physical
contact occurs.

This is not a model capacity problem. Adding more parameters, more cameras, or
better visual backbones does not resolve a geometric occlusion. The missing
modality is **body-relative proximity** — depth information in the space
around the arm, measured relative to the arm itself rather than relative to a
camera.

### 2.2 Biological Motivation

This absence has a well-studied biological analog. Rizzolatti et al. (1997)
identified premotor and parietal neurons in primates that respond selectively
to stimuli approaching the body surface — not touching it, but within a
body-centric safety margin called **peripersonal space**. This representation
enables anticipatory motor adjustment before contact occurs.

Roncone et al. (2016) — our lab's foundational work — demonstrated that this
biological model can be computationally instantiated on a humanoid robot using
distributed skin sensing. The robot learns visuo-tactile receptive fields
around its body surface, enabling proximity-aware reactive control.

Despite this biological and computational precedent, no learned manipulation
policy has integrated whole-body proximity sensing. PLA is the first to do so.

### 2.3 Why This Is Novel

The prior work landscape has three threads, none of which solves this problem:

**Whole-body sensing hardware (no learning):** GenTact (Kohlbrenner et al.,
ICRA 2025), GenTact-Prox (Kohlbrenner et al., ICRA 2026), and ConRich 2025
(Soukhovei et al.) from our lab provide the sensing infrastructure but only
connect it to reactive control systems — not learned policies.

**Tactile sensing + learned policies (wrong scale):** ViSk (CoRL 2024) uses
magnetic skin on the Franka gripper and shows +27.5% improvement — but at
fingertip scale only, contact-only, no language, no whole-body. TACT (T-RO
2025) extends ACT with whole-body tactile on a humanoid — contact only, no
proximity, no VLM.

**VLA policies (vision only):** MolmoBot, OpenVLA, ACT, π₀ — all
vision-language only. None condition on body-mounted proximity sensing.

**The gap:** No prior work integrates distributed pre-contact ToF proximity
across the full robot arm into a deep imitation learning policy with language
conditioning. PLA fills this gap.

-----

## 3. Technical Approach

### 3.1 Hardware: GenTact ToF Skin

We deploy VL53L5CX time-of-flight sensors on the Franka FR3 using the GenTact
procedural skin pipeline. The skin is generated via the GenTact Blender
addon, which procedurally places sensor sites on the arm surface.

**Sensor specifications:**

- Array: 8×8 SPAD zones per sensor
- Field of view: 45°×45° (63° diagonal)
- Range: 20 mm to 4000 mm, 1 mm resolution
- Output: perpendicular distance per zone (on-chip corrected — not slant range)
- Rate: up to 60 Hz in 8×8 mode at <1 m range

**Sensor layout (target — denser end-effector coverage):**

| Link                           | Coverage             | Sensor count |
|--------------------------------|----------------------|--------------|
| link6 + gripper (end effector) | Dense forward-facing | 14–16        |
| link5 (forearm)                | Lateral + forward    | 6            |
| link3 (elbow)                  | All-around           | 6            |
| link2 (upper arm)              | All-around           | 4            |
| **Total**                      |                      | **~30–32**   |

The end-effector region receives denser coverage because near-contact events
during grasping occur primarily in the 0–10 cm zone directly in front of the
gripper.

### 3.2 MuJoCo Simulation

Each sensor is simulated as a fixed pinhole camera in the FR3 MJCF at
GenTact-defined site positions. The simulation pipeline renders depth buffers
at each timestep:

```python
def extend_obs_with_tof(obs, env):
    readings = []
    for cam_name in SENSOR_CAM_NAMES:
        cam_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        renderer.update_scene(env.data, camera=cam_id)
        renderer.enable_depth_rendering()
        depth = renderer.render() * 1000     # metres → mm
        depth = np.clip(depth, 20, 4000)     # VL53L5CX range
        depth += np.random.randn(8, 8) * 5   # σ = 5 mm noise
        readings.append(depth)
    obs['tof'] = np.stack(readings)           # [N_sensors, 8, 8]
    return obs
```

Implemented in `pla/sim/tof.py`.

**Known simulation-to-reality gap:** The noise model does not capture IR
multi-path reflections, specular surface failures, or inter-sensor cross-talk.
This is acknowledged in the paper's limitations section.

### 3.3 Data Collection

Expert trajectories are collected using MolmoBot-Engine TAMP planning in
MolmoSpaces procthor-objaverse scenes (130k+ objects, 94k+ indoor
environments). Target: **1000+ trajectories across 10+ environments**.

**HDF5 schema per episode:**

```
episode_N/
  observations/
    tof:     [T, N_sensors, 8, 8]  float32  — mm, clipped [20, 4000]
    rgb:     [T, 3, 224, 224]       uint8
    qpos:    [T, 7]                 float32  — normalized
  actions:   [T, 7]                 float32  — joint delta, normalized
  metadata:  {task, scene_id, success, seed, policy_phase}
```

Validated by `pla/data/schema.validate`.

**Critical requirement:** At least 30% of trajectories must have ToF readings
below 200 mm (proximity-informative). Standard PnP trajectories may not
achieve this — a dedicated near-contact task must be designed where the
expert path passes within proximity range of fixed obstacles.

### 3.4 PLA Architecture

PLA combines three input modalities decoded by an ACT action head. All
components train jointly except the frozen Molmo2 backbone.

```
RGB + Language  →  [Molmo2-4B, frozen]     →  visual-language tokens  [N_vis, 512]
ToF [N, 8, 8]   →  [Proximity Encoder]     →  proximity tokens        [N,     512]
qpos [7]        →  [Linear(7→512)]         →  proprioception token    [1,     512]
                                                      ↓
                               Fusion: concat → LayerNorm              [N+N_vis+1, 512]
                                                      ↓
                               ACT Decoder (7-layer transformer)
                                                      ↓
                               Action chunk  Δq ∈ ℝ^{100 × 7}
```

Implemented in `pla/models/pla.py` + `pla/models/proximity_encoder.py`.

**Molmo2-4B (frozen):** SigLIP2 vision encoder produces ~192 tokens per RGB
frame at K=2 frames. Qwen3-based LLM provides language grounding. Frozen
entirely — no fine-tuning — to preserve visual-semantic priors from 1.8M
robot trajectories.

**Proximity encoder (shared MLP):** see `pla/models/proximity_encoder.py`.

**ACT action head (exact Zhao et al. 2023 hyperparameters):**

L = (1/k) Σ_j |â_{t+j} − a_{t+j}|_1  +  10 · D_KL(N(μ,σ²) ‖ N(0, I))

- β = 10, constant (not annealed)
- Chunk size k = 100
- LR = 1e-5, batch = 8, Adam
- At inference: z = 0, CVAE encoder discarded

Loss is in `pla/train/losses.py`.

### 3.5 Why Proximity Helps

Three complementary mechanisms explain why proximity sensing improves
manipulation:

1. **Approach direction:** the 8×8 SPAD grid encodes which sub-zone of the
   45° cone an object is approaching from.
2. **Contact imminence gradient:** sensor readings change continuously from
   500 mm to 0 mm as the arm approaches an object — the policy can slow
   approach speed at 100 mm rather than waiting for contact at 0 mm.
3. **Arm-relative spatial awareness:** vision gives object positions relative
   to the camera frame; proximity gives positions relative to the arm body.

These effects emerge from behavioral cloning — no explicit collision
avoidance loss is required.

-----

## 4. Experiments and Baselines

### 4.1 Evaluation Setup

All experiments use MolmoSpaces FrankaPickandPlace benchmarks
(procthor-objaverse, 100 episodes per condition, randomized camera positions).
Hardware: batman — RTX 4090 24GB, Ubuntu 22.04.

**Statistical protocol:** Bootstrap 95% confidence intervals on every
reported number. Paired bootstrap p-values for PLA vs. VLM-only ACT
comparison. Threshold: p < 0.05. See `pla/eval/bootstrap.py`.

### 4.2 Tasks

See `pla/eval/tasks.py`.

| Task            | Description                                              | Expected delta |
|-----------------|----------------------------------------------------------|----------------|
| `pnp`           | Open workspace pick-and-place                            | small          |
| `near_contact`  | **Primary** — fixed obstacle 5–8 cm from expert path     | large          |
| `pnp_color`     | Language-specified object among colored distractors      | moderate       |
| `pnp_next_to`   | Place next to a reference (most challenging language)    | moderate       |

### 4.3 Baseline Ladder

| Method                   | Type                   | Purpose                              | When            |
|--------------------------|------------------------|--------------------------------------|-----------------|
| MolmoBot-Pi0 (zero-shot) | SOTA reference         | Vision-only ceiling                  | Done — 46% Pick |
| Prop-only MLP            | Floor                  | Sanity check                         | Day 4           |
| **VLM-only ACT**         | **Primary comparison** | **Isolates proximity contribution**  | **Day 4–6**     |
| Hand-crafted ToF         | Ablation               | Learned vs. engineered features      | Day 8–10        |
| PLA wrist-only           | Ablation               | Whole-body vs. endpoint              | Day 8–10        |
| PLA Conv2D encoder       | Ablation               | 2D spatial structure                 | Day 8–10        |
| **PLA (ours)**           | **Full method**        | **Main result**                      | **Day 6–8**     |

The VLM-only ACT baseline is the most critical experiment. It is identical to
PLA with proximity tokens removed — one config flag in
`configs/train/act_baseline.yaml`. The delta `PLA − VLM-only ACT` on the
near-contact task is the paper's primary result.

### 4.4 Failure Case Analysis

Categorized in `pla/eval/failure_analysis.py`:

| Failure type                            | Likely cause             | Does proximity help?     |
|-----------------------------------------|--------------------------|--------------------------|
| Approach collision — arm hits obstacle  | Geometric blind spot     | Yes — forearm sensors    |
| Grasp miss — gripper misses target      | End-effector positioning | Maybe — dense EE sensors |
| Place failure — object dropped          | Gripper dynamics         | Unlikely                 |
| Language failure — wrong object         | VLM error                | No                       |

Videos of 3–5 failure cases per type, with proximity sensor readings overlaid,
constitute a key qualitative result.

-----

## 5. Current Status (as of 2026-05-02)

### 5.1 What Is Done

| Item                                  | Status | Notes                          |
|---------------------------------------|--------|--------------------------------|
| MolmoSpaces eval pipeline             | ✅     | Confirmed working on batman    |
| MolmoBot-Pi0 Pick Rand-Cam reproduced | ✅     | 46% (paper: 39.8%)             |
| Pick&Place baseline                   | ✅     | ~44%                           |
| ToF MJCF integration                  | ✅     | 29 sensors, 3 bugs fixed       |
| CVAE proof of concept                 | ✅     | MSE ~8 train and val           |
| Failure case analysis                 | ✅     | Categorized, videos recorded   |

### 5.2 What Needs to Be Done

See [`docs/TIMELINE.md`](TIMELINE.md) for the day-by-day plan.

-----

## 6. The Core Scientific Claim

> **Whole-body distributed time-of-flight proximity sensing is a learnable,
> task-informative modality that improves language-conditioned manipulation
> in near-contact scenarios where RGB observation is insufficient due to
> geometric self-occlusion of the robot arm.**

This claim is:

- **Falsifiable:** tested directly by PLA vs. VLM-only ACT on the
  near-contact task.
- **Novel:** no prior VLA policy conditions on whole-body proximity sensing.
- **Grounded:** biological precedent (peripersonal space), lab precedent
  (Roncone 2016), hardware precedent (GenTact).
- **Specific:** "near-contact scenarios" and "geometric self-occlusion"
  define exactly when and why the claim holds.

-----

## 7. References

1. Deshpande, A., et al. "MolmoBot." arXiv:2603.16861, 2026.
2. Zhao, T.Z., et al. "Learning Fine-Grained Bimanual Manipulation with
   Low-Cost Hardware." RSS, 2023. arXiv:2304.13705.
3. Kohlbrenner, C., et al. "The GenTact Toolbox." ICRA, 2025. arXiv:2412.00711.
4. Soukhovei, P., et al. "Form-Fitting Time-of-Flight Mounts for Robot Skin."
   ConRich, ICRA, 2025.
5. Kohlbrenner, C., et al. "GenTact-Prox." ICRA, 2026. arXiv:2603.04714.
6. Rizzolatti, G., et al. "The Space Around Us." Science, 277(5323), 1997.
7. Roncone, A., et al. "A Computational Model of Peripersonal Space."
   PLOS ONE, 11(10), 2016.
8. Nguyen, A., et al. "ViSk." CoRL, 2024. arXiv:2410.17246.
9. Zhang, Z., et al. "TACT." IEEE T-RO, 2025. arXiv:2506.15146.
10. Kim, M.J., et al. "OpenVLA." arXiv:2406.09246, 2024.
11. Kim, Y., et al. "MolmoSpaces." 2026.
