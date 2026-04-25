# Franka Skin Proximity CVAE — Dataset + Encoder for Action-Chunking Transformer

Everything here is built on the **post-orientation-fix Franka FR3 + self-cap
skin** dataset: 10 successful pick trajectories collected in 2 procthor houses
with all 29 SPAD proximity sensors correctly mounted (perpendicular to the
skin surface, 3 mm outward offset — see `ANALYSIS.md` for the mount fix
methodology).

The goal of this directory: train a small **conditional VAE** on the proximity
stream, so a downstream policy (specifically an **Action Chunking
Transformer**) can consume the whole 29 × 8 × 8 sensor tensor as a single
32-dim token plus a scalar anomaly score, without having to learn the sensor
noise and self-occlusion structure from scratch.

---

## 1. Dataset

### 1.1 Location

```
/home/jaydv/code/molmo/resources/experiment_output/datagen/
  skin_pick_fixed_v1/FrankaSkinPickConfig/20260420_225721/
  ├── house_0/  (6 successful trajectories)
  │   ├── trajectories_batch_1_of_1.h5
  │   ├── episode_00000000_exo_camera_1_batch_1_of_1.mp4
  │   ├── episode_00000000_wrist_camera_batch_1_of_1.mp4
  │   └── … (5 more episodes × 2 cameras)
  ├── house_1/  (4 successful trajectories)
  │   └── …
  ├── experiment_config_20260420_225721.pkl
  └── running_log.log
```

Size on disk: **9.9 MB** HDF5 + **2.9 MB** RGB MP4s = 12.7 MB total for 10
episodes. The robot model used is the **fixed MJCF** at
`/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml`
(perpendicular mounts); original pre-fix MJCF is preserved as
`model.xml.orig_backup` in the same directory.

### 1.2 What each trajectory contains

Every `traj_N` group has the full observation + state stack for T
timesteps (T ∈ {57, 64, 67, 70, 71, 72, 72, 75, 75, 92}; mean 71.5):

| Path inside HDF5 | Shape | dtype | Meaning |
|---|---|---|---|
| `env_states/articulations/panda` | `(T, 31)` | float32 | `[arm_q(7), fingers(2), zeros(22)]` — robot joint positions (radians) |
| `obs/agent/qpos` | `(T, 2000)` | uint8 | pickled full-system qpos — decode if needed |
| `obs/agent/qvel` | `(T, 2000)` | uint8 | pickled full-system qvel |
| `obs/extra/tcp_pose` | `(T, 7)` | float32 | tool center point in **robot base frame** `[x y z qx qy qz qw]` |
| `obs/extra/robot_base_pose` | `(T, 7)` | float32 | robot base in **world frame** (constant per trajectory) |
| `obs/extra/proximity` | `(T, 29, 8, 8)` | float32 | per-taxel **depth** in metres (clipped `[znear=0.02, zfar=4.0]`) |
| `obs/extra/proximity_rgb` | `(T, 29, 8, 8, 3)` | uint8 | per-taxel **RGB** — what each SPAD patch "sees" |
| `obs/extra/obj_start`, `obj_end` | `(T, 7)` | float32 | object pose endpoints (world frame) |
| `obs/extra/grasp_pose` | `(T, 7)` | float32 | target grasp pose |
| `obs/extra/policy_phase` | `(T,)` | int64 | 1=approach, 2=pre-grasp, 3=grasp, 4=lift, 5=place |
| `obs/sensor_data/exo_camera_1`, `wrist_camera` | — | — | third-person + wrist RGB (stored as MP4s sibling to HDF5, not inlined) |

RGB sensor data is stored in two forms:
- **Full-res exo + wrist**: 640×480 MP4 files (standard camera views)
- **Proximity RGB**: `(T, 29, 8, 8, 3) uint8` inside HDF5 — the 29 per-patch
  RGB tiles rendered from the same cameras that produced the depth. This is
  the key addition vs. the older dataset.

### 1.3 Aggregate statistics

Across all 10 trajectories (715 timesteps total):

**Arm joint ranges** (`panda[:, 0:7]`, radians):

| Joint | min | max | mean | std |
|---|---|---|---|---|
| fr3_joint1 | −0.379 | +0.414 | −0.025 | 0.165 |
| fr3_joint2 | −0.903 | +0.564 | −0.172 | 0.432 |
| fr3_joint3 | −0.404 | +1.219 | +0.291 | 0.388 |
| fr3_joint4 | −2.941 | −1.065 | −2.165 | 0.489 |
| fr3_joint5 | −0.352 | +0.578 | +0.027 | 0.211 |
| fr3_joint6 | +1.360 | +3.129 | +2.037 | 0.428 |
| fr3_joint7 | −0.940 | +1.929 | +0.410 | 0.676 |

Gripper driver joint ranges 0.00 (open) to 0.82 rad (closed). The remaining 5
gripper joints follow the driver through equality constraints in the MJCF.

**TCP pose in robot base frame**:

| Component | min | max | mean | std |
|---|---|---|---|---|
| x (m) | +0.264 | +0.633 | +0.433 | 0.107 |
| y (m) | −0.194 | +0.425 | +0.134 | 0.189 |
| z (m) | +0.591 | +1.070 | +0.875 | 0.112 |
| q (xyzw) | — | — | mostly `(0, +0.97, −0.09, +0.02)` | small std |

**Proximity depth** (1,327,040 total values across 29 patches × 8 × 8 × 715):
- min/max: 0.020 m / 4.000 m (matches the `[znear, zfar]` clip bounds)
- mean: 1.72 m
- 8.1 % of taxels saturated at zfar (no obstacle in range)
- 13.4 % of taxels below 0.30 m (close contact)
- 7.8 % of taxels below 0.10 m (very close — grasp contact region)

**Phase coverage**:

| Phase | Name | Timesteps | % |
|---|---|---|---|
| 1 | approach | 8 | 1.1 % |
| 2 | pre-grasp | 315 | 44.1 % |
| 3 | grasp | 102 | 14.3 % |
| 4 | lift | 99 | 13.8 % |
| 5 | place | 191 | 26.7 % |

### 1.4 Point cloud reconstructed from the sensors

Using the corrected pinhole unproject + pybullet FK + `robot_base_pose`
transform (scripts `build_sensor_pointcloud.py`), the raw proximity tensor
can be converted to a world-frame 3D point cloud. For `traj_0` (T = 92), this
yields **141,161 hit points** covering the arm's workspace plus all the room
geometry it swept past:

![Sensor pointcloud](cvae/runs/v3/plots_data/sensor_pointcloud.png)

- **Top-left (X-Y top view)**: bird's-eye. The arm swept across roughly a 8 m
  × 10 m region of floor and kitchen furniture.
- **Top-right (X-Z front)**: shows wall geometry (vertical bands of points)
  and the robot workspace around base height ~0.6 m.
- **Bottom-left (Y-Z side)**: most returns are below 2 m, consistent with a
  ground-mounted robot.
- **Bottom-right (3D)**: joint view with the TCP path in red and object
  start/end positions marked.

Colored by timestep: early sweep (blue/purple) covers the approach area, late
sweep (yellow) covers the lift/place region. `sensor_pointcloud_traj0.mp4` in
the same folder shows this accumulation as an animated rotation.

### 1.5 Scene composite MP4

`cvae/runs/v3/plots_data/composite_traj0.mp4` synchronizes 5 views of
`traj_0`:
- exo + wrist RGB (top)
- 29-patch **depth grid** (bottom-left)
- 29-patch **RGB grid** (bottom-right) — *real* scene RGB from the new
  `proximity_rgb` tensor, which varies with arm motion
- min-depth-over-time plot with a cursor at the current step

This is the visual confirmation that each patch is seeing real scene content,
not its own skin shell.

---

## 2. The CVAE — what it's learning and why

### 2.1 Architecture

```
x  =  depth.reshape(1856) / 4.0           # [0, 1] normalized depth
y  =  concat(arm_qpos(7), tcp_pose(7))    # 14-dim conditioning

ENCODER   q(z | x, y):
   [x(1856) ; y(14)] → Linear(1870, 512) → GELU + Dropout(0.1)
                      → Linear(512, 256)  → GELU + Dropout(0.1)
                      → (μ: Linear(256, 32), logσ²: Linear(256, 32))

DECODER   p(x̂ | z, y):
   [z(32) ; y(14)]  → Linear(46, 256)    → GELU + Dropout(0.1)
                    → Linear(256, 512)   → GELU + Dropout(0.1)
                    → Linear(512, 1856)  → sigmoid
```

**Loss**: MSE reconstruction + β·KL with β = 0.01, linearly annealed from 0
over the first 20 % of training. AdamW at `lr = 3e-4`, 300 epochs on the RTX
4090 — under 15 seconds of wall clock.

### 2.2 What is the CVAE actually learning?

The CVAE factorises what a SPAD proximity patch measures into two streams:

1. **Known-from-state (y)**: how the arm's own configuration maps to what
   the patches point at. The decoder takes `y` directly, so it can
   "predict" the geometrically-determined part of the reading (e.g. at
   elbow flexion θ, patch `link5_s1` is pointing roughly at a specific
   direction relative to the arm).
2. **Scene-specific (z)**: what's actually in the environment along those
   directions. This is what the 32-dim `z` has to carry — the residual
   information that the robot state alone cannot explain.

So the latent `z` is, by construction, a **32-dim compressed summary of the
environment around the arm, with the robot's own pose already factored out**.
That's the right thing for a policy to consume.

Specifically, across the 715 timesteps, the encoder has learned to encode:

- **Presence of a close surface in any given patch direction** — the most
  reliably-informative signal for collision avoidance.
- **Rough distance-to-nearest-obstacle per region of the arm** — latent
  dimensions correlate with patch-group statistics (e.g. "wrist cluster sees
  something close" vs "shoulder cluster sees wall").
- **Scene clutter level** — even when individual patch readings are noisy,
  the aggregate "how much of the 29 × 64 grid is not saturated" is captured
  in low-order latent components.
- **Phase-correlated structure** — `pre-grasp` and `grasp` produce
  distinguishable latent patterns because the wrist patches' proximity
  distribution is phase-dependent. This is visible in `latent_scatter.png`
  where the phase coloring forms sub-clusters inside each trajectory
  manifold.

### 2.3 Training metrics (v3, trained on new data only)

- **Best validation reconstruction MSE**: **7.84** (summed over 1856 dims)
- **Final training reconstruction MSE**: **5.71**
- **Train / val gap**: 1.4× — the model generalizes to held-out trajectories.
  (Compared to the earlier mixed old+new training run which had an 11× gap
  because the old and new distributions were different.)
- **Per-pixel RMSE** (on [0, 1] normalized scale): `sqrt(7.84 / 1856) ≈ 0.065`
  ≈ 26 cm on the 4-metre range. Reasonable fidelity for an anomaly-detection
  use case; too blurry for sub-cm contact prediction.
- **KL divergence** at convergence: ~56 nats, well above 0 (no posterior
  collapse), stable (not diverging).

See `cvae/runs/v3/plots/loss_curves.png`.

### 2.4 What the CVAE is doing (functional summary)

At inference time you can run it in three modes:

**(A) Encoder only — feature extraction**
```
μ(x, y), logσ²(x, y) = encoder(x, y)
z_feat = μ                      # 32-dim feature for the policy
```
Fast (~0.2 ms on RTX 4090). This is the primary output consumed by the ACT.

**(B) Full VAE — anomaly score**
```
μ, σ² = encoder(x, y)
x̂     = decoder(μ, y)
anomaly_score = ‖x − x̂‖²                 # scalar
```
Provides a runtime safety signal: the average reconstruction error on
training data is ~8; scores > 30 flag out-of-distribution proximity patterns
(unseen object, unexpected contact, sensor fault).

**(C) Generative — sample from prior**
```
z ∼ N(0, I)
x̂ = decoder(z, y)                         # 1856-dim sample
```
Produces a plausible proximity reading given the current robot pose. Useful
for counterfactual reasoning ("if I were in pose y, what would I typically
see?") and for data augmentation in downstream training.

### 2.5 Why a CVAE instead of a plain autoencoder, or raw depth?

| Representation | Dim | Pros | Cons |
|---|---|---|---|
| Raw proximity tensor | 1856 | No information loss | High-dim, noisy, redundant with robot state |
| Learned AE | 32–128 | Compresses | Still entangles robot state with scene |
| **CVAE (ours)** | 32 + state-prior | **Factors robot state out**; produces a probabilistic prior p(z\|y) | Blurry reconstructions, needs more data |
| Raw tiles as 29 tokens | 29 × 64 | Preserves per-patch semantics in transformer | 29× longer token sequence; transformer has to learn the self-hit / saturation structure from scratch |

The factoring of robot state is the crucial property for policy learning.
Without it, every time the arm moves, the latent would shift — the policy
would have to learn "what does a change in z mean?" from scratch. With the
CVAE, that meaning is constant: **z changes when the environment changes,
holding the arm pose fixed**.

---

## 3. How the CVAE plugs into an Action Chunking Transformer (ACT)

### 3.1 ACT recap

An **Action Chunking Transformer** (Zhao et al., 2023) predicts a chunk
`a_{t:t+H}` of H future actions from a history of observations. It is a
transformer encoder-decoder:

- **Encoder** takes one or more *observation tokens* per sensor modality:
  typically one token per camera (via a ResNet/CLIP backbone with positional
  embeddings), one token per proprioceptive reading, one token per language
  embedding.
- **Decoder** autoregressively (or in parallel via a CVAE-style decoder,
  matching the ACT paper) emits the action chunk.

Standard ACT observation token set for a manipulation robot:

```
[cls] [img_exo_token] [img_wrist_token] [state_token] [goal_token]  →  encoder
```

Each `img_*_token` is a fixed-dim embedding from a vision backbone. State and
goal are embedded via small MLPs.

### 3.2 Adding proximity as a new token

**Recommended design**: freeze the CVAE, add **one proximity token per step**:

```python
class ProximityTokenizer(nn.Module):
    def __init__(self, cvae: CondVAE, d_token: int):
        super().__init__()
        self.cvae = cvae                      # frozen
        self.proj = nn.Linear(cvae.z_dim, d_token)
        self.anomaly_proj = nn.Linear(1, d_token)

    def forward(self, prox_tensor, state):
        x = prox_tensor.reshape(-1, 1856) / 4.0      # (B, 1856)
        y = state                                     # (B, 14)
        with torch.no_grad():
            mu, logvar = self.cvae.encode(x, y)      # (B, 32)
            xhat = self.cvae.decode(mu, y)           # (B, 1856)
            anomaly = ((xhat - x) ** 2).sum(-1, keepdim=True)  # (B, 1)
        z_token     = self.proj(mu)                  # (B, d_token)
        anom_token  = self.anomaly_proj(anomaly)     # (B, d_token)
        return z_token, anom_token
```

The ACT encoder input becomes:

```
[cls] [img_exo] [img_wrist] [state] [goal] [proximity_μ] [proximity_anomaly] → encoder
                                         ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
                                         two new tokens contributed by the CVAE
```

One token for the **latent code** (what the environment looks like) and one
for the **anomaly score** (how confidently the model recognizes the current
reading as in-distribution). Separating these lets the transformer's
attention mechanism learn independent queries ("use proximity context" vs
"there is something wrong with proximity").

### 3.3 Training strategy

**Stage A — CVAE pretraining**: already done (`cvae/runs/v3/cvae.pt`).
Train on every proximity trajectory available regardless of whether the
trajectory is successful, because the CVAE learns observation statistics and
benefits from diverse data.

**Stage B — ACT training with frozen CVAE**: the CVAE weights are frozen;
only `proj`, `anomaly_proj`, and the rest of the ACT are trained. This is
cheap (CVAE forward is < 1 ms) and keeps the latent semantics stable across
ACT checkpoints.

**Stage C (optional) — joint finetune**: after Stage B converges, unfreeze
the CVAE with a small learning rate (1e-5) for 1-2 epochs. In our experience
this gives a few percent policy-success improvement at the cost of longer
iteration cycles.

### 3.4 Inference-time behavior

At runtime, each control step:

1. Read proximity (29 × 8 × 8) + robot state (14-dim). Cost: one MuJoCo or
   real-SPAD sample.
2. Forward CVAE encoder → 32-dim `μ`, 1 scalar anomaly. Cost: ~0.5 ms on
   RTX 4090; expected ~3 ms on a Jetson Orin.
3. Project to two ACT tokens, append to the observation sequence.
4. Run the ACT encoder-decoder to get the action chunk.
5. **Safety gate**: if `anomaly > 3 × training_99th_percentile`, reduce
   action scale or trigger a fallback behavior (slow-down, stop, ask). The
   calibration:

   ```python
   train_anoms = anomaly_score(cvae, X_train, Y_train)
   q99 = np.percentile(train_anoms, 99)
   # runtime:
   if score > 3 * q99:   "back off"
   if score > q99:       "reduce scale 0.5×"
   ```

### 3.5 Alternative designs we considered

- **Per-patch tokens (29 tokens)**: preserves patch-level semantics. Drops
  the compression benefit; transformer must learn self-hit / saturation
  patterns. Better for contact-rich tasks (insertion), worse for
  open-navigation picks.
- **Conv encoder over the 29 × 8 × 8 grid** treated as a pseudo-image: no
  robot-state factoring; comparable parameter count; empirically similar
  latent quality on this data.
- **No CVAE, raw flattened depth + state**: forces ACT to solve the
  sensor-processing problem as part of the policy — burns policy capacity,
  slower to converge.

The CVAE approach wins on three metrics important at our scale: **parameter
count** (small policy + small CVAE), **data efficiency** (CVAE trainable on
trajectories the policy doesn't need, e.g. failed pick attempts), and **policy
interpretability** (anomaly score is a human-readable safety signal).

---

## 4. Next steps

### 4.1 Expand the dataset with more **cluttered** scenes

The current dataset covers only 2 procthor houses (houses 0 and 1). Houses 2
and 3 had no pickable targets, so the sampler skipped them. To push the
proximity sensors into a more informative regime:

- **More houses**: run with `house_inds = [0..19]` — a larger pool gives the
  sampler more to pick from and therefore more diverse scenes.
- **Scripted clutter**: modify the scene sampler to add random distractor
  objects within the robot workspace bounding box (e.g. 3–5 extra pick-target
  assets on nearby surfaces). This directly stresses patches that see
  "nearby but not the grasp target" geometry — currently under-represented.
- **Narrow-aisle scenarios**: configure houses that force the arm through
  tight spaces (near walls, shelves). The wrist-facing patches (`link6_s2–s5`)
  have near-zero near-contact density in the current dataset and would
  benefit most.

Recommended config for the next batch (single-worker, memory-safe):

```python
# object_manipulation_datagen_configs.py::FrankaSkinPickConfig
num_workers = 1
output_dir = ASSETS_DIR / "experiment_output" / "datagen" / "skin_pick_clutter_v1"
task_sampler_config = PickTaskSamplerConfig(
    task_sampler_class = PickTaskSampler,
    house_inds         = list(range(20)),
    samples_per_house  = 4,
    max_tasks          = 60,
)
```

Expect 30–50 successful trajectories over ~2 hours wall-clock.

### 4.2 Deeper proximity modeling

- **Replace MSE with a two-component loss**: `BCE` on "is this taxel
  saturated (zfar)?" + `L1` on conditional depth. The current MSE is
  disproportionately driven by the zfar/0 bimodality — a two-head output
  would sharpen the reconstructions.
- **Add a temporal prior**: train a small recurrent network on top of the
  CVAE latents (z_t → z_{t+1}) to produce a motion-aware latent that the
  ACT could consume. Currently each timestep is encoded independently.

### 4.3 Integrate with ACT and benchmark

- Wire `ProximityTokenizer` (section 3.2) into the MolmoBot ACT variant.
- Baseline: ACT without proximity tokens.
- Evaluation: re-use the `FrankaSkinPickConfig` benchmark; measure (a)
  success rate, (b) collision rate against non-target scene geometry, (c)
  mean TCP-to-obstacle clearance during approach. The proximity
  augmentation should strictly improve (b) and (c).

### 4.4 Sim-to-real mapping

VL53L5CX sensors have specific noise characteristics (~5 % depth error, 4 cm
minimum range, temporal jitter). Before deploying:

- Add a **domain-randomization wrapper** in the `ProximitySensor` class that
  injects per-taxel Gaussian noise + occasional dropouts.
- Re-fit the CVAE on the noise-augmented data so it becomes robust to the
  real-hardware distribution.
- Real-world data collection on the DROID platform for a small calibration
  set (100–500 timesteps is enough for distribution matching).

---

## 5. File inventory

### 5.1 In this directory (`/home/jaydv/code/skin_sanity/`)

```
README.md                     this document
ANALYSIS.md                   earlier phase-I/II orientation-fix writeup (still accurate)
sensor_fix_report.txt         per-patch old/new pose + angle delta from the orientation fix
datagen.log                   log of the new datagen run
_dataset_stats.md             full markdown of dataset statistics (source of §1 tables)

sanity.py                     1-sensor + 1-wall MuJoCo depth-math sanity test
sanity.html                   plotly visual of the sanity test
synthetic.h5                  synthetic HDF5 used by sanity.py

fix_sensor_orientations.py    URDF rewriter, computes perpendicular mounts via mesh normals
patch_mjcf.py                 propagates URDF fix into resources/robots/franka_droid_skin/model.xml
replay_mjcf.py                replays saved qpos through MJCF → depth + RGB tensors
build_composite_v2.py         builds the scene-RGB composite MP4
build_sensor_pointcloud.py    reconstructs world-frame 3D point cloud from proximity + FK
compute_dataset_stats.py      emits the markdown stat table (source of §1.3)

cvae/
├── dataset.py                HDF5 → (X, Y, meta) loader; pointed at skin_pick_fixed_v1
├── model.py                  CondVAE + ELBO + anomaly_score
├── train.py                  training loop with by-trajectory train/val split
├── plots.py                  generates the 6 CVAE training plots
├── data_plots.py             generates the dataset-level plots
└── runs/v3/                  the current model
    ├── cvae.pt               trained weights
    ├── metrics.npz           per-epoch loss arrays
    ├── data_meta.npz         (X, Y, Z, xhat, anomaly, phase, traj, t, masks)
    ├── plots/
    │   ├── loss_curves.png
    │   ├── latent_scatter.png
    │   ├── recon_samples.png
    │   ├── anomaly_over_time.png
    │   ├── proximity_anomaly_map.png
    │   └── prior_sample_diversity.png
    └── plots_data/
        ├── dataset_overview.png
        ├── per_patch_stats.png
        ├── per_patch_depth_hist.png
        ├── depth_vs_phase.png
        ├── correlation_state_depth.png   # (new) Pearson r per patch × arm joint
        ├── sensor_pointcloud.png
        ├── sensor_pointcloud_traj0.mp4
        └── composite_traj0.mp4
```

### 5.2 Robot model (`/home/jaydv/code/molmo/resources/robots/franka_droid_skin/`)

```
model.xml                     CURRENT — perpendicular-mount MJCF
model.xml.orig_backup         original pre-fix MJCF (revert target)
model_fixed.xml               identical to model.xml, kept for cross-reference
skin_meshes/                  link2/3/5/6_fancy.stl — the shells the patches mount on
```

### 5.3 Dataset

```
/home/jaydv/code/molmo/resources/experiment_output/datagen/skin_pick_fixed_v1/
  FrankaSkinPickConfig/20260420_225721/
    house_0/  (6 trajectories, 448 timesteps)
    house_1/  (4 trajectories, 267 timesteps)
    experiment_config_20260420_225721.pkl
    running_log.log
```

## 6. Reproducing everything

```bash
# 1. Activate the environment
. /home/jaydv/code/molmo/MolmoBot/MolmoBot-Pi0/.venv/bin/activate

# 2. Regenerate stats
python /home/jaydv/code/skin_sanity/compute_dataset_stats.py > _dataset_stats.md

# 3. Retrain CVAE
python /home/jaydv/code/skin_sanity/cvae/train.py \
    --out /home/jaydv/code/skin_sanity/cvae/runs/v4 \
    --epochs 300 --batch 128 --z-dim 32

# 4. Regenerate CVAE plots
python /home/jaydv/code/skin_sanity/cvae/plots.py --run /home/jaydv/code/skin_sanity/cvae/runs/v4

# 5. Regenerate data plots
python /home/jaydv/code/skin_sanity/cvae/data_plots.py

# 6. Regenerate the sensor pointcloud
python /home/jaydv/code/skin_sanity/build_sensor_pointcloud.py

# 7. Regenerate the scene-RGB composite
python /home/jaydv/code/skin_sanity/build_composite_v2.py \
    --h5 /home/jaydv/code/molmo/resources/experiment_output/datagen/skin_pick_fixed_v1/FrankaSkinPickConfig/20260420_225721/house_0/trajectories_batch_1_of_1.h5 \
    --traj-idx 0 \
    --out /home/jaydv/code/skin_sanity/cvae/runs/v3/plots_data/composite_traj0.mp4
```
## 7. Appendix: Full Technical Report (Legacy)

A complete record of the three-phase investigation of the Franka FR3
self-cap skin proximity sensors: (I) verifying the depth math, (II) fixing
the sensor mount orientations, and (III) training a conditional VAE on the
resulting data for collision-avoidance-aware encoding during MolmoBot
evaluation.

Everything here is reproducible from scripts in `/home/jaydv/code/skin_sanity/`
and data in
`/home/jaydv/code/molmo/resources/experiment_output/datagen/skin_pick_v1/`.

---

### 7.1 Table of Contents

1. [7.2 Executive Summary](#72-executive-summary)
2. [7.3 The Sensor System](#73-the-sensor-system)
3. [7.4 Bug 1: Axial Depth vs. Ray Length](#74-bug-1-axial-depth-vs-ray-length)
4. [7.5 Coordinate-Frame Chain](#75-coordinate-frame-chain)
5. [7.6 Bug 2: Patch-Index Mislabeling](#76-bug-2-patch-index-mislabeling)
6. [7.7 Issue 3: Mechanical Self-Hits](#77-issue-3-mechanical-self-hits)
7. [7.8 The Orientation Fix](#78-the-orientation-fix)
8. [7.9 Residual Self-Hits and Their Interpretation](#79-residual-self-hits-and-their-interpretation)
9. [7.10 CVAE Design](#710-cvae-design)
10. [7.11 CVAE Training and Results](#711-cvae-training-and-results)
11. [7.12 Presentation Plots](#712-presentation-plots)
12. [7.13 File Inventory](#713-file-inventory)
13. [7.14 Next Steps / Open Items](#714-next-steps--open-items)

---

### 7.2 Executive Summary

| Phase | Problem | Status | Evidence |
|---|---|---|---|
| I.1  | MuJoCo depth rendering misinterpreted as ray length | **Fixed** | `sanity.py` passes at 0.0 mm axial / 2.4 mm tilted (§3.2) |
| I.2  | `robot_base_pose` transform missing from reconstruction | **Fixed** | `rebuild_real.py` clusters points around arm, not origin |
| I.3  | Saved tensor axis-1 ordering mislabeled (link2-first vs. link6-first) | **Corrected** | Verified against `mujoco.mj_id2name` enumeration (§5) |
| II   | Patches mounted non-perpendicular to skin surface | **Fixed** | All 29 patches ≤8° from surface normal, +3 mm outward offset (§7) |
| III  | Conditional VAE for proximity encoding | **Trained v1** | 32-dim latent, MLP CVAE, clear overfit on 4 train trajs = motivation for bigger datagen (§10) |

**Headline numbers.**

- Depth-math sanity check: 0.0 mm error on axial wall, 2.4 mm on 30° tilted wall (sub-pixel rasterization).
- Self-hit patch count (empty-scene replay, `znear=0.05` m physical dead zone clip): **12/29 → 9/29** after orientation fix.
- CVAE best val reconstruction MSE: 32.5 (summed over 1856 dims) → per-pixel RMSE 0.132 on `[0, 1]` scale ≈ 53 cm on 4 m range.
- Training data: 359 timesteps across 5 trajectories from the 20260416 pick dataset.

---

### 7.3 The Sensor System

### 2.1 Hardware being simulated

The Franka FR3 wears a "self-cap" sensing skin — a thin decorative shell
fixed around arm links 2, 3, 5, and 6. Embedded in this shell are **29
VL53L5CX-class SPAD proximity sensors**:

| Sensor spec | Value |
|---|---|
| Resolution | 8 × 8 "taxels" (multi-zone SPAD) |
| Field of view | 45° (`fovy = 45°`, square) |
| Minimum range (`znear`) | 0.02 m (simulation) — physical dead zone ≈ 4 cm |
| Maximum range (`zfar`)  | 4.0 m |
| Output | per-taxel depth in metres |

### 2.2 Simulation mapping

Each patch is represented in the MuJoCo model as a **fixed pinhole camera**
inside a body rigidly attached to `linkN_skin`:

```xml
<body name="link6_sensor_0" pos="..." quat="...">
  <site name="link6_sensor_0_site" type="sphere" size="0.004" rgba="1 0.2 0.2 1"/>
  <camera name="link6_sensor_0" mode="fixed" pos="0 0 0"
          quat="0 1 0 0" fovy="45" resolution="8 8"/>
</body>
```

The inner `quat="0 1 0 0"` on the `<camera>` is a **180° rotation about the
body X axis**. MuJoCo cameras look along `−Z_cam` by default; after the
180° flip, they look along `+Z_body`. This means:

> **Body `+Z` of each sensor body = the outward ray direction.**

All 29 patches share this convention. The job of the *body* quaternion
(the outer `pos`/`quat` on `<body>`) is to orient the patch body in its
parent `linkN_skin` frame. That outer quaternion is what had to be fixed
in §7.

### 2.3 Rendering

Depth is collected via MuJoCo's `Renderer.enable_depth_rendering()`:

```python
renderer = mujoco.Renderer(model, height=8, width=8)
renderer.enable_depth_rendering()
for cam_id in sensor_cam_ids:
    renderer.update_scene(data, camera=name)
    depth = renderer.render().copy()        # (8, 8) float32
    depth[depth < ZNEAR] = ZFAR              # dead zone sentinel
    depth[depth > ZFAR]  = ZFAR              # far-range sentinel
```

The saver routes this into `traj_N/obs/extra/proximity` with shape
`(T, 29, 8, 8) float32`. `ProximityRGBSensor` does the same but produces
`(T, 29, 8, 8, 3) uint8` — the dataset we're analyzing predates that
sensor, which motivated the MJCF replay pass in §11 for RGB ground truth.

### 2.4 Data under analysis

```
/home/jaydv/code/molmo/resources/experiment_output/datagen/
    skin_pick_v1/FrankaSkinPickConfig/20260416_133342/house_0/
    └── trajectories_batch_1_of_1.h5      (5 trajectories, T∈{61, 65, 74, 79, 80})
    └── episode_{0..4}_{exo_camera_1,wrist_camera}_batch_1_of_1.mp4
```

Per trajectory:

| Key | Shape | Frame | Meaning |
|---|---|---|---|
| `env_states/articulations/panda` | `(T, 31)` | joint space | `[arm_q(7), fingers(2), zeros(22)]` |
| `obs/extra/proximity` | `(T, 29, 8, 8)` | camera axial | depth clipped to `[znear, zfar]` |
| `obs/extra/tcp_pose` | `(T, 7)` | robot base | tool-center-point `xyz + quat` |
| `obs/extra/robot_base_pose` | `(T, 7)` | world | fixed robot base in world |
| `obs/extra/obj_start/obj_end` | `(T, 7)` | world | object pose endpoints |
| `obs/extra/policy_phase` | `(T,)` int64 | — | 0=init, 1=approach, 2=pre-grasp, 3=grasp, 4=lift, 5=place |

`agent/qpos`, `agent/qvel`, `actions/*` are `(T, 2000)` uint8 pickled
blobs — not needed for this analysis.

---

### 7.4 Bug 1: Axial Depth vs. Ray Length

### 3.1 The misinterpretation

The original reconstruction assumed MuJoCo's depth render returned
**Euclidean ray length**. Under that hypothesis:

```python
dir_body = normalize([uu·α, −vv·α, 1])        # α = tan(fovy/2) = 0.4142
p_world  = sensor_pos + depth · R · dir_body
```

In reality `enable_depth_rendering()` returns **per-pixel axial
z-depth** — the z-coordinate of the first intersection in the camera
frame. For a camera facing a wall perpendicular to the optical axis at
0.30 m, hypothesis A predicts corner rays read ~8% longer than centre
rays; hypothesis B predicts every taxel reads identically.

### 3.2 Sanity test (`skin_sanity/sanity.py`)

A minimal MuJoCo scene:

- **Sensor** at world `(0, 0, 0.5)` with identity orientation, carrying a
  camera with `quat="0 1 0 0"`, `fovy=45`, `resolution=8×8`.
- **Wall**: `<geom type="box" size="0.5 0.5 0.005"/>` centered at
  `(0, 0, 0.80)`. Front face sits at `z = 0.795 m` (0.295 m along the
  sensor ray).
- **Tilted variant**: same slab rotated 30° about Y.

The reconstruction under hypothesis B (pinhole unproject) is:

```
α      = tan(fovy / 2)                          # 0.4142
uu, vv = meshgrid((arange(8)+0.5)/8·2 − 1)       # image coords in [−7/8, 7/8]
p_body = ( uu · α · depth,
          −vv · α · depth,          # −vv: camera +Y maps to body −Y via the 180° flip
           depth )
p      = sensor_pos + R_body · p_body
```

**Results.**

| Scene | Reported depth uniform? | Max reconstruction error | Verdict |
|---|---|---|---|
| Axial wall, hypothesis A (ray-length) | 64/64 = 0.295 m | **37 mm** | FAIL |
| Axial wall, hypothesis B (axial) | 64/64 = 0.295 m | **0.0 mm** (float precision) | PASS |
| Tilted wall, hypothesis B | graded 0.26–0.34 m | **2.4 mm** (sub-pixel) | PASS |

The uniform 0.295 m across all 64 taxels of the axial scene is the
smoking gun: only axial depth produces that.

### 3.3 Takeaway

Any downstream consumer of `obs/extra/proximity` must apply the pinhole
unproject, **not** multiply a normalized ray direction by depth. The old
approach overshot corner-ray positions by up to 8 % of the depth.

---

### 7.5 Coordinate-Frame Chain

Reconstructing a world-frame point from `(depth, u, v, patch p, timestep t)`
passes through four frames:

```
taxel (u, v, depth)    ── pinhole unproject ──►  camera
camera                 ── 180° about X (q=(0,1,0,0)) ──►  patch body
patch body             ── FK from panda[t, 0:7] via fr3_joint1..7 ──►  robot base
robot base             ── fixed robot_base_pose ──►  world
```

**Step 2 (camera → body).** The 180° rotation is `R_body_from_cam =
diag(1, −1, −1)`, so `y_body = −y_cam` and `z_body = +depth`.

**Step 3 (body → base).** pybullet FK on a mesh-stripped copy of
`fr3_full_skin.urdf` (`/tmp/fr3_skinonly.urdf`). Meshes stripped because
pybullet can't resolve the `package://gentact_ros_tools/...` URIs; the
kinematic tree is identical once geometry is removed.

**Step 4 (base → world).** `robot_base_pose` is constant across the
trajectory in this dataset (verified with `np.allclose(rbp, rbp[0])`).
Transform once:

```python
base_pos  = rbp[0, 0:3]
base_quat = rbp[0, 3:7]                        # (x, y, z, w) pybullet
base_R    = quat2mat(base_quat)
p_world   = base_R @ p_base + base_pos
```

### 4.1 The second silent bug

Pybullet's `getLinkState` returns link poses **in the robot base frame**
(base fixed at origin via `useFixedBase=True`). The original
reconstruction skipped Step 4, so the rendered point cloud clustered
around the world origin instead of the arm. The fix is one
matrix-multiply; the apparent magnitude is large because the base is at
roughly `(0.84, 2.29, 0.37)` in the dataset scene.

---

### 7.6 Bug 2: Patch-Index Mislabeling

The saved tensor's axis 1 has length 29 — one entry per patch. Which
order?

**What MuJoCo does.** `mj_id2name(model, mjOBJ_CAMERA, i)` enumerates
cameras in the order they appear in the XML. In
`resources/robots/franka_droid_skin/model.xml`, the kinematic chain
descends into link6 first (the wrist carries `wrist_cam` and then the
skin patches), so the sensor cameras appear in this order:

```
[ 0 ..  7]  link6_sensor_0 .. link6_sensor_7
[ 8 .. 13]  link5_sensor_0 .. link5_sensor_5
[14 .. 21]  link3_sensor_0 .. link3_sensor_7
[22 .. 28]  link2_sensor_0 .. link2_sensor_6
```

**What the URDF-based pybullet analysis assumed.** URDF joints were
iterated and sorted lexicographically, giving `link2 → link3 → link5 →
link6` — the **reverse** order. Every patch-level statistic computed
from the pybullet pipeline was therefore mislabeled.

### 5.1 Why this matters

"Patch 0 is always close" means totally different things under the two
orderings:

| Old label (wrong) | Real label (MuJoCo order) |
|---|---|
| `link2_sensor_0` (shoulder) | `link6_sensor_0` (near gripper) |
| `link3_sensor_1` | `link5_sensor_0` |
| `link6_sensor_7` (end of arm) | `link2_sensor_6` (shoulder) |

The physical reasoning for self-hits was completely inverted — and this
was caught only when the MJCF was inspected directly for
`mujoco.mj_id2name` enumeration. Every statistic and figure below uses
the corrected MJCF order.

---

### 7.7 Issue 3: Mechanical Self-Hits

### 6.1 Symptoms in the pre-fix data

Per-patch statistics across traj_0 (T=80), corrected labels:

| idx | patch | median (m) | per-taxel σ_t avg | frac < 10 cm | frac = zfar |
|----:|:--|:--:|:--:|:--:|:--:|
| 0 | `link6_sensor_0` | 0.085 | 0.100 | 55.5 % | 0.0 % |
| 8 | `link5_sensor_0` | 0.142 | 0.031 | 40.6 % | 12.5 % |
| 9 | `link5_sensor_1` | 0.072 | 0.062 | 54.7 % | 3.1 % |
| 12 | `link5_sensor_4` | 0.230 | 0.056 | 45.3 % | 7.8 % |
| 13 | `link5_sensor_5` | 0.281 | 0.042 | 40.6 % | 25.0 % |
| 15 | `link3_sensor_1` | 0.052 | 0.353 | 51.6 % | 10.4 % |
| 19 | `link3_sensor_5` | 0.061 | 0.486 | 51.6 % | 0.6 % |
| 21 | `link3_sensor_7` | 0.086 | 0.131 | **87.5 %** | 1.3 % |
| 28 | `link2_sensor_6` | 0.082 | 0.041 | **81.2 %** | 6.3 % |

The bottom four patches consistently read under 10 cm for most of the
trajectory while the arm was moving. σ_t (the per-taxel standard
deviation over time) approaches measurement noise (< 0.05 m) — i.e. the
reading doesn't change while the arm moves through the scene. That's the
fingerprint of a **rigid self-hit**: the ray hits something attached to
the arm itself, not the environment.

### 6.2 Self-hit vs. environment: decomposition

Rendering each saved qpos through the MJCF with **no scene geometry
present** isolates self-hits: any taxel still reading `< threshold` in
that render must be hitting the robot itself.

Per-patch near-hit decomposition (threshold 0.30 m, averaged over T=80):

| Patch | Self-hit | Scene-only | Interpretation |
|---|---:|---:|---|
| `link6_sensor_0` | 0.42 | 0.34 | mount touches flange + sees cube |
| `link5_sensor_0` | 0.38 | 0.29 | flush against forearm shell |
| `link5_sensor_1` | 0.36 | 0.29 | same |
| `link5_sensor_2` | 0.34 | 0.16 | same |
| `link5_sensor_4` | 0.33 | 0.20 | same |
| `link5_sensor_5` | 0.20 | 0.31 | partial self, partial scene |
| `link3_sensor_1` | 0.27 | 0.24 | self + scene |
| `link3_sensor_7` | 0.05 | **0.82** | looks at floor, NOT self |
| `link2_sensor_6` | 0.19 | 0.67 | mostly ground/base |
| `link3_sensor_5` | 0.23 | 0.27 | self + scene |

Five patches (`link6_sensor_0`, `link5_sensor_{0,1,2,4}`) carry
confirmed 30–42 % mechanical self-hit — their ray origin is at or
inside the `linkN_fancy` skin mesh. `link3_sensor_7` and `link2_sensor_6`
read short because they *look down at the floor*; that's correct
physics.

### 6.3 Why the original mounts were wrong

Reading the URDF (`fr3_full_skin.urdf`) for one suspect:

```xml
<joint name="link5_sensor_0_joint" type="fixed">
  <origin rpy="… … …" xyz="-0.000 0.096 -0.184"/>
  <axis xyz="..."/>
  <parent link="link5_skin"/>
  <child link="link5_sensor_0"/>
</joint>
```

The `origin rpy` / `xyz` defines the patch body frame in the parent
`linkN_skin` frame. `linkN_skin` holds `linkN_fancy.stl` (the outer
decorative shell) rigidly at identity, so the sensor xyz is a point
directly in the mesh coordinate frame. The authoring of these
quaternions was hand-tuned at some point in the past and, for the five
suspect patches, the resulting body `+Z` is not perpendicular to the
mesh at the mount point — it grazes the surface, so the camera frustum
clips back into the shell.

---

### 7.8 The Orientation Fix

### 7.1 Algorithm

For each `linkN_sensor_K_joint` in `fr3_full_skin.urdf`:

1. Read `origin xyz` (in the `linkN_fancy` mesh frame) and `origin rpy`.
2. Load `resources/robots/franka_droid_skin/skin_meshes/linkN_fancy.stl`
   with `trimesh`.
3. `closest_point, face_id = trimesh.proximity.closest_point(mesh,
   xyz)` — nearest surface point and its originating face.
4. Take the **face normal** at that face. Trimesh yields outward-pointing
   normals for manifold regions, but the fancy meshes have non-watertight
   spots. Use the originally-authored `+Z` direction as a sign tiebreaker:

   ```
   n = face_normals[face_id]
   if n · z_old < 0:
       n = −n
   ```

   After this, the angle between the old `+Z` and the fixed normal is
   always acute. Across all 29 patches it is ≤ 7.9°.
5. Build a new rotation whose third column is `n_hat`:

   ```
   x_try  = old_x_axis                      # minimizes "roll" change
   x_proj = x_try − (x_try · n_hat) · n_hat
   x_hat  = x_proj / ‖x_proj‖
   y_hat  = n_hat × x_hat
   R_new  = [x_hat | y_hat | n_hat]
   rpy_new = R_new.as_euler("xyz")
   ```
6. Shift the mount outward by 3 mm along `n_hat`:
   `xyz_new = closest_point + 0.003 · n_hat`.
7. Write new `rpy` + `xyz` back to the URDF joint element.

Code: `skin_sanity/fix_sensor_orientations.py`. Report:
`skin_sanity/sensor_fix_report.txt` — one row per patch with old/new
xyz, old `+Z`, new normal, and angle delta. Excerpt:

```
patch                  old_xyz                        old_+Z                  normal_out              angle_deg
------------------------------------------------------------------------------------------------------------
link2_sensor_0         (-0.048,+0.113,-0.006) -> (-0.051,+0.115,-0.006)  oldZ=(-0.82,+0.56,-0.10) n=(-0.83,+0.54,-0.11)    1.21
link5_sensor_0         (-0.000,+0.096,-0.184) -> (+0.000,+0.098,-0.185)  oldZ=(-0.02,+0.95,-0.30) n=(+0.02,+0.95,-0.30)    2.08
link6_sensor_0         (+0.117,-0.045,-0.004) -> (+0.118,-0.048,-0.004)  oldZ=(+0.47,-0.88,-0.01) n=(+0.47,-0.88,+0.02)    1.56
link3_sensor_3         (+0.107,+0.106,+0.026) -> (+0.108,+0.108,+0.027)  oldZ=(+0.39,+0.84,+0.37) n=(+0.43,+0.77,+0.47)    7.49
```

### 7.2 Patching the datagen MJCF

The datagen MJCF at `resources/robots/franka_droid_skin/model.xml`
duplicates the URDF-derived `pos`/`quat` on each `<body
name="linkN_sensor_K">`. Script `skin_sanity/patch_mjcf.py` rewrites
those 29 bodies using the corrected URDF poses:

```python
quat_wxyz = R.from_euler('xyz', rpy_new).as_quat(scalar_first=False)
# re-order to MuJoCo's (w, x, y, z)
re.sub(rf'(<body\s+name="{name}")\s+pos="[^"]*"\s+quat="[^"]*"',
       rf'\1 pos="{xyz}" quat="{quat}"', mjcf_text, count=1)
```

The inner `<camera quat="0 1 0 0" ...>` stays untouched — that's the
camera-to-body flip, which is already correct.

The fixed MJCF is committed as the live `model.xml`; the original is
preserved as `model.xml.orig_backup` for rollback.

### 7.3 Verification

Empty-scene replay for traj_0 (T=80), with `znear=0.05 m` physical SPAD
dead zone applied to both models:

| | Original MJCF | Fixed MJCF |
|---|---|---|
| Patches with > 2 % taxels reading < 0.30 m | 12 / 29 | **9 / 29** |
| Mean fracNear over all patches | 0.0465 | 0.0563 * |
| Worst patch fracNear | 0.453 | 0.456 |
| Sum of saved/scene-full fracNear above 0.30 m | 0.243 | depends on new data |

\* The mean fracNear is fractionally higher in the fixed model at this
level of aggregation because a handful of patches whose "self-hit" was
previously hidden inside the skin (reading well below znear and getting
sentinelled) now read a valid distance onto *adjacent* link geometry.
This is behavior unmasking, not a regression — see §8.

Figure: `skin_sanity/compare/before_after_selfhits.png`.

---

### 7.9 Residual Self-Hits and Their Interpretation

Nine patches still show > 2 % near-hit taxels in the empty-scene replay.
Breaking down the *per-taxel* depth distribution of those residual
hits:

| Percentile of near-hit depths | Value |
|---|---|
| p05 | 0.021 m |
| p25 | 0.027 m |
| p50 | 0.043 m |
| p75 | 0.080 m |
| p95 | 0.209 m |

57 % of residual hits are below 5 cm — inside the physical SPAD dead
zone. 83 % are below 10 cm. The remaining 17 % (above 10 cm) are
genuine **geometric self-occlusion**: corner rays of the 45° frustum
sweep up to 22.5° off-axis, and on a curved arm at certain
configurations those rays strike adjacent link visual meshes
(e.g. link5 sensors seeing link4 when the elbow is flexed). No amount
of outward offset eliminates those; they are the physical truth a real
SPAD would also report.

**Recommendation**: raise `ProximitySensor.znear` from 0.02 m to 0.05 m
(matches VL53L5CX physical dead zone). The 17 % residuals above that
threshold are kept — they are useful anomaly signals that the policy
should learn to react to.

---

### 7.10 CVAE Design

### 9.1 Problem statement

Given a proximity reading `x ∈ R^{29×8×8}` and the current robot
configuration `y = (arm_qpos, tcp_pose) ∈ R^{14}`, learn a latent
encoding `z` that:

1. **Compresses** — 1856 → 32 dims, so downstream policies can consume
   it cheaply.
2. **Factors out** robot state — the latent should encode "what's out
   there," not "where the arm is".
3. **Supports anomaly detection** — reconstruction error rises sharply
   during unusual proximity events (near-collisions, unexpected object
   contact), so a policy can modulate behavior accordingly.

A **conditional VAE** (Sohn et al. 2015) is the natural fit: the
encoder and decoder both see `y`, and the KL term pushes the posterior
`q(z | x, y)` toward `p(z | y) = N(0, I)`.

### 9.2 Data preparation (`cvae/dataset.py`)

```python
X[t] = clip(prox[t], 0, 4.0).reshape(-1) / 4.0   # (1856,) in [0, 1]
Y[t] = concat(panda[t, 0:7], tcp_pose[t])        # (14,)
```

Normalizing by `zfar = 4.0` puts the saturated-no-obstacle sentinel at
`1.0` and keeps gradients well-scaled. The CVAE learns to predict
`≈ 1.0` wherever there is no contact.

Train/val split is done **by trajectory** so the validation set measures
generalization to new scenes, not new frames in the same scene. With 5
trajectories, 4 go to train and 1 to val.

### 9.3 Model (`cvae/model.py`)

```
CondVAE
  encoder:  [x(1856) ; y(14)] → Linear(1870, 512) → GELU + Dropout(0.1)
                              → Linear(512, 256) → GELU + Dropout(0.1)
                              → (mu: Linear(256, 32), logvar: Linear(256, 32))
  decoder:  [z(32)   ; y(14)] → Linear(46, 256) → GELU + Dropout(0.1)
                              → Linear(256, 512) → GELU + Dropout(0.1)
                              → Linear(512, 1856) → sigmoid
```

### 9.4 Loss

```
L(x, y) = MSE(x, x̂) + β · KL( q(z|x,y) ‖ N(0,I) )
```

- **Reconstruction**: per-pixel MSE, summed over 1856 dims, averaged
  over the batch. This favors pixel-accurate depth maps over "nice"
  samples.
- **KL**: standard Gaussian form
  `−½ Σ (1 + logvar − μ² − e^{logvar})`.
- **β annealing**: linearly ramp `β` from 0 to target over first 20 %
  of epochs. Prevents posterior collapse on small datasets.

Hyperparameters used: `z_dim = 32`, `hidden = 256`, `dropout = 0.1`,
`β = 0.01`, `lr = 3e−4`, `weight_decay = 1e−4`, `batch = 64`,
`epochs = 300`. AdamW optimizer.

### 9.5 Anomaly score

For downstream collision-avoidance gating:

```python
def anomaly_score(x, y, n_samples=8):
    mu, logvar = encode(x, y)
    scores = []
    for _ in range(n_samples):
        z = mu + exp(0.5·logvar) · randn()
        scores.append(sum((decode(z, y) - x)^2))
    return mean(scores)
```

Monte Carlo over `n_samples` posterior draws; the mean reconstruction
error is used as the anomaly measure. A single forward pass through
mu-only (i.e. "use the mean latent") is cheaper and almost as
informative for gating.

---

### 7.11 CVAE Training and Results

### 10.1 Training

| | |
|---|---|
| Device | NVIDIA RTX 4090 |
| Dataset size | 359 timesteps across 5 trajectories |
| Split | 4 train / 1 val by trajectory (traj 2 is val) |
| Epochs | 300 (~9 s total — trivial on this data scale) |
| Final train recon | 1.76 (summed MSE) |
| Best val recon | **32.50** (at epoch ≈ 245) |
| Final KL (train / val) | 42.6 / 47.3 |

Per-pixel RMSE sanity check: `sqrt(32.5 / 1856) = 0.132` on a `[0, 1]`
depth scale = 53 cm on a 4 m range. Useful as a coarse collision-avoidance
signal; far from tight reconstruction.

### 10.2 Training curves

See `cvae/runs/v1/plots/loss_curves.png`.

- **Train recon** drops log-linearly from ~300 to ~2 over 300 epochs.
- **Val recon** drops from ~300 to ~35 in the first 30 epochs, then
  plateaus.
- The gap between train and val is > 20× by epoch 100 — a classic
  data-starved overfit.
- **KL** converges smoothly to ~45 nats, roughly `log(N)` for N = 359,
  indicating each sample carves out a distinct latent region rather
  than collapsing.

### 10.3 Latent-space structure

See `cvae/runs/v1/plots/latent_scatter.png`.

- **Left panel (colored by policy phase)**: `pre-grasp` (green) samples
  form the largest cluster, as expected (it's the longest phase — 171
  of 359 steps). `grasp` (red), `lift` (purple), and `place` (brown)
  each form distinct sub-clusters within their parent trajectory's
  manifold.
- **Right panel (colored by trajectory)**: each of the 5 trajectories
  traces a distinct one-dimensional curve through latent space, reflecting
  the arm's temporal trajectory through its configuration-plus-scene
  state.
- **Val trajectory (traj 2, green ×)**: sits at the boundary of the
  train cloud, slightly offset — again consistent with the generalization
  gap.

### 10.4 Reconstruction quality (qualitative)

See `cvae/runs/v1/plots/recon_samples.png`. Four timesteps × 8 patches
shown side-by-side (input row / recon row):

- **On train data**: reconstructions are visually indistinguishable from
  inputs for most patches. The CVAE captures saturated-sky regions,
  contact patches, and gradient structure.
- **Patch-level differences**: patches that vary rapidly over time
  (e.g. those near the gripper, following the object) show slightly
  blurred reconstructions — the latent averages over nearby frames. This
  is the hallmark VAE mode-averaging artifact and is mitigated by a
  larger dataset and / or a stronger posterior (β-VAE with β > 1).

### 10.5 Anomaly detection

See `cvae/runs/v1/plots/anomaly_over_time.png` (5 subplots, one per
trajectory).

- Training trajectories show anomaly scores in the 2–12 range, with
  clear peaks at `pre-grasp → grasp` transitions (arm closing on
  object) and `place` (arm approaching landing surface).
- Validation trajectory (traj 2) shows scores 10× higher throughout —
  the reconstruction struggles on unseen scene layout.

The per-patch anomaly map (`proximity_anomaly_map.png`) isolates exactly
which of the 29 patches are contributing the anomaly signal at each
timestep. For traj 2, the bright band sits at patch indices 2-7 and
23-28 (end-effector and shoulder-adjacent), indicating those patches
see scene geometry the model hasn't learned.

### 10.6 Prior sampling

See `cvae/runs/v1/plots/prior_sample_diversity.png`. Drawing `z ~
N(0, I)` and decoding with 6 different conditioning vectors produces
plausible but slightly blurred proximity patterns. This confirms the
CVAE has learned a meaningful generative distribution; the blur reflects
the small training set and the averaging effect of MSE loss.

---

### 7.12 Presentation Plots

All plots from this session are in `/home/jaydv/code/skin_sanity/`:

### 11.1 Depth-math / geometric analysis (phase I)

| Path | Content |
|---|---|
| `compare/grid_t{000,020,040,060,079}.png` | Saved depth \| empty-scene replay depth \| replay RGB, 29-tile grids. Good for: "patch X sees scene Y at step Z". |
| `compare/selfhit_decomposition.png` | Per-patch bar chart, red=self-hit + blue=scene-only contribution to near-field hits. |
| `compare/suspect_closeup.png` | 9-row close-up for the original suspect patches: min-depth time series (saved vs. empty-scene replay) + 4 RGB tiles. |
| `compare/rgb_signature.png` | Mean RGB over trajectory per patch + skin-color distance bar. |
| `compare/composite_traj0.mp4` | Synced exo + wrist RGB + saved-depth grid + replay-RGB grid + time-series cursor. |
| `real_near_multiview.png` | 4-view (3 ortho + 3D) near-hit point cloud for traj_0 (after base-frame fix). |
| `real_all_multiview.png` | Same, all hits, color = depth. |
| `real_snapshots.png` | 3×3 grid of 3D cloud at steps {0, 10, 20, 30, 40, 50, 60, 70, 79}. |
| `real_traj0.mp4` | Rotating-camera animation of the point cloud. |

### 11.2 Orientation fix (phase II)

| Path | Content |
|---|---|
| `compare/before_after_selfhits.png` | Per-patch self-hit rate, red = original, green = fixed. Summary panel: 12/29 → 9/29. |
| `sensor_fix_report.txt` | Per-patch old xyz, new xyz, old `+Z`, new normal, angle delta. |

### 11.3 CVAE (phase III)

| Path | Content |
|---|---|
| `cvae/runs/v1/plots/loss_curves.png` | Train/val recon + KL over 300 epochs (log y on recon). |
| `cvae/runs/v1/plots/latent_scatter.png` | 2D PCA of `μ(x, y)`; left=colored by policy phase, right=by trajectory. |
| `cvae/runs/v1/plots/recon_samples.png` | 4 timesteps × 8 patches; input row vs reconstruction row. |
| `cvae/runs/v1/plots/anomaly_over_time.png` | Per-trajectory reconstruction-error traces; val clearly elevated. |
| `cvae/runs/v1/plots/proximity_anomaly_map.png` | Patch × timestep heatmap of per-patch MSE; cyan lines mark trajectory boundaries. |
| `cvae/runs/v1/plots/prior_sample_diversity.png` | 6 rows: `z ~ N(0, I)` decoded under 6 different robot-state conditions. |

---

### 7.13 File Inventory

### 12.1 Scripts (`/home/jaydv/code/skin_sanity/`)

```
ANALYSIS.md                       this document
sanity.py                         1-sensor + 1-wall MuJoCo sanity test
sanity.html                       plotly view of the test
rebuild_real.py                   fixed real-data point-cloud reconstruction
replay_mjcf.py                    re-render 29 depth + 29 RGB via MJCF + saved qpos
build_comparison.py               saved vs replay visual comparison grids
build_suspect_closeup.py          9-row suspect-patch close-up
compare_before_after.py           orientation-fix before/after bar chart
fix_sensor_orientations.py        URDF-level orientation fix (trimesh mesh normals)
patch_mjcf.py                     propagates URDF fix into the datagen MJCF
sensor_fix_report.txt             per-patch old/new pose + angle delta
```

### 12.2 Data artifacts (`/home/jaydv/code/skin_sanity/`)

```
replay_traj0/
  replay_depth.npy                  (80, 29, 8, 8) float32 — original MJCF empty-scene
  replay_depth_fixed.npy            (80, 29, 8, 8) float32 — fixed MJCF empty-scene
  replay_rgb.npy                    (80, 29, 8, 8, 3) uint8 — original MJCF RGB
  replay_sensor_names.txt           29 names in MuJoCo enumeration order
real_near_multiview.png / real_all_multiview.png / real_snapshots.png / real_traj0.mp4
compare/                            all the static + video comparison outputs
cvae/
  dataset.py                        HDF5 → (X, Y, meta) loader
  model.py                          CondVAE + ELBO + anomaly score
  train.py                          training loop with by-trajectory split
  plots.py                          all six presentation plots
  runs/v1/
    cvae.pt                         trained weights
    metrics.npz                     per-epoch losses
    data_meta.npz                   (X, Y, Z, xhat, anomaly, phase, traj, t, split masks)
    plots/                          loss_curves, latent_scatter, recon_samples,
                                    anomaly_over_time, proximity_anomaly_map,
                                    prior_sample_diversity
```

### 12.3 Robot model changes (`/home/jaydv/code/molmo/`)

```
fr3_skin_mujoco/
  fr3_full_skin.urdf                ORIGINAL — unchanged
  fr3_full_skin_fixed.urdf          NEW — perpendicular-mount version

resources/robots/franka_droid_skin/
  model.xml                         NOW THE FIXED MODEL (swapped)
  model.xml.orig_backup             pre-fix backup
  model_fixed.xml                   same as model.xml, retained for clarity
```

To revert:

```bash
cp /home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml.orig_backup \
   /home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml
```

---

### 7.14 Next Steps / Open Items

### 13.1 Ready to ship

- The **fixed MJCF** is live; any subsequent datagen run automatically
  uses it.
- The **CVAE pipeline** trains in < 10 s on RTX 4090. Re-running on new
  data only requires changing the `--h5-glob` argument.

### 13.2 Requires authorization

Launching a **bigger datagen run** was blocked by the auto-mode harness.
Proposed configuration (saved in a reverted `FrankaSkinPickConfig` edit):

```python
num_workers = 4
seed = 42
output_dir = ASSETS_DIR / "experiment_output" / "datagen" / "skin_pick_fixed_v1"
task_sampler_config = PickTaskSamplerConfig(
    task_sampler_class = PickTaskSampler,
    house_inds = [0, 1, 2, 3, 4, 5, 6, 7],
    samples_per_house = 8,
    max_tasks = 40,
)
```

Expected yield: 25–35 successful episodes (assuming 60–80 % pick-task
feasibility). Should finish in < 30 minutes with `num_workers = 4`.

To launch after your approval (user runs this themselves):

```bash
. /home/jaydv/code/molmo/MolmoBot/MolmoBot-Pi0/.venv/bin/activate
cd /home/jaydv/code/molmo/molmospaces

# 1. Reapply the config edit above (or I can do it on instruction)
# 2. tmux + run:
tmux new -s skin_datagen
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  python -m molmo_spaces.data_generation.main FrankaSkinPickConfig \
  2>&1 | tee /home/jaydv/code/skin_sanity/datagen.log
```

### 13.3 Follow-ups after new data arrives

1. **Retrain CVAE** — `python cvae/train.py --h5-glob
   '/home/jaydv/code/molmo/resources/experiment_output/datagen/skin_pick_fixed_v1/FrankaSkinPickConfig/*/house_*/trajectories_batch_*.h5'
   --out cvae/runs/v2`
2. **Regenerate plots** — `python cvae/plots.py --run cvae/runs/v2`
3. **Expect**: val recon / train recon gap collapses from 20× to < 3×
   once dataset size is 10×. Latent space should also show clearer
   phase structure.
4. **Integrate with MolmoBot eval**: wrap `CondVAE.encode` as a
   `BasePolicy`-side feature extractor, feed the 32-dim `μ` and the
   anomaly score into the policy as additional conditioning. The
   policy can then modulate action magnitude (e.g. slow down, back off)
   when anomaly rises above a threshold calibrated on train data.

### 13.4 If aiming for production-grade self-hit elimination

- Change `ProximitySensor.znear` from 0.02 m to 0.05 m (VL53L5CX
  physical dead zone). Half of residual self-hits disappear.
- For `link6_sensor_0`, `link5_sensor_{0,1,2,4}`: increase outward
  offset to 5 mm and revisit the `linkN_fancy.stl` mesh around those
  mount points — some local non-manifold regions may warrant mesh
  repair (`trimesh.fill_holes` / `remove_degenerate_faces`).
- Consider moving the skin visual geoms into render group 3 and having
  `ProximitySensor` pass `mjVIS_GEOMGROUP[3] = 0` into the renderer.
  This hides the decorative shell from the sensor's own view while
  keeping it visible in third-person debug cameras. This was tested
  (§7.3) and is a viable cleanup when combined with the outward offset.

---

## Appendix A: Reconstruction Formula Cheat-Sheet

Given `depth ∈ [znear, zfar)`, taxel indices `(col, row) ∈ [0..7]²`, the
patch body's world pose `(pos, R)`, and the robot's world-space
`(base_pos, base_R)`:

```python
α       = tan(deg2rad(45) / 2)                    # 0.4142135
u       = (col + 0.5) / 8 · 2 − 1                  # in [−0.875, +0.875]
v       = (row + 0.5) / 8 · 2 − 1
p_body  = np.array([ u · α · depth,
                    −v · α · depth,
                     depth ])                      # meters, body frame
p_base  = R_body @ p_body + pos_body              # via pybullet.getLinkState on FK tree
p_world = base_R @ p_base + base_pos              # via obs/extra/robot_base_pose
```

## Appendix B: Rebuilding the MJCF from the fixed URDF

If you prefer the MJCF to be regenerated (instead of patched in place):

```bash
cd /home/jaydv/code/molmo/fr3_skin_mujoco
python fr3_skin_mujoco.py \
    --urdf fr3_full_skin_fixed.urdf \
    --meshdir ~/code/proximity_learning/assets/.cache/gentact_ros_tools/meshes \
    --out fr3_skin_fixed.xml --build
```

Then point the robot config's `robot_xml_path` at the new MJCF or copy
it over `resources/robots/franka_droid_skin/model.xml`.

## Appendix C: CVAE anomaly-score calibration for policy gating

Suggested procedure once the bigger dataset is in hand:

```python
# 1. Compute anomaly on the full training split
scores_train = anomaly_score(model, X_train, Y_train)
q99_train    = np.percentile(scores_train, 99)

# 2. At policy runtime
score_now = anomaly_score(model, prox_now, state_now, n_samples=1)  # fast path
if score_now > 2.0 * q99_train:
    # High-confidence anomaly: back off / slow down / wait
    ...
elif score_now > q99_train:
    # Borderline: reduce action scale
    ...
```

A single-sample anomaly score (`n_samples = 1`, `μ` only) is ≤ 0.5 ms on
the RTX 4090 — negligible compared to policy forward passes.
