# Repo file structure — PLA training/eval artifacts

This document describes every file produced or consumed by the PLA training
+ eval pipeline. It complements `README.md` (the project / sim writeup) and
`TODO.md` (the session log). The pipeline lives under five top-level
directories:

- [`pla/`](#pla) — source code (the model + data + train + eval + rollout stack)
- [`diagnostics_output/`](#diagnostics_output) — dataset sanity check plots + pointcloud
- [`runs/`](#runs) — saved training checkpoints
- [`eval_output/`](#eval_output) — outputs from `pla.eval` (the JsonBenchmark route, currently blocked)
- [`rollout_output/`](#rollout_output) — outputs from `pla.rollout_eval` (the custom-rollout route that worked end-to-end)
- [`analysis_output/`](#analysis_output) — outputs from `pla.rollout_compare` (PLA vs baseline side-by-side)
- [`logs/`](#logs) — captured stdout from every training, eval, datagen, rollout run

There is also `scripts/` (one-off utilities) and `assets/eval_subsets/`
(the patched JSON benchmark that `pla.eval` would consume if the schema
issue were fixed upstream). Those are listed at the end.

---

## Vocabulary used in this document

These short tags appear all over the file names — defining once.

| Tag / suffix | Meaning |
|--------------|---------|
| **smoke** | A small, fast validation slice. Used in two senses: (1) the **smoke dataset** = 36 successful franka_skin pick-and-place demonstrations on iTHOR houses 1-10 collected with `FrankaSkinPickAndPlacePilotSmokeConfig`; (2) a **smoke test** = a quick end-to-end run on a tiny input to verify plumbing before committing compute to the full scale. E.g. `rollout_smoke_test/` is a 1-house, 2-episode trial of the rollout pipeline before launching the full 10-house run. |
| **holdout** | Data the policy has never seen. We have two: (a) the *eval-holdout dataset* (houses 11-20, seed 2027, generated with `FrankaSkinPickAndPlacePilotEvalHoldoutConfig`) used to build a benchmark of expert trajectories; and (b) the *rollout holdout* (houses 11-20, seed 2028) used at policy eval time. The seed difference means the rollout encounters fresh task instances even on the same houses. |
| **pilot** | Production-scale data-gen runs. `pilot_smoke` = 10 houses (for proximity-pipeline validation); `pilot_medium` = 100 houses (registered, not launched); `pilot` = 1999 houses (the full dataset, not collected). |
| **v3_full** | Third revision of a full 20 000-step training run (v1 was the 500-step shakedown). |
| **pla / vlm_only_act** | The two ablation variants: PLA = ACT + 29 proximity tokens (`use_proximity=true`); vlm_only_act = same architecture, proximity tokens zeroed (`use_proximity=false`). |

---

<a id="pla"></a>
## `pla/` — source

The training/eval stack, code-complete except for language conditioning.
**Total: 2139 lines across 9 modules + an init.**

```
pla/
├── __init__.py            (7 lines)   Package init.
├── dataset.py             (~430 lines) FrankaSkinHDF5Dataset + precompute_language_embeddings.
├── proximity_encoder.py   (63 lines)  Shared MLP (B,29,8,8) → (B,29,512).
├── policy.py              (~385 lines) PLA_DETRVAE wrapper + PLAPolicy loss/optim. Adds lang token.
├── language_encoder.py    (~50 lines)  CLIPTextEncoder: frozen HF ViT-B-32 text tower, 512-d output.
├── train.py               (~240 lines) CLI training entry point. Precomputes CLIP at startup.
├── diagnostics.py         (246 lines) Dataset sanity-check plots + summary.json.
├── eval.py                (180 lines) JsonBenchmark eval CLI (currently blocked).
├── eval_policy.py         (~200 lines) InferencePolicy wrapper. Loads CLIP, encodes at reset().
├── rollout_eval.py        (~300 lines) Custom rollout eval (datagen-pipeline route).
└── rollout_compare.py     (228 lines) PLA vs baseline per-episode classifier.
```

**Two-environment requirement (as of 2026-05-12):**
- **Training** uses `submodules/MolmoBot/MolmoBot-Pi0/.venv/` (has CLIP / transformers / ACT
  dependencies; does not have `mujoco_warp`, which is fine because training reads h5
  data and never spins up the simulator).
- **Rollout + data collection** uses `/opt/conda/envs/mlspaces/` (has `mujoco_warp`
  installed via pip + filament renderer; `transformers` was added on 2026-05-12 so
  CLIP can encode at inference). The "do not use mlspaces" anti-pattern only applies
  to training paths that need MolmoBot-Pi0's torch ABI.

See `scripts/launch_medium_v1.sh` for the canonical pipeline that uses the right
env for each phase.

### `pla/__init__.py`

7 lines. Re-exports `dataset`, `proximity_encoder`, `policy` for cleaner
imports.

### `pla/dataset.py`

`FrankaSkinHDF5Dataset` — the PyTorch `Dataset` over the HDF5 trajectories
produced by molmospaces' data-gen pipeline.

What it does:
- Walks `<dataset_root>/house_*/trajectories_batch_*.h5`, indexes one
  sample per timestep across all successful trajectories.
- Per sample: reads 29 proximity sensors (`obs/proximity/<sensor>`, shape
  `(T, n_substeps, 8, 8)`, mean-pooled over substeps, divided by
  `depth_max_m=4.0` and clipped to [0,1]); 7-DOF arm qpos (JSON-encoded
  in `obs/agent/qpos` uint8 blobs); and an 8-DOF action chunk of length
  `k=100` (7 arm joints + gripper command `{0.0, 255.0}` rescaled to
  `{0, 1}`).
- Optionally reads RGB frames for `exo_camera_1` + `wrist_camera` from the
  per-episode `.mp4` files via decord; resizes to the configured
  resolution; returns `(num_cam, 3, H, W) ∈ [0, 1]`.
- Optionally returns the per-trajectory language string parsed from
  `obs_scene`.
- `use_proximity=False` zeros the proximity tensor so the same loader
  feeds both the PLA and baseline trainings.

Key class: `FrankaSkinDatasetConfig` (Pydantic-style dataclass with all
the knobs: chunk_size, qpos_dim, action_dim, depth_max_m, image_resolution,
sample_full_episodes, etc.).

### `pla/proximity_encoder.py`

The shared MLP that turns 29 SPAD frames into 29 hidden tokens. Per
TODO §2:

```python
self.mlp = nn.Sequential(
    nn.Linear(64,  128), nn.ReLU(inplace=True),
    nn.Linear(128, 512),
)
# forward: (B, 29, 8, 8) → reshape → (B*29, 64) → MLP → (B*29, 512) → (B, 29, 512)
```

Weights are shared across all 29 sensors. ~90 k trainable params.

### `pla/policy.py`

The core wrapper around upstream ACT. Reuses
`submodules/act/detr/models/detr_vae.py`'s `Transformer`, `Backbone`,
CVAE encoder, and `get_sinusoid_encoding_table` **without modification**.

Two key classes:
- `PLAConfig` — dataclass with every hyperparameter
  (chunk_size=100, hidden_dim=512, enc_layers=dec_layers=7, nheads=8,
  dim_feedforward=2048, kl_weight=10, qpos_dim=7, **action_dim=8**, etc.).
- `PLA_DETRVAE` — mirrors upstream `DETRVAE`. When `use_proximity=True`,
  the encoder context becomes
  `[latent_z, qpos, *29 proximity tokens, *image tokens]`; otherwise it
  falls back to the upstream `[latent_z, qpos, *image tokens]`. Adds 29
  extra slots in `additional_pos_embed` to position the proximity tokens.
- `PLAPolicy` — wraps `PLA_DETRVAE` with ImageNet normalization, the L1
  + 10·KL loss, and `configure_optimizer` returning Adam with separate
  param groups for backbone vs the rest.

Param counts (verified): PLA 96.37 M, baseline 96.28 M (diff ≈ 90 k for
the proximity branch).

### `pla/train.py`

CLI training entry. Adam, lr=1e-5, batch_size=8, L1+10·KL loss,
WandB logging (on by default since 2026-05-11), checkpoints every
`--ckpt_every` steps with a stable `latest.pt` symlink.

Defaults: `--num_steps 20_000`, `--num_workers 2` (RAM-safe for the 62 GB
box), `--use_wandb true`. Resumable via `--resume <path-to-ckpt>`.

### `pla/diagnostics.py`

Generates 5 PNGs + a `summary.json` for dataset sanity checks. The
top-line gate is `proximity_signal_ok: bool` (True iff any sensor's max
depth > 1 mm) — if False, don't bother training.

CLI: `python -m pla.diagnostics --root <dataset_root> --out <out_dir>`.

### `pla/eval.py`

CLI for the **JsonBenchmark eval route** — wraps
`molmo_spaces.evaluation.run_evaluation` with `PLABenchmarkEvalConfig`.

**Currently blocked**: the upstream `CameraSpec` schema has no per-camera
resolution, so the 29 SPAD sensors render as `(352, 624, 3)` RGB instead
of `(8, 8)` depth. The class definitions are correct and module-level
(picklable), so the moment the schema is extended upstream, this works.
See `README.md` §7.5.d for the three fix paths.

Key classes (module-level for picklability):
- `PLAPolicyConfig(BasePolicyConfig)` — wraps the checkpoint path,
  use_proximity, image_h/w, depth_max_m, gripper_schedule, etc.
- `PLABenchmarkEvalConfig(JsonBenchmarkEvalConfig)` — wires
  `FrankaSkinRobotConfig`, `FrankaSkinCameraSystem`, our
  `PLAPolicyConfig`, `terminate_upon_success=True`. CLI overrides land in
  a module-level `_EVAL_OVERRIDES` dict that `model_post_init` consumes.

### `pla/eval_policy.py`

`PLAInferencePolicy(InferencePolicy)` — the molmospaces-compatible
inference wrapper. Loads our checkpoint, buffers action chunks of size
100 (refilled when drained), reads obs (`qpos`, `exo_camera_1`,
`wrist_camera`, 29 `link*_sensor_*`), resizes images, normalizes
proximity, runs `model.forward` for inference. Predicts gripper directly
from the 8th action channel (thresholded at 0.5 to snap to `{0, 255}`).

### `pla/rollout_eval.py`

CLI for the **custom-rollout eval route** — the workaround for the
JsonBenchmark schema blocker. Reuses the data-generation pipeline (which
uses `FrankaSkinCameraSystem` natively, preserving the 8×8 SPAD depth
path) but swaps the planner for our trained policy.

Defines `FrankaSkinPLARolloutConfig`, a subclass of
`FrankaSkinPickAndPlacePilotEvalHoldoutConfig` with broadened
`policy_config: BasePolicyConfig` field and
`filter_for_successful_trajectories=False` (so failures get saved too).
`model_post_init` reads `_ROLLOUT_OVERRIDES` and installs our
`PLAPolicyConfig` + house list + seed + workers.

Tasks are deterministic given (houses, seed). With the same seed for
PLA + baseline, each policy attempts the same set of tasks → head-to-head.

After the datagen pipeline finishes, it walks the saved h5 files, counts
per-trajectory success, writes `results.json` with Wilson 95% CI.

### `pla/rollout_compare.py`

Side-by-side classifier for two `pla.rollout_eval` runs. For each
matching `(house, traj_key)` pair, assigns a bucket:
- **A**: baseline failed, PLA succeeded — the proximity-helps headline
- **B**: baseline succeeded, PLA failed — sanity check
- **C**: both succeeded
- **D**: both failed (typical at low training scale; sub-analysed by
  behavioural gap)

Per-episode metrics computed: `tcp_to_pickup_{start,end}`,
`approach_delta_m = d_start − d_end`, `gripper_open_frac`,
`link6_prox_min_end_m`. Output:
`<out_dir>/comparison.{md, json}`.

### `pla/diagnostics_output/` (subdir)

Leftover from an earlier diagnostics run pointed inside the package
directory. The canonical diagnostics outputs live at
`<repo>/diagnostics_output/` (see below). This subdir is safe to ignore.

---

<a id="diagnostics_output"></a>
## `diagnostics_output/` — dataset sanity check plots

Produced by `python -m pla.diagnostics --root <dataset> --out <here>`.
The gate before any training run.

```
diagnostics_output/pilot_skin_smoke_v1/
├── summary.json                          (6.5 KB)  proximity_signal_ok, per-sensor stats, lengths
├── 01_proximity_depth_histogram.png     (20 KB)   depth-value histogram across all 29 sensors
├── 02_proximity_per_sensor_stats.png    (43 KB)   per-sensor mean depth + zero-pixel fraction
├── 03_episode_length_hist.png           (24 KB)   trajectory length distribution
├── 04_qpos_action_distribution.png      (52 KB)   per-joint qpos + action histograms (7+7)
├── 05_language_top_descriptions.png     (124 KB)  top-20 task descriptions
└── episode_house2_traj0/                          single-trajectory deep dive
    ├── pointcloud.ply                   (21 MB)   1.79 M world-frame points, view in MeshLab
    ├── pointcloud_full.png              (399 KB)  2D projection of the whole episode
    ├── pointcloud_at_t.png              (522 KB)  colored by Franka link, t-snapshots
    ├── pointcloud_overlay.png           (982 KB)  projected onto wrist + exo RGB
    ├── sensor_panel.png                 (996 KB)  29 depth tiles + RGB at the same instant
    ├── sensor_min_depth.png             (327 KB)  closest-thing-each-sensor-sees over time
    ├── proximity_traces.png             (137 KB)  per-sensor mean depth over time
    ├── proximity_heatmap.png            (82 KB)   29 sensors × T heatmap
    ├── rgbd_samples.png                 (1.1 MB)  representative RGB + depth frames
    ├── states.png                       (230 KB)  qpos / qvel / tcp_pose over time
    └── report.md                        (5 KB)    4-question verification (Q1-Q4 pass)
```

`summary.json` is the machine-readable gate: `proximity_signal_ok = true`
must hold before training. For the smoke set this passed (99.94% nonzero
proximity pixels). For the original (broken) 20260508 pilot it failed —
that's how we discovered the `proximity_sensor_period_ms=0` bug.

`episode_house2_traj0/` is generated by a separate script
(`submodules/molmospaces/scripts/datagen/analyze_sample_episode.py
<h5> --traj traj_0 --out <dir>`) — it does the full pointcloud
reconstruction using the per-sensor cam2world transforms stored in
`obs/sensor_param/*/cam2world_gl`.

---

<a id="runs"></a>
## `runs/` — training checkpoints

```
runs/
├── smoke_pla_v1/                   500-step PLA shakedown (smoke)
├── smoke_pla_v2_grip/              100-step action_dim=8 verification
├── smoke_pla_v3_full/              **20k-step PLA (the real run)**
├── smoke_vlm_only_act_v1/          500-step baseline shakedown
└── smoke_vlm_only_act_v3_full/     **20k-step baseline (the real run)**
```

Each "full" run dir contains 10 step checkpoints (every 2000 steps:
`step_00002000.pt` through `step_00020000.pt`) plus a `latest.pt` symlink
pointing at the last one. Each ckpt is 1.16 GB and contains
`{model, optim, step, args, policy_cfg}`. `latest.pt` is what
`pla.eval`, `pla.rollout_eval`, and resume-able training use.

Final training losses (full validation round, see `README.md` §7.5.e):
- `smoke_pla_v3_full`: 12.21 → **0.0619** (L1 0.032, KL 0.003)
- `smoke_vlm_only_act_v3_full`: 12.40 → **0.0689** (L1 0.039, KL 0.003)

WandB: <https://wandb.ai/jayluvsgeography/pla> (PLA `731wnt1d`,
baseline `gjl5aijc`, both backfilled via
`scripts/backfill_wandb_from_log.py`).

---

<a id="eval_output"></a>
## `eval_output/` — JsonBenchmark eval outputs (route currently blocked)

Produced by `python -m pla.eval --checkpoint <ckpt> --benchmark_dir <bench>
--run_name <name>`. Each run dir contains:

```
eval_output/<run_name>/
├── results.json                    success_rate, Wilson 95% CI, per-arg snapshot
└── PLABenchmarkEvalConfig/
    └── <timestamp>/                one subdir per (failed) attempt
        ├── experiment_config_<timestamp>.pkl   pickled exp config
        └── running_log.log                     stdout/stderr capture
```

Both currently-present runs (`smoke_pla_v3_full/`,
`smoke_pla_v3_full_holdout/`) report `n_episodes=0, n_success=0` —
artifacts from the `JsonEvalRunner` debugging marathon. Multiple
timestamp subdirs per run because each failed iteration left its own dir.
**Safe to delete once the upstream schema PR lands and we re-run.**

The actual results that matter for the validation round are in
`rollout_output/` (below), not here.

---

<a id="rollout_output"></a>
## `rollout_output/` — custom-rollout eval outputs

Produced by `python -m pla.rollout_eval --checkpoint <ckpt> --run_name <name>
--use_proximity {true,false} --seed <int> --house_inds 11,12,...,20`. This
is the eval that worked end-to-end.

Each run dir layout:

```
rollout_output/<run_name>/
├── results.json                    aggregate + per-episode summary
└── datagen_raw/
    └── FrankaSkinPLARolloutConfig/
        └── <timestamp>/            (one subdir per launch; usually one)
            ├── experiment_config_<timestamp>.pkl      pickled exp config
            ├── running_log.log                        stdout of the datagen worker
            └── house_<N>/                             one subdir per house
                ├── trajectories_batch_1_of_1.h5       (~25 MB) all 2 trajectories in this house
                ├── episode_00000000_exo_camera_1_batch_1_of_1.mp4         (~118 KB) RGB exo
                ├── episode_00000000_exo_camera_1_depth_batch_1_of_1.mp4   (~1.4 MB) depth exo
                ├── episode_00000000_wrist_camera_batch_1_of_1.mp4         (~200 KB) RGB wrist
                ├── episode_00000000_wrist_camera_depth_batch_1_of_1.mp4   (~1.7 MB) depth wrist
                ├── episode_00000001_exo_camera_1_batch_1_of_1.mp4         …
                ├── episode_00000001_exo_camera_1_depth_batch_1_of_1.mp4   …
                ├── episode_00000001_wrist_camera_batch_1_of_1.mp4         …
                └── episode_00000001_wrist_camera_depth_batch_1_of_1.mp4   …
```

The `trajectories_batch_1_of_1.h5` files contain everything: all 29
proximity sensor readings as `(T, 4, 8, 8)` float32 arrays, per-step
qpos/qvel/actions as JSON-encoded uint8 blobs, success/fail flags per
timestep, scene+task metadata in `obs_scene`. The sibling `.mp4`s carry
RGB + depth video for the two regular cameras. The `_depth_` mp4s are
encoded depth maps (separate file per camera per episode).

Three runs present:

| Run | Eps | Success | Wall | Notes |
|-----|-----|---------|------|-------|
| `rollout_smoke_test/` | 2 | 0 | 878 s | 1 house, sanity-check launch before the real runs |
| `rollout_pla_v3_holdout/` | 18 | 0 | 4215 s (~70 min) | PLA, houses 11-20, seed 2028. *Note: 18 not 20 — one house was skipped due to a transient task-sampling error.* |
| `rollout_vlm_v3_holdout/` | 20 | 0 | 4380 s (~73 min) | Baseline (use_proximity=false), same args |

Both real runs ended with 0% success on held-out houses. See
`analysis_output/rollout_compare_v1/` for the side-by-side comparison.

### `results.json` schema (per run)

```json
{
  "run_name":             "rollout_pla_v3_holdout",
  "checkpoint":           "<path>",
  "use_proximity":        true,
  "house_inds":           [11, 12, ..., 20],
  "samples_per_house":    2,
  "seed":                 2028,
  "n_episodes":           18,
  "n_success":            0,
  "success_rate":         0.0,
  "wilson_95_ci":         [0.0, 0.1758],
  "rollout_duration_s":   4215.1,
  "per_episode": [
    {
      "house": 11, "traj": "traj_0", "n_steps": 301,
      "success": false, "fail": true,
      "task_description": "Pick up the brown pinecone keychain ..."
    },
    ...
  ]
}
```

---

<a id="analysis_output"></a>
## `analysis_output/` — PLA vs baseline comparison

Produced by `python -m pla.rollout_compare --pla_run <dir>
--baseline_run <dir> --out_dir <dir>`.

```
analysis_output/rollout_compare_v1/
├── comparison.md       human-readable report with bucket counts + per-episode table
└── comparison.json     same data, structured for further analysis
```

The headline result lives in this folder. From `comparison.md`:

| Quantity | PLA | Baseline |
|----------|-----|----------|
| Success rate | 0.0% (0/18) | 0.0% (0/20) |
| Mean approach Δ (m) | +0.020 | +0.037 |
| Mean gripper-open fraction | 93.0% | 75.0% |

Bucket counts (paired episodes):
- **A** (baseline_fail, PLA_succeed): 0
- **B** (baseline_succeed, PLA_fail): 0
- **C** (both succeed): 0
- **D** (both fail): 18

Interpretation in `README.md` §7.5.e: at 36 training trajectories with no
language conditioning, neither policy generalizes. The pipeline works
end-to-end; the data scale + language are the bottlenecks.

---

<a id="logs"></a>
## `logs/` — captured stdout

One file per training/eval/datagen launch. Used as the source for
`scripts/backfill_wandb_from_log.py` (which parses
`[step ...] loss=… l1=… kl=… N samp/s` lines) and for the rollout-eval
diagnostics.

```
logs/
├── train_pla_v3.log                    20 k-step PLA training stdout
├── train_vlm_v3.log                    20 k-step baseline training stdout
├── eval_pla_v3.log                     JsonEval first attempt (local-class pickle bug)
├── eval_pla_v3_subset.log              JsonEval 51-episode subset attempt
├── eval_pla_v3_holdout.log             JsonEval against held-out franka_skin benchmark (camera shape error)
├── eval_holdout_v1_datagen.log         Held-out dataset gen (the planner-collected 35/52 trajectories)
├── pilot_skin_medium_v1.log            Killed medium-pilot launch (the num_workers=8 OOM)
├── rollout_smoke_test.log              1-house smoke test of pla.rollout_eval
├── rollout_pla_v3_holdout.log          Full 18-episode PLA rollout
└── rollout_vlm_v3_holdout.log          Full 20-episode baseline rollout
```

---

## Other relevant directories

### `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/`

A 35-episode JsonBenchmark built from the held-out dataset via
`submodules/molmospaces/scripts/benchmarks/create_json_benchmark.py`.
Has the right robot (`franka_skin`) and the right camera names (29 SPAD
+ wrist + exo). Patched in-place to fix two builder bugs:
(1) `object_poses` filtered to drop unresolvable `place_receptacle/*`
keys; (2) `task.max_place_receptacle_pos_displacement` set to 0.15
(was 0.1, eval asserts 0.15) + `..._rot_displacement` to `radians(60)`.

Currently unusable by `pla.eval` because of the `CameraSpec` schema
issue — kept on disk for the day the schema is extended upstream.

### `assets/datagen/pick_and_place_skin_pilot_smoke_v1/`

The training set. 36 successful trajectories across iTHOR houses 1-10
collected with `FrankaSkinPickAndPlacePilotSmokeConfig` (seed=2026,
samples_per_house=4, num_workers=2). Same on-disk format as the rollout
output (1 h5 per house + 4 mp4s per episode).

### `assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/`

Same format, different scope: houses 11-20, seed 2027,
`FrankaSkinPickAndPlacePilotEvalHoldoutConfig`. The **expert** trajectories
on the held-out tasks (the planner solved 35/52 of them). This is the
source for the `assets/eval_subsets/...Holdout_v1/` benchmark above.

### `scripts/launch_medium_v1.sh`

Single-command orchestrator for the medium-v1 pipeline. Auto-discovers the
latest timestamped subdir under `assets/datagen/.../FrankaSkinPickAndPlacePilotMediumConfig/`,
refuses to run if fewer than 50 houses are present (override with
`--force-go`), trains PLA then baseline in MolmoBot-Pi0 venv, then rolls out
both in mlspaces conda env, then runs `pla.rollout_compare`. Env vars:
`DRY=1` (print only), `SKIP_TRAIN=1`, `SKIP_ROLLOUT=1`. All output files
have a shared `${TS}` suffix so re-runs don't collide.

### `scripts/backfill_wandb_from_log.py`

Standalone utility (~110 lines). Parses a `pla.train` stdout log and
replays the metrics into a fresh WandB run. Used after we did the 20k
training runs with `--use_wandb false` and needed cloud-side curves
retroactively. CLI:

```bash
python scripts/backfill_wandb_from_log.py \
    --log logs/train_pla_v3.log \
    --run_name smoke_pla_v3_full \
    --project pla \
    --tags backfill,smoke,validation_round
```

---

## What a complete pipeline run looks like (end-to-end)

For posterity, the canonical sequence:

```bash
# 1. Generate training data (one-time, 10 houses ~ 25 min on this box)
cd submodules/molmospaces && PYTHONPATH=. python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotSmokeConfig
# → assets/datagen/pick_and_place_skin_pilot_smoke_v1/.../

# 2. Verify the data (~10 sec)
python -m pla.diagnostics --root <dataset_root> --out diagnostics_output/<run_id>
# → diagnostics_output/<run_id>/summary.json (must have proximity_signal_ok=true)

# 3. (Optional) Deep dive into a single trajectory (~1 min, builds 1.79M-pt pointcloud)
python submodules/molmospaces/scripts/datagen/analyze_sample_episode.py \
    <h5> --traj traj_0 --out diagnostics_output/<run_id>/episode_houseX_trajY

# 4. Train PLA (~2 h 20 min for 20k steps)
python -m pla.train --use_proximity true  --run_name smoke_pla_v3_full \
    --data_root <dataset_root> --num_steps 20000 --num_workers 2

# 5. Train baseline (~2 h 5 min)
python -m pla.train --use_proximity false --run_name smoke_vlm_only_act_v3_full \
    --data_root <dataset_root> --num_steps 20000 --num_workers 2

# 6. Generate held-out task set (~25 min)
cd submodules/molmospaces && PYTHONPATH=. python -m molmo_spaces.data_generation.main \
    FrankaSkinPickAndPlacePilotEvalHoldoutConfig
# → assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/...

# 7. Rollout both policies on the held-out task set (~70 min each)
python -m pla.rollout_eval --checkpoint runs/smoke_pla_v3_full/latest.pt \
    --run_name rollout_pla_v3_holdout --use_proximity true \
    --seed 2028 --house_inds 11,12,13,14,15,16,17,18,19,20 --samples_per_house 2
python -m pla.rollout_eval --checkpoint runs/smoke_vlm_only_act_v3_full/latest.pt \
    --run_name rollout_vlm_v3_holdout --use_proximity false \
    --seed 2028 --house_inds 11,12,13,14,15,16,17,18,19,20 --samples_per_house 2

# 8. Compare (~5 sec)
python -m pla.rollout_compare \
    --pla_run rollout_output/rollout_pla_v3_holdout \
    --baseline_run rollout_output/rollout_vlm_v3_holdout \
    --out_dir analysis_output/rollout_compare_v1
# → analysis_output/rollout_compare_v1/comparison.md
```

Total wall time for a full cycle on this box: **~6 hours** (training
dominates). Memory ceiling: ~14 GB (with num_workers=2).
