# PLA — Training Pipeline TODO
**Deadline: 17 days remaining (CoRL 2026, hit 2026-05-11)**
**Goal: Train PLA (with proximity) vs VLM-only ACT (without proximity), eval both on a franka_skin benchmark**

---

## Status (2026-05-11)

The pipeline is end-to-end functional on the smoke dataset; the only open work
on the critical path is **eval against a franka_skin-native benchmark** (the
cached `FrankaPickandPlaceHardBench` is for `franka_droid` and is unusable as-is).

Full results write-up: `README.md` §7.5.e.

WandB project: <https://wandb.ai/jayluvsgeography/pla> (two backfilled runs:
PLA `731wnt1d`, baseline `gjl5aijc`).

---

## TODO

### 1. DataLoader ✅ DONE
- ✅ HDF5 reader: proximity, qpos, actions, language, image (decord MP4).
- ✅ Mean-pool substep dim.
- ✅ Normalize proximity by `depth_max_m=4.0` (meters, not mm — the original
  TODO text "divide by 4000.0" assumed millimeters; actual data is meters).
- ✅ Chunk size k=100; `use_proximity` flag.
- ✅ Action dim **bumped 7→8**: 7 arm joints + 1 normalized gripper command
  (the raw binary `{0.0, 255.0}` is divided by 255 for L1 compatibility
  with arm magnitudes ~rad).
- ✅ Verification report: see `pla/diagnostics.py` and the molmospaces
  `analyze_sample_episode.py` — 5+5 = 10 PNGs + `pointcloud.ply` per
  trajectory at `diagnostics_output/.../`.

### 2. Proximity Encoder ✅ DONE
- ✅ Shared MLP: `Linear(64→128) → ReLU → Linear(128→512)`.
- ✅ Output shape verified `(B, 29, 512)`.
- ✅ ~90k params; weights shared across all 29 sensors per spec.

### 3. Attach to ACT ✅ DONE
- ✅ Wraps `submodules/act/detr` without modification (reuses
  `build_transformer`, `build_backbone`, `build_encoder`, CVAE encoder).
- ✅ Proximity tokens concatenated into encoder context next to qpos +
  ResNet image tokens; baseline mode falls back to upstream context.
- ✅ Hyperparameters: chunk_size=100, hidden_dim=512, enc/dec_layers=7,
  β (kl_weight)=10.
- ✅ Param counts verified: PLA 96.37 M, baseline 96.28 M.

### 4. Training Script ✅ DONE
- ✅ L1 + 10·KL loss, Adam, lr=1e-5, batch_size=8.
- ✅ WandB logging supported (default ON). Local logs always tee'd.
- ✅ Checkpoint every `--ckpt_every` steps (default 1000; we used 2000 for
  smoke). `latest.pt` symlink kept up to date.
- ✅ Backfill helper: `scripts/backfill_wandb_from_log.py` for runs that
  were launched with `--use_wandb false`.

### 5. Run Experiments ✅ DONE on smoke; medium pilot deferred
- ✅ PLA: 20k steps, final loss **0.0619** (L1 0.032, KL 0.003). 2h 19m wall.
- ✅ Baseline: 20k steps, final loss **0.0689** (L1 0.039, KL 0.003). 2h 02m wall.
- ⏸ Medium pilot (`FrankaSkinPickAndPlacePilotMediumConfig`, 100 houses,
  num_workers=2 RAM-safe) registered but NOT launched — gated on a positive
  signal from the held-out smoke eval (see Item 6).

### 6. Eval ⚠️ BLOCKED on schema limitation

Two architectural mismatches surfaced today; the second is the actual
blocker.

- **Mismatch 1 (worked around)**: cached
  `FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark` is built for
  `franka_droid` — DROID-randomized cameras, no SPAD sensors. We built a
  franka_skin alternative.
  - ✅ `FrankaSkinPickAndPlacePilotEvalHoldoutConfig` registered (houses
        11-20, seed 2027, num_workers=2, samples_per_house=4).
  - ✅ Held-out data-gen ran (35/52 success across 10 houses,
        `assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/.../20260511_021228/`).
  - ✅ Converted to JsonBenchmark via
        `submodules/molmospaces/scripts/benchmarks/create_json_benchmark.py`:
        `assets/eval_subsets/FrankaSkinPickAndPlaceHoldout_v1/`.
        35 episodes, all with `robot_name=franka_skin` + 31 cameras
        (29 SPAD + wrist + exo).

- **Mismatch 2 (the blocker)**: even on the franka_skin benchmark,
  `pla.eval` returns 0/0 success. The `CameraSpec` schema at
  `submodules/molmospaces/molmo_spaces/evaluation/benchmark_schema.py:58-83`
  only stores `name / type / reference_body_names / camera_offset /
  lookat_offset / camera_quaternion / fov / record_depth` — no per-camera
  resolution, no `is_proximity_sensor` flag. The benchmark uses a single
  global `img_resolution = (624, 352)`. At eval time the 29 SPAD sensors
  render as 352×624 RGB instead of 8×8 depth; `pla.eval_policy`
  raises `could not broadcast (624,3) into shape (8,8)`.

  Two minor benchmark issues were patched along the way (so the next path
  doesn't re-trip them) and the patched JSON is on disk:
  - `place_receptacle/*` entries in `object_poses` that aren't in the
    same episode's `added_objects` (the builder over-reports poses);
    filter applied in-place.
  - `task.max_place_receptacle_pos_displacement` was 0.1 but eval asserts
    0.15; patched. `task.max_place_receptacle_rot_displacement` patched
    to `radians(60)`.

  **Forward paths**, in increasing rigor:
  1. **Custom rollout** (lowest cost): write a script that loads each
     `benchmark.json` episode, instantiates the env with our
     `FrankaSkinCameraSystem` (preserving 8×8 sensors), runs the policy,
     and tallies success. Reuses task specs but bypasses
     `JsonEvalRunner` entirely. ~1-2 hours.
  2. **Extend `CameraSpec` schema** (most correct): add
     `resolution: tuple[int,int] | None = None` and
     `is_proximity_sensor: bool = False` per camera, propagate through
     `camera_manager.py` setup, re-run `create_json_benchmark.py`.
     Upstream change to molmospaces. ~Half a day.
  3. **Retrain on DROID cameras** (rules out proximity): drops the
     project premise. Not viable.

  Recommend path 1 for the deadline. Path 2 is the right long-term fix
  (file an upstream PR after the CoRL submission).

### Decision rule (post-eval)

Three outcomes for the smoke eval:
- **PLA CI strictly above baseline CI**: launch the 100-house medium
  pilot, then re-run the full pipeline. Add Molmo VLM language tokens
  before the next train iteration.
- **CIs overlap heavily**: either need more episodes (200 vs 40) to
  resolve, or proximity provides no signal at this scale. Inspect L1
  gap vs success-rate gap.
- **Both at ~0% success**: bug or task is too hard for 36 trajectories
  of training. Debug before any more collection.

---

## Open code-level items (not on critical path)

- **Language conditioning** via Molmo VLM tokens (TODO §3 spec). Dataset
  already returns `task_description`; the policy doesn't consume it.
  Defer until the smoke eval gives a directional answer.
- **Eval CI scale**: 200 episodes per checkpoint with the canonical
  `JsonBenchmarkEvalConfig` once the held-out benchmark is large enough.
- **Real-robot transfer**: external blocker; not part of this CoRL
  submission.
