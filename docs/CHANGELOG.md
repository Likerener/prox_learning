# Changelog

User-facing summary of changes. Newest at the top.

For the deep "what was implemented and why" log see
[IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md). For verification of
specific claims see [SANITY_CHECKS.md](SANITY_CHECKS.md).

---

## 2026-05-04 (Day 3)

### Added
- **Annotated sensor reference images (`pla/viz/sensor_overlay.py`,
  `assets/reference_images/annotated/`)**: the existing
  `franka_skin_*.png` renders show the FR3 mesh but the 4 mm sensor
  sites are invisible at the render resolution. New tool patches the
  on-disk MJCF mesh paths, injects fixed orbit cameras, then projects
  all 29 sensor body positions into image pixels with link-colored
  labelled disks (red=link2, orange=link3, green=link5, blue=link6) and
  back-face culling. Outputs: 4 PNG views (az = -180 / -90 / 0 / +90)
  + 4-up legend grid + `sensor_layout_table.csv` (mjcf_index →
  link/idx/world XYZ). Re-runnable: `MUJOCO_GL=egl python -m
  pla.viz.sensor_overlay`.
- **Pure-NumPy pointcloud projection core (`pla/viz/pointcloud_core.py`)**:
  factored the inverse-pinhole math out of `pla/viz/pointcloud.py` into
  a deterministic, side-effect-free module supporting two frames —
  `frame="camera"` (standard OpenGL, +x right / +y up / -z forward) for
  use with `data.cam_xpos / data.cam_xmat`, and `frame="depth_axis"`
  for the legacy body-frame convention. Added `taxel_directions`,
  `unproject_taxels`, `transform_points`, `reconstruct_world_pts`,
  `fit_plane`. Back-compat alias `unproject_taxels_to_body` retained.
- **6-test pointcloud reconstruction suite (`pla/viz/pointcloud_tests.py`)**:
  rigorous validation of the ToF → world-point pipeline against
  mujoco-rendered ground truth. T1 intrinsics; T2 synthetic flat wall;
  T3 single-sensor mujoco wall (target rms < 1 mm); T4 29-sensor
  coverage with no points reconstructed behind any sensor; T5 same
  wall reconstructed from two arm poses; T6 legacy convention regression
  report. Generates 4 diagnostic PNGs + `results.json` in
  `reports/checks/pointcloud_tests/`. **All 6 tests pass**: T3 reaches
  0.171 mm rms, T5 reaches 0.17 / 0.30 mm across two poses.

### Verified
- 29 sensors found at home pose (link2 ×7, link3 ×8, link5 ×6,
  link6 ×8) — matches the `MJCF_ORDER` constant in `pla.viz.pointcloud`.
- Sensor overlay projection uses fixed orbit cameras (mujoco computes
  `cam_xpos / cam_xmat`), so projection always matches the renderer
  exactly — no hand-rolled azimuth/elevation math.
- Camera-frame reconstruction (`frame="camera"` + `cam_xpos / cam_xmat`)
  recovers a known wall plane to **0.171 mm rms / 0.177 mm worst-case**
  at 0.18 m wall distance (mujoco depth pipeline + our intrinsics).

### Found (regression report)
- The legacy formula in `pla/viz/pointcloud.py`
  `(u·half·d, -v·half·d, d) × R_body` has the wrong y-sign relative to
  the body frame defined by the MJCF camera quat `(0,1,0,0)`. T6
  measures **88 mm rms / 140 mm worst case** vs camera-frame ground
  truth at 0.18 m wall distance. The corrected body-frame formula is
  `(u·half·d, +v·half·d, d) × R_body`, equivalent to using
  `pla.viz.pointcloud_core.reconstruct_world_pts(..., frame="camera")`
  with `data.cam_xpos / data.cam_xmat`. Fix lives in
  `pointcloud_core`; `pla/viz/pointcloud.py` itself still calls the
  back-compat alias and inherits the bug — flagged for fix.

---

## 2026-05-03 (Day 2)

### Added
- **Pipeline-validation demo (`pla/sim/demo_pnp.py`,
  `reports/demo_pnp/`)**: self-contained kinematic FR3 pick-and-place
  trajectory rendered through the real PLA collection + audit pipeline.
  Uses the cached `scene_fr3.xml` molmo-spaces resource bundle (no
  procthor required). 3 episodes × T=230 → 3 HDF5 shards (105 MB), 3
  h264 MP4 videos, 15 keyframe PNGs, 7 audit-plot PNGs, deep verify
  PASS. Runs in ~30 s on CPU. **Not real procthor house-1 data** —
  documented explicitly in `reports/demo_pnp/README.md`. Demonstrates
  the schema, audit visualizer, and verify pipeline all work
  end-to-end.

- **Visual audit suite (`pla/viz/dataset_audit.py`)**: 7 plots for
  eyeballing data quality before launching a long run. ToF heatmap
  montage at far/mid/late/peak-contact frames; per-sensor depth
  histograms (32-panel grid); per-sensor coverage diagnostics (min
  reading + std); per-episode traces of depth-min and action-norm vs
  time; RGB sanity strips; episode-length histogram; per-joint action
  distribution + summary heatmap. Generates PNG + PDF + INDEX.md per
  run. Hooked into `scripts/preflight.sh` step [5/5] so the pilot
  produces these automatically. Verified on 12 synthetic episodes —
  all 7 plots render correctly.
- **Pre-flight + streaming validator (`pla/data/preflight.py`,
  `pla/data/sentinel.py`)**: protect the long collection run from wasted
  compute. `preflight` runs single-shot checks (MJCF loads, sensor
  cameras count matches config, ToF render produces sane depths, env
  imports + reset+step, disk space, optional 1-episode round-trip).
  `sentinel` watches the output dir during collection, audits each new
  shard, writes `SENTINEL_ABORT` marker after a configurable bad streak
  or low prox-informative coverage; the collector polls the marker
  between episodes and exits cleanly.
- **Deep `verify` (extended `pla/data/verify.py`)**: post-collection
  audit beyond schema — per-sensor stuck/dead detection, frozen-RGB
  signature, action |max|, episode-length distribution, success-rate
  warning band. Required-vs-warn distinction; `--strict` exits non-zero
  on any required failure.
- **Orchestration scripts**: new `scripts/preflight.sh` runs the full
  preflight → 50-traj pilot → sentinel → deep-verify gate sequence.
  `scripts/collect_data.sh` updated to launch sentinel + collector in
  parallel tmux sessions, with the abort marker contract.

### Verified
- Preflight correctly fails on missing MJCF and gracefully skips on
  missing MolmoSpaces env (status `SKIP`, not `FAIL`).
- Sentinel detects each new shard, writes heartbeats every N eps, exits
  on target reached.
- Sentinel abort path: 3 consecutive bad shards triggers marker write
  (verified on synthetic broken h5 files).
- Collector honours pre-existing abort marker — exits at iteration 0
  with 0 files written.
- Deep verify passes on synthetic-healthy 12-episode dataset, fails on
  synthetic-pathological dataset (planted: stuck sensor, frozen RGB,
  oversized action). All 3 defects caught; `--strict` exits 1.
- All 5 modified modules AST-parse and import cleanly.

## 2026-05-02 (Day 1)

### Added
- **Paper draft (`paper/`)**: `main.tex` (full CoRL-style LaTeX with TikZ
  architecture figure + bibliography), `references.bib` (15 entries),
  `Makefile` (pdflatex + bibtex pipeline), `main.md` (markdown mirror),
  `paper/README.md`. Methods + protocol locked; results table is
  pre-registered placeholder pending Day-14 numbers.
- **Paper figures (`paper/figures/`)**: 9 figures total — TikZ
  versions inline in `main.tex` (LaTeX side) plus matplotlib PDF + PNG
  for every figure (PNG embedded in `main.md` for viewers that don't
  render PDF inline). Schematics: architecture overview, sensor skin
  layout, ToF rendering pipeline, statistical pairing protocol. Results
  figures: per-config encoder grad norms, 50-step synthetic loss curve,
  bootstrap CI illustration, paired-bootstrap null distribution,
  sample-size power curve. All matplotlib output is generated by
  `make_figures.py` from real Day-1 data — no fabricated empirical
  claims.
- **Full model layer**: ProximityEncoder (shared MLP), HandcraftedToFEncoder,
  Conv2DToFEncoder, ModalityFusion (concat + cross_attn), full ACTDecoder
  (Zhao 2023), FrozenMolmo2 + DummyVLBackbone, unified PLA class with
  `vlm_only` flag and `sensor_mask`.
- **Data layer**: `pla.data.dataset.PLADataset` (sliding-window),
  `pla.data.normalize` (per-channel training-only stats),
  `pla.data.verify` (Day-2 sanity), expanded `pla.data.collect`.
- **Training**: unified `pla.train.train` for every config; W&B optional
  with stdout JSONL fallback; per-step encoder grad-norm logging.
- **Evaluation**: `pla.eval.run_eval` (per-method/per-task runner +
  results-table aggregator), bootstrap stats helpers, sensor-importance
  sweep, failure-mode categorizer.
- **Sim**: `ToFSensorArray` class (cached renderer, MJCF-order sensor list).
- **Skin pipeline scripts**: `scripts/build_skin_mjcf.py` (Blender JSON →
  MJCF camera bodies), `scripts/verify_skin.py` (empty-scene self-hit).
- **Sanity checks**: `pla.checks.forward_pass` (5 variants),
  `pla.checks.grad_norm` (CLI driver).
- **Configs**: 4 ablation YAMLs (wrist_only, handcrafted, conv2d,
  cross_attn). Rewrote `pla.yaml` and `act_baseline.yaml` to flat schema —
  the only difference between them is `vlm_only`.
- **Per-folder READMEs**: `pla/`, `pla/data/`, `pla/sim/`, `pla/models/`,
  `pla/train/`, `pla/eval/`, `pla/checks/`, `pla/viz/`, `pla/ablations/`,
  `assets/`, `configs/`, `scripts/`, `reports/`.
- **Tracking docs**: `docs/STATUS.md`, `docs/IMPLEMENTATION_LOG.md`,
  `docs/SANITY_CHECKS.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN_DECISIONS.md`,
  `docs/FILE_INVENTORY.md`, `docs/STATISTICAL_PROTOCOL.md`,
  `docs/CHANGELOG.md`.

### Verified
- 49/49 Python files parse cleanly.
- 29/29 modules smoke-import.
- Forward+backward+inference passes for: PLA, vlm_only, handcrafted,
  conv2d, cross_attn variants.
- Grad-norm > 1e-2 across all 4 ablation configs; baseline correctly skips.
- Synthetic data pipeline end-to-end: collect → verify → normalize →
  PLADataset → train (3 steps) — loss decreases, encoder grads non-zero.
- Bootstrap CI + paired bootstrap p-value match independent χ² check.
- Wrist-only mask zeros indices 8-31 and preserves 0-7 element-wise.

### Removed / replaced
- Stub `NotImplementedError` bodies in: `pla.data.collect`, `pla.eval.run_eval`,
  `pla.eval.sensor_importance`, `pla.eval.failure_analysis`.
- Legacy `train_pla.py` / `train_baseline.py` are now thin wrappers
  around `pla.train.train`; logic moved.

### Not yet done (deferred to Day 2+)
- Real Molmo2 forward pass (HF download).
- Built MJCF skin from Day-1-PM Blender redesign.
- 1000-trajectory near-contact dataset.
- Any real training run with real data.
- Paper figures.

---

## 2026-04-21 .. 2026-05-01 (Day 0 — pre-project setup)

### Added
- Initial repo structure with `pla/` package skeleton.
- Legacy `docs/PROJECT.md`, `docs/TIMELINE.md`, `docs/SKIN_PIPELINE.md`,
  `docs/CVAE.md`, `docs/DATASET.md`.
- Submodules: MolmoBot, molmospaces, ACT.
- URDF + MJCF for the legacy 10-traj sanity dataset
  (`skin_pick_fixed_v1`).
