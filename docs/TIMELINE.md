# 26-Day Execution Plan (May 2 – May 28, 2026)

## Week 1 (May 2–8): Infrastructure + Data + First Training

### Day 1 (May 2) — Blender skin + failure cases
- Morning: Learn Blender basics (≤2 hours).
- Afternoon: Build new skin with 14–16 sensors on link6/gripper, export positions.
- Evening: Run failure case analysis on existing MolmoBot-Pi0 eval, record videos.

### Day 2 (May 3) — MJCF integration + sanity check
- Morning: Patch FR3 MJCF with new sensor positions (`pla/sim/patch_mjcf.py`).
- Afternoon: Sanity check (`pla/checks/depth_reconstruction.py`,
  `pla/checks/replay_mjcf.py`). Targets: >30% proximity-informative
  trajectories, 0 NaN, schema validates (`pla/data/schema.validate`).
- Evening: Design near-contact task — obstacle 5–8 cm from arm path
  (`configs/data/near_contact.yaml`).

### Day 3 (May 4) — Data collection starts
- Morning: Fix any integration issues from Day 2.
  Launch `bash scripts/collect_data.sh near_contact 1000` in tmux.
- Afternoon: Implement VLM-only ACT (one flag flipped vs PLA). Start training.
- Evening: Implement `ProximityEncoder` and fusion layer in `pla/models/pla.py`.

### Day 4 (May 5) — Baselines
- Morning: Check overnight collection. Run prop-only MLP baseline.
- Afternoon: Verify VLM-only ACT loss curves are converging.
- Evening: Test full PLA forward pass with dummy data — verify tensor shapes.

### Day 5 (May 6) — VLM-only ACT eval
- Morning: VLM-only ACT trained. Eval 50 episodes on near-contact +
  standard PnP. **This is the most important number in the paper.**
- Afternoon: Start PLA training; continue VLM-only eval on more tasks.

### Day 6 (May 7) — PLA training running
- Morning: Verify proximity-encoder grad-norm > 1e-8
  (`pla.checks.grad_norm.assert_learning`). If zero, debug before proceeding.
- Afternoon: W&B analysis — verify all three loss terms behave correctly.
- Evening: LR sweep ∈ {1e-5, 3e-5} if time allows.

### Day 7 (May 8) — First PLA numbers
- Morning: Eval PLA checkpoint vs VLM-only ACT (50 episodes near-contact).
  - delta > 10 pp on near-contact → strong result, proceed.
  - delta < 5 pp → check proximity-informative %, encoder grads, task design.
- Evening: 1:1 prep for Alessandro.

## Week 2 (May 9–15): Full Evaluation + Ablations

### Day 8-9 (May 9–10) — Full eval + ablation training
Launch all ablations overnight (`bash scripts/run_ablations.sh`).
Full evaluation: 100 episodes per condition on all 4 tasks.

### Day 10-11 (May 11–12) — Ablation results + sensor importance
- Collect all ablation numbers with bootstrap CI.
- `python -m pla.eval.sensor_importance` — mask each sensor, measure drop.
- Plot heatmap on FR3 body via `pla.viz.heatmap.sensor_importance_heatmap`.

### Day 12 (May 13) — Figures start
1. **System overview** with tensor shapes.
2. **ToF heatmap sequence** — far / mid / near / pre-grasp.
3. **Results table** with bootstrap CI + p-values.
4. **Sensor importance heatmap** on FR3 body.

### Day 13-14 (May 14–15) — Figures finalized + results locked
- All figures: ≥8pt fonts, colorblind-safe, self-contained captions.
- Results locked. No more experiments after Day 14 unless something is broken.

## Week 3 (May 16–22): Write the Paper

Order: §3 System → §4 Method → §5 Experiments → §6 Discussion → §2 Related →
§1 Intro → Abstract.

- Day 15 (May 16): §3 System + §4 Method
- Day 16 (May 17): §5 Experiments
- Day 17 (May 18): §6 Discussion + Limitations
- Day 18 (May 19): §2 Related Work
- Day 19 (May 20): §1 Introduction
- Day 20 (May 21): Abstract + read-through
- Day 21 (May 22): Hostile reviewer pass + fixes

## Week 4 (May 23–28): Revise, Format, Submit

- Day 22-23 (May 23–24): Send draft to Alessandro. Fix everything within 24 hours.
- Day 24 (May 25): CoRL formatting pass (PMLR template, 8-page limit, double-blind).
- Day 25 (May 26): Supplementary (per-task tables, hyperparams, sensor diagram, videos).
- Day 26 (May 27): Final checks (citations, no [?], page count, figure fonts).
- **Day 27 (May 28): SUBMIT** before midnight AoE.

## Risk Register

| Risk                              | Probability | Impact | Mitigation                                                       |
|-----------------------------------|-------------|--------|------------------------------------------------------------------|
| PLA delta < 5 pp                  | Medium      | Fatal  | Design near-contact task carefully; verify proximity coverage.   |
| Proximity encoder grad = 0        | Low         | Fatal  | Daily grad-norm checks during training.                          |
| Dataset not proximity-informative | Medium      | High   | Run schema check before full collection.                         |
| Running out of time               | Medium      | High   | Writing starts Day 14 regardless of ablation status.             |
| Alessandro requests major changes | Low         | High   | Send draft by Day 22 — 6 days buffer.                            |
| VRAM OOM on PLA training          | Low         | Medium | Pi0 is 7 GB. PLA adds ~2 MB. Monitor with `nvidia-smi`.          |
