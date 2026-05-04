# `pla/data/` — collection, schema, normalization, DataLoader

## Purpose

Turn raw MolmoSpaces rollouts into a dataset that PyTorch can train on. The
contract is fixed: we always write **HDF5 shards, one episode per file**,
with the schema below. Every downstream module (training, eval, viz) reads
through `PLADataset` so the schema is checked exactly once, at load time.

## HDF5 schema (per episode)

```
episode_0/
  observations/
    tof:    [T, N_sensors, 8, 8]   float32   millimetres, clipped [20, 4000]
    rgb:    [T, 3, 224, 224]       uint8     standard image bytes
    qpos:   [T, 7]                 float32   FR3 joint positions (rad)
  actions:  [T, 7]                 float32   joint deltas
  policy_phase: [T]                int32     (optional) which TAMP stage
  attrs:
    success:    bool
    n_sensors:  int
    task:       str (optional)
    language:   str (optional, used by DataLoader)
    seed:       int
```

Validators live in `schema.py` (`validate`, `proximity_informative_fraction`).

## Files

| file                | what it does                                                   |
|---------------------|----------------------------------------------------------------|
| `collect.py`        | drives MolmoSpaces TAMP rollouts; writes HDF5 shards            |
| `schema.py`         | structural validator + proximity-informative coverage           |
| `preflight.py`      | **single-shot pre-flight checks** (env, MJCF, sensor render, disk, 1-traj round-trip) |
| `sentinel.py`       | **streaming validator** during collection (per-shard audit, abort marker) |
| `verify.py`         | post-collection deep audit (per-sensor, frozen-frame, action stats, length dist) |
| `normalize.py`      | per-channel mean/std; writes `stats.json`                       |
| `dataset.py`        | `PLADataset` — sliding-window DataLoader (chunk_size=100)        |
| `cvae_dataset.py`   | legacy: skin-CVAE pretrain dataset (kept for the encoder proof) |
| `stats.py`          | dataset-level summary statistics + plot helpers                  |

## Order of operations (TIMELINE.md Days 3-5)

A long collection run is a hours-long compute investment. Follow the
gate sequence below — every step must pass before the next runs:

```bash
# 0. PRE-FLIGHT (10-15 min, MUST pass before launching the long run).
#    This runs: MJCF check, ToF render check, env import + reset+step,
#    1-episode round-trip via the real env, disk-space check, then a
#    50-episode pilot collection watched by the sentinel, then the
#    deep verifier on the pilot, THEN the visual audit (7 plots you
#    must look at before launching the full run).
bash scripts/preflight.sh near_contact 1000 --full --strict
# Now eyeball reports/checks/preflight_<task>/audit/INDEX.md before going further.

# 1. Collect (Day 3): launches MolmoSpaces TAMP + sentinel side-by-side.
#    Sentinel writes data/raw/near_contact/SENTINEL_ABORT if the run goes
#    bad; the collector polls and stops between episodes.
bash scripts/collect_data.sh near_contact 1000

# 2. Deep verify (Day 3 evening, again at end of run):
python -m pla.data.verify \
    --data-dir data/raw/near_contact \
    --report reports/checks/verify_near_contact.json \
    --strict

# 3. Normalize (Day 4): compute training-only stats. THIS MUST COME FIRST.
python -m pla.data.normalize --data-dir data/raw/near_contact --out stats.json

# 4. Smoke-test the loader (Day 4):
python -c "
from pla.data import PLADataset
ds = PLADataset('data/raw/near_contact', 'stats.json', chunk_size=100, split='train')
print(len(ds), ds[0]['tof'].shape, ds[0]['actions'].shape)
"
```

## Gate sequence — what passes/fails the run

```
                pre-flight (preflight.py)
                        │
                ┌───────┴────────┐
                │  PASS          │  FAIL
                ▼                ▼
          50-traj pilot      do not launch
                │
                ├── sentinel watches each shard
                │      ├── streak of 5 bad ⇒ ABORT marker
                │      └── prox-informative < 30% after 50 ⇒ ABORT
                │
        post-pilot deep verify (--strict)
                │
                ├── PASS ⇒ launch full 1000-traj run
                └── FAIL ⇒ debug; do not launch
                        │
                ┌───────┴────────┐
                ▼                ▼
        full 1000-traj      sentinel side-by-side
                │                │
                └────────────────┘
                        │
                end-of-run deep verify (--strict)
                        │
                normalize → train
```

## Sentinel + abort marker contract

* `pla.data.sentinel` watches `data_dir/*.h5`. For each new shard it
  audits structure, NaN/Inf, depth range, action range, episode length,
  per-sensor stuck signal, frozen-RGB count, prox-informative status.
* On a streak of `--bad-streak` consecutive bad shards, OR if the
  rolling prox-informative fraction drops below the threshold after
  ≥ 50 episodes, it writes `data_dir/SENTINEL_ABORT` with a JSON
  reason.
* `pla.data.collect` calls `collector_should_stop(out_dir)` between
  episodes; finding the marker exits cleanly. No corrupt half-written
  shard.

## What `verify.py` checks beyond schema

| metric                         | required? | failure mode it catches                          |
|--------------------------------|-----------|--------------------------------------------------|
| schema_ok / total              | YES       | DataLoader will throw at training time           |
| NaN / Inf in tof, qpos, acts   | YES       | gradient explosion, silent loss spikes           |
| prox-informative ≥ 30%         | YES       | proximity stream has no signal to learn from     |
| `|action|_max ≤ 1.0`           | YES       | runaway TAMP planner; physically unsafe          |
| success rate ≥ 30% (configurable) | warn   | task too hard / env broken                       |
| per-sensor std ≥ 0.5 mm        | warn      | sensor stuck (build pipeline mis-orientation)    |
| per-sensor min < 1500 mm        | warn      | dead sensor (never sees scene; geometry bug)     |
| no frozen-RGB streaks          | warn      | camera stalled; renderer / threading issue       |
| episode length within band     | warn      | TAMP planner timing out / env truncating early   |

## Why these design choices

* **One episode per file.** Lets us shard and resume collection without
  rewriting; lets the verifier surface bad shards individually; lets the
  DataLoader build its sliding-window index in parallel.
* **mm on disk, normalized in the model.** Storing raw mm makes verification
  trivial (`tof < 200` is a literal "below 20 cm"), preserves the physical
  meaning across machines, and only costs one extra subtract+divide on
  load. Networks learn faster on standardized inputs (PROJECT.md §3.2).
* **Training-only stats.** Computing stats on the full set leaks val/test
  info into the input distribution; reported numbers would be optimistic.
  The val split is held out using the same `(seed, val_frac)` tuple in both
  `normalize.py` and `dataset.py` so the splits agree by construction.
* **Sliding window with `chunk_size=100`.** Matches Zhao et al. 2023; the
  CVAE encoder needs the whole future chunk at training, the decoder needs
  the same at inference. The window starts at `t=K-1` where `K=2` so we
  always have two RGB frames of history (PROJECT.md §3.4 uses K=2).
* **`proximity_informative_fraction >= 30%`.** Without near-contact frames
  the proximity stream has no signal during PLA training. The verify step
  fails loud rather than letting us train a model on uninformative data.
  This is the most common failure mode for the headline experiment.

## Sanity-check checklist (run before training each day)

- [ ] `pla.data.verify` exits 0 (schema OK, NaN-free, ≥30% prox-informative)
- [ ] `stats.json` exists and `n_sensors` matches the model config
- [ ] `len(PLADataset('train'))` > 10 * batch_size (else val/train split is broken)
- [ ] `(ds[0]['tof'].mean(), ds[0]['tof'].std())` are O(1) — confirms normalization
- [ ] First batch loads in < 1 s on warm cache (else IO is the bottleneck)
