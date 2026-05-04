# `reports/demo_pnp/` — pipeline-validation pick-and-place demo

**Read this first.** This dataset is **NOT real procthor house-1 data**.
It is a self-contained kinematic FR3 pick-and-place trajectory rendered
through the actual PLA collection + audit pipeline. The point is to
validate the stack end-to-end and produce real images and videos you
can look at.

## Why a pipeline demo and not real procthor?

Real MolmoSpaces house-1 collection in this environment requires:

  - `pip install -e submodules/molmospaces` (~3-5 GB: JAX, MuJoCo-MJX,
    pydantic, procthor's `prior` package, beaker-py, decord, …)
  - Running `submodules/molmospaces/scripts/datagen/fetch_assets.py` to
    pull procthor + objaverse house assets (potentially many GB)
  - The `FrankaPickAndPlaceDataGenConfig` data-gen pipeline configured
    against scene index 1

None of those dependencies are currently installed. Rather than launch a
~10-minute install + multi-GB asset download without your green light,
we built a self-contained MuJoCo-only demo that exercises the **same**
collection pipeline (`pla.data.collect._write_episode_h5`), the **same**
HDF5 schema, and the **same** audit visualizer (`pla.viz.dataset_audit`)
that the real run will use.

## What's faithful and what's faked

| component                | this demo                                            | real molmospaces house-1 |
|--------------------------|------------------------------------------------------|---------------------------|
| robot model              | **real FR3** (`scene_fr3.xml` from molmo-spaces cache) | same                      |
| scene                    | wooden table + blue cube + red cylinder + green target | procthor-objaverse house  |
| rendering                | real `mujoco.Renderer` (224×224 RGB)                | same                      |
| trajectory               | hand-tuned 10-waypoint kinematic interpolation       | TAMP planner (curobo)     |
| physics                  | none — qpos set directly                             | full mujoco-mjx physics   |
| HDF5 schema              | **identical** to production schema                   | same                      |
| ToF stream               | **synthesised** — distance-correlated 8×8 grids per sensor | rendered from skin cameras |
| `n_sensors`              | 8 (proxy for EE-only)                                | 32 (full skin)            |
| episode count            | 3 demo episodes                                      | 1000+                     |

The ToF stream is intentionally synthesised from the live hand→object
distance so the audit traces show structurally-correct temporal
patterns (depth dipping when the hand approaches the cube). The real
run renders depth from physical sensor cameras mounted on the skin.

## What's in this directory

| file                                    | what it is                                         |
|-----------------------------------------|-----------------------------------------------------|
| `episode_000.mp4` … `episode_002.mp4`   | 3 watchable PnP rollouts at 24 fps, 9.6 s each, h264 |
| `episode_NNN_frame{0..4}_t*.png`        | 5 keyframes per episode (start, 1/4, 1/2, 3/4, end)  |
| `raw/episode_000000.h5` … `_000002.h5`  | 3 HDF5 shards in the canonical PLA schema           |
| `audit/01_tof_montage.png` … `07_*.png` | 7 audit plots from `pla.viz.dataset_audit`          |
| `audit/INDEX.md`                        | the audit's own browse-in-order index                |
| `verify.json`                           | the full deep-verify report (PASS)                  |

## Verification status

Deep verify (`pla.data.verify`) PASS:

```
Episodes processed:       3
Schema OK:                3/3
NaN episodes:             0/3
Successful:               3/3 (rate 100.0%)
Proximity-informative:    3/3 (100.0%)
Frac steps with <200 mm reading: 56.2%
Episode length (T):       T=230 (constant; kinematic = same length)
Action |max| global:      0.027 (sane bound 1.0)
Dead sensors:             0/8
Stuck sensors:            0/3 episodes
Frozen RGB:               0/3 episodes
Required: prox_informative >= 30%, NaN/Inf == 0, schema_ok == 100%, |action| <= 1.0: PASS
```

## How to regenerate

```bash
PYTHONPATH=. python -m pla.sim.demo_pnp --out reports/demo_pnp --n-episodes 3
```

The script is deterministic (seeded). Episode 0 produces an identical
HDF5 across runs.

## Next step — real procthor house-1 collection

When you give the green light, the path to the real run is:

```bash
# 1. Install MolmoSpaces + dependencies (~5-10 min, ~5 GB).
pip install -e submodules/molmospaces

# 2. Fetch procthor-objaverse assets (multi-GB download).
python submodules/molmospaces/scripts/datagen/fetch_assets.py

# 3. Run preflight against the real env.
bash scripts/preflight.sh near_contact 1000 --full --strict

# 4. If preflight passes — launch the long run.
bash scripts/collect_data.sh near_contact 1000
```
