# prox_learning — PLA: Peripersonal Language-Action

Code, configs, and assets for the CoRL 2026 submission **PLA: Peripersonal
Language-Action Policies via Whole-Body Time-of-Flight Proximity Sensing**.

- **Researcher:** Jay Vakil — HIRO Lab, CU Boulder
- **Advisor:** Alessandro Roncone
- **Submission deadline:** 2026-05-28

The full project description and 26-day plan live in
[`docs/PROJECT.md`](docs/PROJECT.md) and [`docs/TIMELINE.md`](docs/TIMELINE.md).

## Repository layout

```
pla/                      # importable Python package — all code lives here
├── data/                 # collection harness, HDF5 schema, dataset stats
├── sim/                  # URDF→MJCF builder, sensor-orientation fixes, ToF rendering
├── models/               # PLA, baselines, ProximityEncoder, CVAE
├── train/                # training entry points + losses
├── eval/                 # eval runner, tasks, bootstrap CIs, failure analysis
├── checks/               # pre-training sanity checks (depth, replay, grad-norm)
└── viz/                  # composite videos, point clouds, paper figures

assets/                   # MJCF, URDF, reference renders
configs/                  # data/, train/, eval/ YAML configs
scripts/                  # shell wrappers (collect, train, eval, ablations)
docs/                     # PROJECT.md, TIMELINE.md, DATASET.md, etc.
reports/                  # paper-bound artifacts (figures, tables, logs)
paper/                    # LaTeX submission
runs/                     # gitignored — training outputs
data/                     # gitignored — HDF5 trajectory shards
submodules/               # MolmoBot, molmospaces, ACT (git submodules)
```

## Quick start

```bash
# Install the package and submodules.
git submodule update --init --recursive
pip install -e .

# 1. Sanity-check the MJCF + sensor mounts.
python -m pla.checks.depth_reconstruction
python -m pla.checks.replay_mjcf

# 2. Collect trajectories (PROJECT.md §3.3).
bash scripts/collect_data.sh near_contact 1000

# 3. Inspect dataset stats and proximity coverage.
python -m pla.data.stats
python -c "from pathlib import Path; from pla.data.schema import proximity_informative_fraction; \
           print(proximity_informative_fraction(Path('data/raw/near_contact').rglob('*.h5')))"

# 4. Train baselines first, then PLA.
bash scripts/train_baselines.sh
bash scripts/train_pla.sh

# 5. Evaluate on all 4 tasks.
bash scripts/eval_all.sh
```

## The headline experiment

The paper's primary scientific claim (PROJECT.md §8) is tested by the delta
between PLA and the **VLM-only ACT** baseline on the **near-contact** task.
That comparison is one config flag (`use_proximity: false`) different from PLA
— see `configs/train/act_baseline.yaml`. Without this baseline run, the
proximity-sensing claim cannot be substantiated.

## Submodules

| Path                       | Purpose                                                 |
|----------------------------|---------------------------------------------------------|
| `submodules/MolmoBot`      | Vision-language-action backbone + TAMP planning         |
| `submodules/molmospaces`   | procthor-objaverse simulation + FrankaPickandPlace eval |
| `submodules/act`           | Action Chunking Transformer decoder                     |
