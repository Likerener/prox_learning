"""Trajectory collection harness.

Drives MolmoBot-Engine TAMP planning in MolmoSpaces procthor-objaverse scenes
to collect 1000+ expert trajectories with whole-body ToF, RGB, and qpos.

Critical requirement (PROJECT.md §3.3): at least 30% of trajectories must
include ToF readings below 200 mm. Standard PnP may not hit this — use the
near-contact task with a fixed obstacle 5–8 cm from the expert path.

HDF5 schema per episode::

    episode_N/
      observations/
        tof:  [T, N_sensors, 8, 8]  float32  mm, clipped [20, 4000]
        rgb:  [T, 3, 224, 224]       uint8
        qpos: [T, 7]                 float32
      actions:  [T, 7]              float32  joint delta
      metadata: {task, scene_id, success, seed, policy_phase}

Run::

    python -m pla.data.collect --config configs/data/near_contact.yaml --n-traj 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect expert trajectories")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/"))
    p.add_argument("--n-traj", type=int, default=1000)
    p.add_argument("--n-envs", type=int, default=10, help="distinct procthor scenes")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Hook this up to submodules/MolmoBot TAMP. See docs/TIMELINE.md Day 3."
    )


if __name__ == "__main__":
    main()
