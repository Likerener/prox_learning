"""Build a combined ACT dataset = house_1 (356 ep) + house_3 (132 ep).

Uses symlinks for the episode HDF5s (saves disk and keeps the originals as the
source of truth) and merges the two `prox_mapping.json` files into one with
remapped episode indices.

Output:
  act_style_data/mug_houses_1_3_random_everything/
    episode_0.hdf5 ... episode_487.hdf5         (symlinks)
    prox_mapping.json                            (combined, 488 entries)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path("/home/jaydv/code/prox_learning")
H1 = REPO / "act_style_data" / "mug_house1_random_everything"
H3 = REPO / "act_style_data" / "mug_house3_random_everything"
OUT = REPO / "act_style_data" / "mug_houses_1_3_random_everything"


def link_episodes(src_dir: Path, dst_dir: Path, start_idx: int) -> int:
    """Symlink episode_<i>.hdf5 from src into dst at indices [start_idx, ...]."""
    src_eps = sorted(src_dir.glob("episode_*.hdf5"),
                     key=lambda p: int(p.stem.split("_")[1]))
    for offset, src in enumerate(src_eps):
        dst = dst_dir / f"episode_{start_idx + offset}.hdf5"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    return len(src_eps)


def merge_mappings(map_h1: dict, map_h3: dict, h1_count: int) -> dict:
    """Combine two prox_mapping.jsons; h3 episodes get +h1_count index shift."""
    assert map_h1["n_sensors"] == map_h3["n_sensors"], "sensor count mismatch"
    assert map_h1["sensor_names"] == map_h3["sensor_names"], "sensor name mismatch"
    eps = {}
    for k, v in map_h1["episodes"].items():
        eps[k] = v  # untouched
    for k, v in map_h3["episodes"].items():
        new_k = str(int(k) + h1_count)
        eps[new_k] = v
    return {
        "act_dataset_dir": str(OUT),
        "source_glob":     [map_h1["source_glob"], map_h3["source_glob"]],
        "sensor_names":    map_h1["sensor_names"],
        "n_sensors":       map_h1["n_sensors"],
        "qpos_atol":       map_h1["qpos_atol"],
        "episodes":        eps,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    n_h1 = link_episodes(H1, OUT, start_idx=0)
    print(f"[combine] linked {n_h1} house_1 episodes -> indices 0..{n_h1 - 1}")

    n_h3 = link_episodes(H3, OUT, start_idx=n_h1)
    print(f"[combine] linked {n_h3} house_3 episodes -> indices {n_h1}..{n_h1 + n_h3 - 1}")

    map_h1 = json.loads((H1 / "prox_mapping.json").read_text())
    map_h3 = json.loads((H3 / "prox_mapping.json").read_text())
    combined = merge_mappings(map_h1, map_h3, h1_count=n_h1)
    (OUT / "prox_mapping.json").write_text(json.dumps(combined, indent=2))

    print(f"[combine] wrote {OUT / 'prox_mapping.json'}  "
          f"({len(combined['episodes'])} episodes)")
    print(f"[combine] DONE  total episodes = {n_h1 + n_h3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
