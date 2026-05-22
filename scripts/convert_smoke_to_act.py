"""Convert the multi-house smoke dataset (10 houses) into the per-episode HDF5
layout expected by `submodules/act/utils.py:EpisodicDataset`, AND write a
sidecar `mapping.json` describing which (house, traj_key) each output
`episode_<i>.hdf5` came from. The mapping is used by
`pla.prox_residual_dataset` to fetch the matching proximity readings from
the original source HDF5 files.

Source layout:
  assets/datagen/pick_and_place_skin_pilot_smoke_v1/
    FrankaSkinPickAndPlacePilotSmokeConfig/
      20260510_124831/
        house_<i>/trajectories_batch_1_of_1.h5
        house_<i>/episode_<NNNNNNNN>_<cam>_batch_1_of_1.mp4

Target layout:
  act_style_data/<set>/
    episode_<i>.hdf5    (sequential global index)
    mapping.json        ({episode_idx: [house_id (int), source_h5_path (str), traj_key]})
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_run_dir", required=True, type=Path,
                   help="Parent of house_*/ directories (the date-stamped dir)")
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    args = p.parse_args()

    src = args.src_run_dir.resolve()
    dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    house_dirs = sorted(
        d for d in src.iterdir()
        if d.is_dir() and d.name.startswith("house_")
    )
    if not house_dirs:
        raise SystemExit(f"no house_*/ under {src}")

    mapping: dict[int, list] = {}
    global_idx = 0
    for hd in house_dirs:
        h_id = int(hd.name.split("_", 1)[1])
        src_h5 = hd / "trajectories_batch_1_of_1.h5"
        if not src_h5.exists():
            print(f"[skip] {hd.name}: no h5")
            continue
        # Convert this house into a staging dir, then move with offset.
        stage = dst / f".stage_house_{h_id}"
        if stage.exists():
            for f in stage.iterdir():
                f.unlink()
            stage.rmdir()
        cmd = [
            sys.executable, "-m", "scripts.convert_pla_to_act",
            "--src", str(src_h5),
            "--dst", str(stage),
            "--image_h", str(args.image_h),
            "--image_w", str(args.image_w),
        ]
        print(f"[convert] {hd.name}")
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        if rc != 0:
            raise SystemExit(f"converter failed on {src_h5}")

        ep_files = sorted(
            stage.glob("episode_*.hdf5"),
            key=lambda p: int(p.stem.split("_", 1)[1]),
        )
        for src_ep in ep_files:
            local_idx = int(src_ep.stem.split("_", 1)[1])
            new_path = dst / f"episode_{global_idx}.hdf5"
            src_ep.rename(new_path)
            mapping[global_idx] = [h_id, str(src_h5), f"traj_{local_idx}"]
            global_idx += 1
        stage.rmdir()

    with open(dst / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[done] wrote {global_idx} episodes + mapping.json to {dst}")


if __name__ == "__main__":
    main()
