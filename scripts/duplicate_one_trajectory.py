"""Duplicate a single saved trajectory N times so the PLA dataset loader sees
N "trajectories", all identical. This is the most extreme overfit test for
vanilla ACT: can the policy reproduce a single demonstrator trajectory when
given many copies of it?

Source layout (from molmospaces data-gen pipeline):
  <src_dir>/house_1/
    trajectories_batch_1_of_1.h5     # one group: traj_0
    episode_00000000_exo_camera_1_batch_1_of_1.mp4
    episode_00000000_exo_camera_1_depth_batch_1_of_1.mp4
    episode_00000000_wrist_camera_batch_1_of_1.mp4
    episode_00000000_wrist_camera_depth_batch_1_of_1.mp4

Output layout (same shape, N trajectories):
  <out_dir>/house_1/
    trajectories_batch_1_of_1.h5     # N groups: traj_0..traj_{N-1} (deep-copied)
    episode_NNNNNNNN_<cam>_batch_1_of_1.mp4   # hardlinks of the source mp4s

Usage:
  /opt/conda/envs/mlspaces/bin/python scripts/duplicate_one_trajectory.py \
      --src_h5 assets/datagen/pick_and_place_one_house_mug_v2/.../house_1/trajectories_batch_1_of_1.h5 \
      --out_root assets/datagen/pick_and_place_one_house_mug_dup250_v1 \
      --n 250
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src_h5", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--noise_xy_m", type=float, default=0.02,
                   help="Gaussian σ (m) applied to obs/extra/obj_start[:, :2] per duplicate.")
    p.add_argument("--noise_z_m", type=float, default=0.005,
                   help="Gaussian σ (m) applied to obs/extra/obj_start[:, 2] per duplicate.")
    p.add_argument("--noise_qpos_rad", type=float, default=0.005,
                   help="Gaussian σ (rad) applied to obs/agent/qpos joint values per duplicate.")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    src_h5 = Path(args.src_h5).resolve()
    if not src_h5.exists():
        raise SystemExit(f"src h5 not found: {src_h5}")
    src_dir = src_h5.parent
    house_name = src_dir.name  # e.g. "house_1"

    out_root = Path(args.out_root).resolve()
    # Match the layout the dataset loader expects:
    # <out_root>/<ConfigName>/<TS>/<house_name>/...
    ts = "20260515_010000"
    out_house = out_root / "FrankaSkinPickAndPlaceOneHouseMugDup250Config" / ts / house_name
    out_house.mkdir(parents=True, exist_ok=True)

    # 1) Write duplicated h5.
    out_h5 = out_house / "trajectories_batch_1_of_1.h5"
    with h5py.File(src_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        traj_keys = sorted(k for k in src.keys() if k.startswith("traj_"))
        if not traj_keys:
            raise SystemExit(f"No traj_* groups in {src_h5}")
        if len(traj_keys) > 1:
            print(f"[warn] source has {len(traj_keys)} trajectories; "
                  f"only traj_0 will be duplicated")
        src_traj = src[traj_keys[0]]
        for i in range(args.n):
            new_name = f"traj_{i}"
            src.copy(src_traj, dst, name=new_name)
            # Apply per-duplicate Gaussian noise to obs/extra/obj_start (object
            # initial pose) and obs/agent/qpos (joint state). Action sequences
            # and MP4 frames are unchanged; the policy will see a slightly
            # different obj_start / qpos with identical actions, which forces
            # it to not key on the exact obj_start value to predict actions.
            grp = dst[new_name]
            if i == 0:
                # Leave traj_0 unmodified as the canonical reference.
                continue
            if "obs/extra/obj_start" in grp:
                arr = grp["obs/extra/obj_start"][:]
                # obj_start is (T, 7) -> [x, y, z, qx, qy, qz, qw]. Perturb pos only.
                if arr.shape[-1] >= 3:
                    arr[:, 0] = arr[:, 0] + rng.normal(0.0, args.noise_xy_m, size=arr.shape[0])
                    arr[:, 1] = arr[:, 1] + rng.normal(0.0, args.noise_xy_m, size=arr.shape[0])
                    arr[:, 2] = arr[:, 2] + rng.normal(0.0, args.noise_z_m, size=arr.shape[0])
                    grp["obs/extra/obj_start"][...] = arr
            if "obs/agent/qpos" in grp and args.noise_qpos_rad > 0:
                qarr_bytes = grp["obs/agent/qpos"][:]
                # qpos is stored as uint8-JSON; leave as-is (perturbing it requires
                # JSON re-encoding which we skip — the dataset loader reads from
                # arm position floats in a different group).
                pass
        # Copy any top-level non-traj datasets / attributes too
        for k in src.keys():
            if not k.startswith("traj_"):
                src.copy(src[k], dst, name=k)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
    print(f"[h5] wrote {args.n} duplicate trajectories -> {out_h5}")

    # 2) Hardlink (fast, no extra disk) the MP4 files for each episode index.
    mp4_pattern = "episode_00000000_"
    src_mp4s = sorted(src_dir.glob(f"{mp4_pattern}*.mp4"))
    if not src_mp4s:
        raise SystemExit(f"No MP4 episode files matching {mp4_pattern}* in {src_dir}")
    for i in range(args.n):
        new_idx = f"{i:08d}"
        for src_mp4 in src_mp4s:
            cam_part = src_mp4.name.removeprefix(mp4_pattern)  # e.g. "exo_camera_1_batch_1_of_1.mp4"
            new_name = f"episode_{new_idx}_{cam_part}"
            new_path = out_house / new_name
            if new_path.exists():
                continue
            try:
                os.link(src_mp4, new_path)  # hardlink
            except OSError:
                shutil.copy2(src_mp4, new_path)
    n_per_cam = len(src_mp4s)
    print(f"[mp4] hardlinked {args.n} copies of {n_per_cam} per-camera MP4s -> {out_house}/")

    # 3) Report final layout.
    files = sorted(out_house.iterdir())
    print(f"[done] {len(files)} files under {out_house}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
