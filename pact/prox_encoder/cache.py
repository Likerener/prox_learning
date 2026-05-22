"""Preprocess h5 trajectories into a flat windowed cache.

Each training sample is one (trajectory, sensor, time t) tuple. Inputs are
the window of W control-step proximity readings ending at t (shape
(W, 4, 8, 8)). Labels are the 3D position of the pickup object expressed
in the sensor's local frame at time t.

Filtering rules (a sample is kept iff ALL hold):
  - The pickup object's bounding-box projection has num_points > 0 in this
    sensor at time t (i.e. the object is in the sensor's field of view).
  - The gripper is NOT yet holding the object at time t (grasp_state.held
    is false), so obs/extra/obj_start is still the true object world pose.
  - t >= W - 1, so the window has W real frames (no edge padding).

We compute the label as extrinsic_cv @ [obj_world; 1] using
obs/sensor_param/<sensor>/extrinsic_cv (world -> sensor, OpenCV).

Output layout (single .npz):
  prox:        (N, W, 4, 8, 8) fp16  raw depth, NOT normalized.
  label:       (N, 3)          fp32  object position in sensor frame [m].
  sensor_id:   (N,)            int16 index into sensor_names array.
  traj_id:     (N,)            int32 index into traj_paths array.
  t:           (N,)            int32 time index inside trajectory.
  sensor_names: (S,)           U32   sensor key strings.
  traj_paths:  (M,)            U512  source h5 paths.
  prox_mean / prox_std: (4, 8, 8) fp32 channel-wise stats on the cached samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


SENSOR_LINKS = ("link2", "link3", "link5", "link6")


def list_sensors(g: h5py.Group) -> List[str]:
    keys = []
    for k in sorted(g["obs/proximity"].keys()):
        if any(k.startswith(L + "_sensor_") for L in SENSOR_LINKS):
            keys.append(k)
    return keys


def grasp_held_mask(g: h5py.Group) -> np.ndarray:
    """Return per-timestep boolean held=True (object is grasped)."""
    raw = g["obs/extra/grasp_state_pickup_obj"][:]
    T = raw.shape[0]
    held = np.zeros(T, dtype=bool)
    for t in range(T):
        b = raw[t]
        # bytes are null-padded; decode then JSON-parse.
        s = bytes(b).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        if not s:
            continue
        try:
            d = json.loads(s)
            held[t] = bool(d.get("gripper", {}).get("held", False))
        except (json.JSONDecodeError, KeyError, AttributeError):
            held[t] = False
    return held


def collect_from_traj(
    h5_path: Path,
    window: int,
    keep_every: int,
) -> List[Tuple[np.ndarray, np.ndarray, str, int]]:
    """Return list of (window_arr, label_arr, sensor_name, t_idx) for this traj."""
    out = []
    with h5py.File(h5_path, "r") as f:
        if "traj_0" not in f:
            return out
        g = f["traj_0"]
        T = g["obs/extra/obj_start"].shape[0]
        if T < window:
            return out

        obj_world = g["obs/extra/obj_start"][:, :3].astype(np.float32)  # (T,3) constant
        held = grasp_held_mask(g)

        sensors = list_sensors(g)
        for sk in sensors:
            ip_key = f"obs/extra/object_image_points/pickup_obj/{sk}/num_points"
            if ip_key not in g:
                continue
            n_pts = g[ip_key][:].squeeze(-1).astype(np.int32)  # (T,)
            visible = n_pts > 0
            # Require: visible, not held, full window available.
            valid_t = np.where(visible & (~held))[0]
            valid_t = valid_t[valid_t >= window - 1]
            if valid_t.size == 0:
                continue
            if keep_every > 1:
                valid_t = valid_t[::keep_every]
            if valid_t.size == 0:
                continue

            prox_ds = g[f"obs/proximity/{sk}"]  # (T, 4, 8, 8)
            ext_ds = g[f"obs/sensor_param/{sk}/extrinsic_cv"]  # (T, 3, 4) world -> sensor (cv)
            # Bulk read what we need.
            for t in valid_t:
                start = int(t) - window + 1
                end = int(t) + 1
                w = prox_ds[start:end].astype(np.float16)  # (W, 4, 8, 8)
                E = ext_ds[int(t)]
                obj_w = obj_world[int(t)]
                obj_in_sensor = (E[:, :3] @ obj_w + E[:, 3]).astype(np.float32)
                out.append((w, obj_in_sensor, sk, int(t)))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_glob",
        default="/home/jaydv/code/prox_learning/assets/datagen/"
        "mug_house_1_random_everything/FrankaSkinPickAndPlacePilotMediumConfig/"
        "*/house_1/trajectories_batch_1_of_1.h5",
        help="Glob for source h5 trajectories.",
    )
    p.add_argument("--out", default="pact/outputs_prox/cache.npz")
    p.add_argument("--window", type=int, default=8,
                   help="Number of control steps in the proximity window (each step = 4 sub-frames).")
    p.add_argument("--keep_every", type=int, default=2,
                   help="Subsample valid timesteps by this stride to limit dataset size.")
    p.add_argument("--max_trajs", type=int, default=0,
                   help="If >0, limit to this many trajectories (smoke testing).")
    p.add_argument("--label_clip_m", type=float, default=3.0,
                   help="Drop samples whose ||obj-in-sensor|| exceeds this (likely "
                        "object behind the camera or visibility-flag bug).")
    args = p.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if args.max_trajs > 0:
        paths = paths[: args.max_trajs]
    if not paths:
        print(f"[cache] no trajectories matched {args.data_glob!r}", file=sys.stderr)
        return 1
    print(f"[cache] {len(paths)} trajectories, window={args.window}, keep_every={args.keep_every}")

    samples: List[Tuple[np.ndarray, np.ndarray, str, int, int]] = []
    t0 = time.time()
    for i, p_ in enumerate(tqdm(paths, desc="scan")):
        try:
            traj_samples = collect_from_traj(Path(p_), args.window, args.keep_every)
        except (OSError, KeyError) as e:
            print(f"[cache] skip {p_}: {e}", file=sys.stderr)
            continue
        for w, lab, sk, t in traj_samples:
            if np.linalg.norm(lab) > args.label_clip_m:
                continue
            samples.append((w, lab, sk, t, i))
    print(f"[cache] collected {len(samples)} samples in {time.time()-t0:.1f}s")
    if not samples:
        print("[cache] no samples; aborting", file=sys.stderr)
        return 1

    # Stack.
    sensor_names_list = sorted({s[2] for s in samples})
    sensor_idx = {n: i for i, n in enumerate(sensor_names_list)}

    prox = np.stack([s[0] for s in samples], axis=0)               # (N, W, 4, 8, 8) fp16
    label = np.stack([s[1] for s in samples], axis=0)              # (N, 3) fp32
    sensor_id = np.array([sensor_idx[s[2]] for s in samples], np.int16)
    t_arr = np.array([s[3] for s in samples], np.int32)
    traj_id = np.array([s[4] for s in samples], np.int32)

    # Channel-wise stats (over valid samples, fp32).
    prox_f32 = prox.astype(np.float32)
    prox_mean = prox_f32.reshape(-1, 4, 8, 8).mean(axis=0)
    prox_std = prox_f32.reshape(-1, 4, 8, 8).std(axis=0) + 1e-6

    # Label stats for normalization.
    label_mean = label.mean(axis=0)
    label_std = label.std(axis=0) + 1e-6

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        prox=prox,
        label=label,
        sensor_id=sensor_id,
        t=t_arr,
        traj_id=traj_id,
        sensor_names=np.array(sensor_names_list),
        traj_paths=np.array([str(p_) for p_ in paths]),
        prox_mean=prox_mean,
        prox_std=prox_std,
        label_mean=label_mean,
        label_std=label_std,
        window=np.int32(args.window),
    )
    print(
        f"[cache] wrote {out_path} -> N={len(samples)} "
        f"size={os.path.getsize(out_path)/1e6:.1f}MB"
    )
    print(f"  label_mean={label_mean} label_std={label_std}")
    print(f"  prox_mean range [{prox_mean.min():.3f}, {prox_mean.max():.3f}]")
    print(f"  per-sensor sample counts:")
    for sn, sid in sensor_idx.items():
        c = int((sensor_id == sid).sum())
        print(f"    {sn:24s}  {c:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
