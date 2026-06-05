"""Quantify whether proximity sensing is NECESSARY in a dataset of franka_skin
trajectories — i.e. how often vision loses the target while the skin still has
actionable signal. This is both the diagnostic ("is this environment useless for
the proximity thesis?") and the selection filter ("keep trajectories where the
skin is active >=80% and vision is blind").

Per trajectory it computes, over the T policy steps:
  vision_sees_target   : exo_camera_1 OR wrist_camera has >0 projected target points
  vision_blind         : neither RGB camera sees the target
  prox_active          : >=1 skin sensor reads a surface within --near_m   (skin is "doing work")
  prox_sees_target     : >=1 skin sensor has >0 projected target points
  necessity (hard)     : vision_blind  AND  prox_sees_target   (only the skin sees the target)
  necessity (soft)     : vision_blind  AND  prox_active        (vision blind, skin has near info)

A good "proximity-useful" environment has high prox_active_frac (target ~>=0.8)
AND non-trivial vision_blind_frac / necessity_frac. The current pick-and-place
pipeline guarantees vision sees the target (visibility constraint), so necessity ~ 0.

Usage:
    python scripts/proximity_necessity.py --glob 'assets/datagen/**/house_*/trajectories_batch_*.h5'
    python scripts/proximity_necessity.py --h5 <one.h5> [--near_m 0.15] [--out diagnostics_output/prox_necessity]
"""
from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROX_MIN_VALID = 0.05
PROX_MAX_VALID = 4.0
TARGET = "pickup_obj"
RGB_CAMS = ("exo_camera_1", "wrist_camera")


def npoints(oip_group, target, cam, T):
    key = f"{target}/{cam}/num_points"
    if key not in oip_group:
        return np.zeros(T)
    return np.asarray(oip_group[key]).reshape(-1)[:T].astype(float)


def analyze_traj(t, near_m):
    T = t["rewards"].shape[0]
    oip = t["obs/extra/object_image_points"]

    # vision: does any RGB camera see the target?
    vis = np.zeros(T, dtype=bool)
    for cam in RGB_CAMS:
        vis |= npoints(oip, TARGET, cam, T) > 0
    vision_blind = ~vis

    # proximity sensors
    prox_keys = sorted(t["obs/proximity"].keys(),
                       key=lambda k: (int(k.split("_")[0][4:]), int(k.split("_")[-1])))
    nearest = np.full((T, len(prox_keys)), np.nan)
    sees_t = np.zeros(T, dtype=bool)
    for j, k in enumerate(prox_keys):
        px = t["obs/proximity"][k][:].reshape(T, -1)
        valid = np.where((px > PROX_MIN_VALID) & (px <= PROX_MAX_VALID), px, np.nan)
        with np.errstate(all="ignore"):
            nearest[:, j] = np.nanmin(valid, axis=1)
        sees_t |= npoints(oip, TARGET, k, T) > 0
    with np.errstate(all="ignore"):
        closest = np.nanmin(nearest, axis=1)
    prox_active = np.isfinite(closest) & (closest < near_m)

    return {
        "n_steps": int(T),
        "success": bool(t["success"][-1]),
        "vision_sees_target_frac": float(vis.mean()),
        "vision_blind_frac": float(vision_blind.mean()),
        "prox_active_frac": float(prox_active.mean()),
        "prox_sees_target_frac": float(sees_t.mean()),
        "necessity_hard_frac": float((vision_blind & sees_t).mean()),
        "necessity_soft_frac": float((vision_blind & prox_active).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", action="append", default=[])
    p.add_argument("--glob", default=None)
    p.add_argument("--near_m", type=float, default=0.15,
                   help="a skin sensor is 'active' when it reads a surface closer than this (m)")
    p.add_argument("--out", default="diagnostics_output/prox_necessity")
    args = p.parse_args()

    paths = list(args.h5)
    if args.glob:
        paths += sorted(globlib.glob(args.glob, recursive=True))
    paths = sorted(set(paths))
    if not paths:
        print("no h5 files matched"); return 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pth in paths:
        try:
            f = h5py.File(pth, "r")
        except Exception as e:
            print(f"[skip] {pth}: {e}"); continue
        for tk in f.keys():
            try:
                m = analyze_traj(f[tk], args.near_m)
            except Exception as e:
                print(f"[skip] {pth}:{tk}: {e}"); continue
            m["file"] = str(pth); m["traj"] = tk
            m["house"] = Path(pth).parent.name
            rows.append(m)

    if not rows:
        print("no trajectories analyzed"); return 1

    # ranked table
    rows.sort(key=lambda r: r["necessity_soft_frac"], reverse=True)
    cols = ["house", "traj", "n_steps", "success", "vision_blind_frac",
            "prox_active_frac", "prox_sees_target_frac", "necessity_hard_frac", "necessity_soft_frac"]
    w = {"house": 9, "traj": 7}
    hdr = "  ".join(c.ljust(w.get(c, 10)) for c in cols)
    print(f"\nproximity-necessity (near_m={args.near_m} m), {len(rows)} trajectories\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print("  ".join(
            (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])).ljust(w.get(c, 10))
            for c in cols))

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ["vision_blind_frac", "prox_active_frac", "prox_sees_target_frac",
                     "necessity_hard_frac", "necessity_soft_frac"]}
    agg["n_trajectories"] = len(rows)
    agg["frac_meeting_prox_active_0.8"] = float(np.mean([r["prox_active_frac"] >= 0.8 for r in rows]))
    print("\n=== DATASET MEANS ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    verdict = ("USELESS for the proximity thesis: vision basically never fails"
               if agg["necessity_soft_frac"] < 0.1 else
               "promising: vision fails while the skin is active a meaningful fraction of the time")
    print(f"\nVERDICT: {verdict}")

    # write csv + json
    with open(out / "necessity.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols + ["file"])
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r[k] for k in cols + ["file"]})
    (out / "necessity_summary.json").write_text(json.dumps({"means": agg, "near_m": args.near_m}, indent=2))

    # scatter: vision_blind vs prox_active, one point per trajectory
    fig, ax = plt.subplots(figsize=(7, 6))
    vb = [r["vision_blind_frac"] for r in rows]
    pa = [r["prox_active_frac"] for r in rows]
    sc = [r["necessity_soft_frac"] for r in rows]
    s = ax.scatter(vb, pa, c=sc, cmap="viridis", s=60, edgecolor="k", vmin=0, vmax=max(0.05, max(sc)))
    ax.axhline(0.8, color="red", ls="--", lw=1, label="prox active 80% target")
    ax.set(xlabel="vision-blind fraction of trajectory",
           ylabel="proximity-active fraction of trajectory",
           title=f"Proximity necessity per trajectory (n={len(rows)})",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    fig.colorbar(s, ax=ax, label="necessity (vision-blind & prox-active)")
    ax.legend()
    fig.savefig(out / "necessity_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote necessity.csv / necessity_summary.json / necessity_scatter.png to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
