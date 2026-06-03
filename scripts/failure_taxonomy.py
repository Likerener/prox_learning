"""Failure taxonomy classifier for ACT and P+ACT eval rollouts.

Given a directory of `runs/.../trajectories_batch_1_of_1.h5` files from an
aggregate eval (e.g. `eval_output/act_prox_mug_v1_aggregate_n50`), classifies
every FAILED rollout into one of 5 buckets defined a priori (Exp 3):

  1. missed_reach    — TCP never came within ~10 cm of object xy
  2. bad_pregrasp    — TCP did approach the object but gripper never grasped
                       (object never held)
  3. grasp_lift_fail — gripper grasped the object but the object never lifted
                       above the lift threshold while held
  4. placement_error — object was lifted but not placed near the target
                       (task_info.position_error never went below threshold)
  5. other           — anything we couldn't classify, or timeout

Reads:
  obs/extra/tcp_pose          (T, 7)  robot-frame
  obs/extra/obj_start         (T, 7)  world-frame (constant — initial pose)
  obs/extra/robot_base_pose   (T, 7)  world-frame
  obs/extra/task_info         (T, ?)  json (per step), position_error key
  obs/extra/grasp_state_pickup_obj  (T, ?)  json (per step), gripper.held
  obs/agent/qpos              (T, ?)  json — gripper finger pos
  success                     (T,)    bool

Run:
    /opt/conda/envs/mlspaces/bin/python scripts/failure_taxonomy.py \\
        --baseline_dir eval_output/act_house1_mug_random_v1_aggregate_n50 \\
        --pact_dir     eval_output/act_prox_mug_v1_aggregate_n50 \\
        --output_dir   eval_output/exp3_failure_taxonomy
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Classification thresholds (locked before looking at data).
# ----------------------------------------------------------------------
APPROACH_DIST_M     = 0.10   # within this distance of object xy → "reached"
LIFT_HEIGHT_M       = 0.05   # vertical TCP rise while held → "lifted"
PLACE_ERROR_M       = 0.10   # task_info.position_error below this → "placed"
HELD_MIN_STEPS      = 3      # require object held for ≥ this many steps

CATEGORIES = ["missed_reach", "bad_pregrasp", "grasp_lift_fail",
              "placement_error", "other"]


@dataclass
class TrajClassification:
    run_idx: int
    condition: str
    success: bool
    category: str
    notes: str
    # Diagnostic stats:
    min_tcp_obj_dist: float
    max_held_steps: int
    held_lift_height: float    # max TCP rise while held
    min_position_error: float


def _world_xy_from_pose(pose7: np.ndarray) -> np.ndarray:
    return np.asarray(pose7[:2], dtype=np.float32)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert (qw, qx, qy, qz) or (qx, qy, qz, qw) → 3x3. Assumes wxyz."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _tcp_to_world(tcp_pose: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """tcp_pose, base_pose: (7,) = (xyz, qw, qx, qy, qz) in robot/world frames.
    Returns TCP xyz in world frame."""
    base_xyz = base_pose[:3]
    base_q = base_pose[3:7]
    R = _quat_to_R(base_q)
    tcp_xyz_in_robot = tcp_pose[:3]
    return base_xyz + R @ tcp_xyz_in_robot


def _decode_json_field(arr: np.ndarray) -> list[dict]:
    """Decode a (T, N) uint8 array of NUL-padded JSON bytes to a list of dicts."""
    out = []
    for row in arr:
        try:
            s = bytes(row).decode("utf-8", errors="ignore").rstrip("\x00 ")
            # Strip trailing nulls.
            s = s.split("\x00")[0].strip()
            out.append(json.loads(s) if s else {})
        except Exception:
            out.append({})
    return out


def _safe_float(d: dict, key: str, default: float = np.nan) -> float:
    v = d.get(key, default)
    if isinstance(v, str) and v == "Infinity":
        return float("inf")
    try:
        return float(v)
    except Exception:
        return default


def classify_trajectory(h5_path: Path, run_idx: int, condition: str
                        ) -> Optional[TrajClassification]:
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        if "traj_0" not in f:
            return None
        traj = f["traj_0"]
        tcp_pose   = traj["obs/extra/tcp_pose"][:]              # (T, 7)
        base_pose  = traj["obs/extra/robot_base_pose"][:]       # (T, 7)
        obj_start  = traj["obs/extra/obj_start"][:]             # (T, 7) (constant)
        success    = bool(traj["success"][-1])

        task_info_arr  = traj["obs/extra/task_info"][:]
        grasp_arr      = traj["obs/extra/grasp_state_pickup_obj"][:]

    T = tcp_pose.shape[0]

    # Compute TCP in world frame at each step.
    tcp_world = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        tcp_world[t] = _tcp_to_world(tcp_pose[t], base_pose[t])

    obj_world_xy = obj_start[0, :2]   # constant over time
    tcp_obj_dist_xy = np.linalg.norm(tcp_world[:, :2] - obj_world_xy[None], axis=1)
    min_dist = float(tcp_obj_dist_xy.min())

    # Parse per-step json.
    grasp = _decode_json_field(grasp_arr)
    held_steps = np.array([
        bool(d.get("gripper", {}).get("held", False)) for d in grasp
    ])
    max_held_run = 0
    run = 0
    for h in held_steps:
        if h:
            run += 1
            if run > max_held_run:
                max_held_run = run
        else:
            run = 0
    n_held = int(held_steps.sum())

    # Lift height: max TCP z-rise while held, relative to TCP z at first-held step.
    held_idx = np.where(held_steps)[0]
    if len(held_idx) > 0:
        z0 = tcp_world[held_idx[0], 2]
        held_lift = float((tcp_world[held_idx, 2] - z0).max())
    else:
        held_lift = 0.0

    # task_info: min position_error (object → target distance)
    task_info = _decode_json_field(task_info_arr)
    pos_errors = np.array([_safe_float(d, "position_error", np.nan) for d in task_info])
    finite_pe = pos_errors[np.isfinite(pos_errors)]
    min_pe = float(finite_pe.min()) if len(finite_pe) else float("inf")

    # ---- Classification ----
    if success:
        cat = "success"
        notes = "task succeeded"
    elif min_dist > APPROACH_DIST_M:
        cat = "missed_reach"
        notes = f"closest TCP-obj xy = {min_dist:.3f} m > {APPROACH_DIST_M}"
    elif n_held < HELD_MIN_STEPS:
        cat = "bad_pregrasp"
        notes = f"reached object (d={min_dist:.3f}) but gripper never held (n_held={n_held})"
    elif held_lift < LIFT_HEIGHT_M:
        cat = "grasp_lift_fail"
        notes = f"held but max-lift={held_lift:.3f} m < {LIFT_HEIGHT_M}"
    elif min_pe > PLACE_ERROR_M:
        cat = "placement_error"
        notes = f"lifted (held={n_held}, lift={held_lift:.2f}) but min_pe={min_pe:.3f} > {PLACE_ERROR_M}"
    else:
        cat = "other"
        notes = f"task-info: n_held={n_held}, lift={held_lift:.3f}, min_pe={min_pe:.3f}"

    return TrajClassification(
        run_idx=run_idx,
        condition=condition,
        success=success,
        category=cat,
        notes=notes,
        min_tcp_obj_dist=min_dist,
        max_held_steps=max_held_run,
        held_lift_height=held_lift,
        min_position_error=min_pe,
    )


def collect_dir(agg_dir: Path, condition: str) -> list[TrajClassification]:
    rows: list[TrajClassification] = []
    run_dirs = sorted([p for p in agg_dir.iterdir()
                       if p.is_dir() and p.name.startswith("run_")])
    for rd in run_dirs:
        run_idx = int(rd.name.split("_")[-1])
        h5 = rd / "house_1" / "trajectories_batch_1_of_1.h5"
        c = classify_trajectory(h5, run_idx, condition)
        if c is not None:
            rows.append(c)
    return rows


def chi_square_2x5(baseline_rows: list[TrajClassification],
                   pact_rows: list[TrajClassification]) -> dict:
    """Chi-squared on (baseline_failures × 5 categories) vs (pact_failures × 5).
    Returns {chi2, dof, p, cramers_v, table}.
    """
    cats = ["missed_reach", "bad_pregrasp", "grasp_lift_fail",
            "placement_error", "other"]
    obs = np.zeros((2, len(cats)), dtype=np.int64)
    for r in baseline_rows:
        if r.category in cats:
            obs[0, cats.index(r.category)] += 1
    for r in pact_rows:
        if r.category in cats:
            obs[1, cats.index(r.category)] += 1
    n = int(obs.sum())
    row_tot = obs.sum(axis=1, keepdims=True)   # (2, 1)
    col_tot = obs.sum(axis=0, keepdims=True)   # (1, 5)
    if n == 0:
        return {"chi2": 0.0, "dof": 0, "p": 1.0, "cramers_v": 0.0,
                "table": obs.tolist(), "categories": cats}
    exp = row_tot @ col_tot / n
    # Use a small epsilon to avoid div by zero where expected==0.
    nz = exp > 0
    chi2 = float(((obs[nz] - exp[nz]) ** 2 / exp[nz]).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(1.0 - chi2_dist.cdf(chi2, dof))
    except Exception:
        p = float("nan")
    cramers_v = float(np.sqrt(chi2 / (n * min(obs.shape[0]-1, obs.shape[1]-1))))
    return {
        "chi2": chi2, "dof": dof, "p": p, "cramers_v": cramers_v,
        "table": obs.tolist(), "categories": cats,
        "row_totals": row_tot.flatten().tolist(),
    }


def plot_taxonomy(rows_by_cond: dict[str, list[TrajClassification]], out: Path) -> None:
    cats = ["missed_reach", "bad_pregrasp", "grasp_lift_fail",
            "placement_error", "other"]
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5))

    x = np.arange(len(cats))
    colors = ["#1f77b4", "#ff7f0e"]
    for i, (cond, rows) in enumerate(rows_by_cond.items()):
        counts = [sum(1 for r in rows if r.category == c) for c in cats]
        ax.bar(x + (i - 0.5) * width, counts, width,
               label=f"{cond} (n={sum(counts)} failures)",
               color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)
        for j, n in enumerate(counts):
            ax.text(x[j] + (i - 0.5) * width, n + 0.2, str(n),
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("# failed rollouts")
    ax.set_title("Failure taxonomy — baseline vs P+ACT")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", required=True,
                   help="Aggregate dir for vanilla ACT (contains run_NN/)")
    p.add_argument("--pact_dir", required=True,
                   help="Aggregate dir for P+ACT (contains run_NN/)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_rows = collect_dir(Path(args.baseline_dir), "baseline")
    pact_rows = collect_dir(Path(args.pact_dir), "pact")
    base_fails = [r for r in base_rows if not r.success]
    pact_fails = [r for r in pact_rows if not r.success]

    # CSV.
    import csv
    with open(out / "classifications.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "run_idx", "success", "category",
                    "min_tcp_obj_dist", "max_held_steps", "held_lift_height",
                    "min_position_error", "notes"])
        for r in (base_rows + pact_rows):
            w.writerow([r.condition, r.run_idx, r.success, r.category,
                        f"{r.min_tcp_obj_dist:.4f}", r.max_held_steps,
                        f"{r.held_lift_height:.4f}",
                        ("inf" if not np.isfinite(r.min_position_error)
                         else f"{r.min_position_error:.4f}"),
                        r.notes])

    # Chi-squared.
    stat = chi_square_2x5(base_fails, pact_fails)
    with open(out / "chi_square.json", "w") as f:
        json.dump(stat, f, indent=2)

    plot_taxonomy({"baseline": base_fails, "P+ACT": pact_fails},
                  out / "failure_taxonomy.png")

    print(f"[exp3] base: {len(base_fails)}/{len(base_rows)} failed")
    print(f"[exp3] pact: {len(pact_fails)}/{len(pact_rows)} failed")
    print(f"[exp3] chi2={stat['chi2']:.2f}  dof={stat['dof']}  "
          f"p={stat['p']:.4f}  Cramers V={stat['cramers_v']:.3f}")
    print(f"[exp3] table:")
    for i, cond in enumerate(["baseline", "P+ACT"]):
        print(f"  {cond:>8}: " + " ".join(f"{c}={n}" for c, n in
                                          zip(stat['categories'], stat['table'][i])))
    print(f"[exp3] outputs in {out}")


if __name__ == "__main__":
    main()
