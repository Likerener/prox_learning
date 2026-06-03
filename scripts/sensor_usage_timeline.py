"""Per-sensor proximity-reading + attention timeline across the manipulation trajectory.

For every h5 in the given aggregate eval dir, this script:
  1. Extracts per-step proximity readings (29 sensors × 4 substeps × 8 × 8).
  2. Computes a "sensor activity score" per step per sensor: max activation
     across the (4, 8, 8) tile and the substeps.
  3. Phase-classifies each step using TCP-object distance + gripper qpos + lift.
  4. Aggregates: per-sensor mean activity per phase, per-sensor activity over
     normalized time (0=start, 1=end of episode).

Outputs:
  * sensor_phase_heatmap.png  — rows=sensors, cols=phases, color=mean activity
  * sensor_time_heatmap.png   — rows=sensors, cols=normalised time bins,
                                color=mean activity
  * sensor_per_phase_topk.png — top-K sensors per phase (bar chart)
  * sensor_timeline_examples.png — per-sensor activity time-series for 4
                                   example trajectories (2 success + 2 fail)
  * summary.json              — per-sensor / per-phase aggregates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

LIFT_M = 0.05
PREGRASP_M = 0.10
PLACE_M = 0.10
GRIPPER_CLOSED_THRESH = 0.10   # >0.10 = closed (molmospaces inverted convention)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _tcp_world(tcp_robot: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    base_xyz = base_pose[:3]
    R = _quat_to_R(base_pose[3:7])
    return base_xyz + R @ tcp_robot[:3]


def _decode_json_field(arr: np.ndarray) -> list[dict]:
    out = []
    for row in arr:
        try:
            s = bytes(row).decode("utf-8", errors="ignore").split("\x00")[0].strip()
            out.append(json.loads(s) if s else {})
        except Exception:
            out.append({})
    return out


def classify_phases(traj: h5py.Group) -> tuple[np.ndarray, dict]:
    """Returns per-step phase labels + diagnostic dict."""
    T = traj["obs/extra/tcp_pose"].shape[0]
    tcp_world = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        tcp_world[t] = _tcp_world(traj["obs/extra/tcp_pose"][t],
                                  traj["obs/extra/robot_base_pose"][t])
    obj_xyz = traj["obs/extra/obj_start"][0, :3].astype(np.float32)
    d_xy = np.linalg.norm(tcp_world[:, :2] - obj_xyz[None, :2], axis=1)

    # Gripper held: from grasp_state_pickup_obj
    held = np.array([
        bool(d.get("gripper", {}).get("held", False))
        for d in _decode_json_field(traj["obs/extra/grasp_state_pickup_obj"][:])
    ])
    # Per-step gripper qpos (from agent qpos JSON)
    qpos = _decode_json_field(traj["obs/agent/qpos"][:])
    grip = np.array([
        float(np.mean(d.get("gripper", [0.0, 0.0])[:2])) if d else 0.0
        for d in qpos
    ])
    grip_closed = grip > GRIPPER_CLOSED_THRESH

    # Lift: TCP z relative to first-held TCP z
    first_held = int(np.argmax(held)) if held.any() else -1
    if first_held >= 0:
        lift = tcp_world[:, 2] - tcp_world[first_held, 2]
    else:
        lift = np.zeros(T)

    phases = np.empty(T, dtype=object)
    for t in range(T):
        if held[t] and lift[t] > LIFT_M:
            phases[t] = "transit"
        elif held[t]:
            phases[t] = "grasp_lift"
        elif grip_closed[t]:
            phases[t] = "grasp_lift"   # closed but not held yet (closing motion)
        elif d_xy[t] < PREGRASP_M:
            phases[t] = "pregrasp"
        else:
            phases[t] = "approach"

    # "place" overrides transit when object is close to target.
    obj_end = traj["obs/extra/obj_end"][:]
    target_xy = obj_end[0, :2] if np.linalg.norm(obj_end[0, :3]) > 1e-6 else None
    if target_xy is not None:
        obj_world = obj_xyz   # constant
        for t in range(T):
            if phases[t] == "transit" and held[t]:
                # Use TCP world as proxy for object world (since held).
                d_target = float(np.linalg.norm(tcp_world[t, :2] - target_xy))
                if d_target < PLACE_M:
                    phases[t] = "place"
    return phases, {
        "first_held": first_held,
        "T": T,
        "any_held": held.any(),
        "any_lift": bool(lift.max() > LIFT_M),
        "min_dxy": float(d_xy.min()),
        "n_held": int(held.sum()),
    }


def compute_sensor_activity(traj: h5py.Group, sensor_names: list[str]) -> np.ndarray:
    """Returns (T, N_sensors) — per-step max activation per sensor."""
    T = traj[f"obs/proximity/{sensor_names[0]}"].shape[0]
    N = len(sensor_names)
    activity = np.empty((T, N), dtype=np.float32)
    for i, sn in enumerate(sensor_names):
        # Shape (T, 4, 8, 8). For "activity" use max over the (4, 8, 8) tile but
        # take negative of value since proximity readings are distance (smaller = closer = more relevant).
        # Actually the prox sensor values are "depth" in metres; ~0.5 m typical max range.
        # A nearby object → small reading → "active".  We use (max_range - min_reading) as activity.
        arr = traj[f"obs/proximity/{sn}"][:]       # (T, 4, 8, 8)
        min_reading = arr.min(axis=(1, 2, 3))      # smaller = closer
        max_range = 0.5                             # VL53L5CX nominal max in metres
        activity[:, i] = np.clip(max_range - min_reading, 0.0, max_range)
    return activity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True,
                   help="Aggregate eval dir, e.g. eval_output/act_prox_mug_v1_aggregate_n50")
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_trajs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sensor_names = list(json.load(open(args.prox_mapping_json))["sensor_names"])
    N = len(sensor_names)

    phases_all = ["approach", "pregrasp", "grasp_lift", "transit", "place"]

    # Accumulators
    phase_activity_sum = np.zeros((len(phases_all), N), dtype=np.float64)
    phase_count_sum   = np.zeros((len(phases_all),),    dtype=np.int64)
    time_activity_sum = np.zeros((20, N), dtype=np.float64)       # 20 time bins
    time_count_sum    = np.zeros((20,),     dtype=np.int64)

    # Per-trajectory snapshots (for example plots)
    example_success: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    example_fail: list[tuple[str, np.ndarray, np.ndarray, dict]] = []

    n_processed = 0
    run_dirs = sorted([p for p in Path(args.agg_dir).iterdir()
                       if p.is_dir() and p.name.startswith("run_")])
    for rd in run_dirs[: args.max_trajs]:
        h5p = rd / "house_1" / "trajectories_batch_1_of_1.h5"
        if not h5p.exists():
            continue
        try:
            with h5py.File(h5p, "r") as f:
                if "traj_0" not in f:
                    continue
                traj = f["traj_0"]
                phases, diag = classify_phases(traj)
                activity = compute_sensor_activity(traj, sensor_names)   # (T, N)
                success = bool(traj["success"][-1])
                T = activity.shape[0]
        except Exception as e:
            print(f"skip {rd}: {e}")
            continue

        n_processed += 1

        for i, ph in enumerate(phases_all):
            mask = (phases == ph)
            if mask.any():
                phase_activity_sum[i] += activity[mask].sum(axis=0)
                phase_count_sum[i] += int(mask.sum())

        # Time bins
        for t_idx in range(T):
            bin_id = min(int(20 * t_idx / T), 19)
            time_activity_sum[bin_id] += activity[t_idx]
            time_count_sum[bin_id] += 1

        if success and len(example_success) < 2:
            example_success.append((rd.name, activity, phases, diag))
        elif (not success) and len(example_fail) < 2:
            example_fail.append((rd.name, activity, phases, diag))

    print(f"[sensor-usage] processed {n_processed} trajectories")

    # Mean activity per phase per sensor
    phase_mean = np.zeros((len(phases_all), N), dtype=np.float64)
    for i in range(len(phases_all)):
        if phase_count_sum[i] > 0:
            phase_mean[i] = phase_activity_sum[i] / phase_count_sum[i]

    # Mean activity per time-bin
    time_mean = np.zeros((20, N), dtype=np.float64)
    for b in range(20):
        if time_count_sum[b] > 0:
            time_mean[b] = time_activity_sum[b] / time_count_sum[b]

    # -- Plot 1: per-phase heatmap (rows=sensors, cols=phases)
    fig, ax = plt.subplots(figsize=(7, 9))
    ph_used = [i for i, c in enumerate(phase_count_sum) if c > 0]
    h = ax.imshow(phase_mean[ph_used].T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(N))
    ax.set_yticklabels(sensor_names, fontsize=7)
    ax.set_xticks(range(len(ph_used)))
    ax.set_xticklabels([f"{phases_all[i]}\n(n={phase_count_sum[i]})" for i in ph_used],
                       fontsize=9)
    plt.colorbar(h, ax=ax, label="sensor activity (max_range − min reading, m)")
    ax.set_title(f"Per-sensor mean activity by manipulation phase (n={n_processed} trajs)")
    # Light grid between link groups
    for boundary in [7, 15, 21]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "sensor_phase_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'sensor_phase_heatmap.png'}")

    # -- Plot 2: per-time-bin heatmap (rows=sensors, cols=time bin)
    fig, ax = plt.subplots(figsize=(11, 9))
    h = ax.imshow(time_mean.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(N))
    ax.set_yticklabels(sensor_names, fontsize=7)
    ax.set_xticks(range(0, 20, 4))
    ax.set_xticklabels([f"{i*5}%" for i in range(0, 20, 4)])
    ax.set_xlabel("normalised episode time")
    plt.colorbar(h, ax=ax, label="sensor activity")
    ax.set_title(f"Per-sensor activity over normalised time (n={n_processed} trajs)")
    for boundary in [7, 15, 21]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "sensor_time_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'sensor_time_heatmap.png'}")

    # -- Plot 3: top-K sensors per phase
    K = 5
    fig, axes = plt.subplots(1, len(ph_used), figsize=(3.4*len(ph_used), 4.2), sharey=True)
    if len(ph_used) == 1:
        axes = [axes]
    for ax, ph_i in zip(axes, ph_used):
        vec = phase_mean[ph_i]
        top = np.argsort(vec)[-K:][::-1]
        names_top = [sensor_names[j] for j in top]
        vals_top = vec[top]
        ax.barh(range(K), vals_top[::-1], color="#1f77b4", edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(K))
        ax.set_yticklabels(names_top[::-1], fontsize=8)
        ax.set_title(f"{phases_all[ph_i]}\n(n_steps={phase_count_sum[ph_i]})", fontsize=10)
        ax.set_xlabel("mean activity")
    fig.suptitle(f"Top-{K} most active sensors per phase ({n_processed} trajs)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "sensor_per_phase_topk.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'sensor_per_phase_topk.png'}")

    # -- Plot 4: per-trajectory examples
    examples = example_success + example_fail
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(13, 2.8 * len(examples)), sharex=False)
        if len(examples) == 1:
            axes = [axes]
        for ax, (name, activity, phases, diag) in zip(axes, examples):
            T = activity.shape[0]
            t = np.arange(T)
            # Show TOP-3 most active sensors for this trajectory + average of all.
            mean_act = activity.mean(axis=0)
            top = np.argsort(mean_act)[-3:][::-1]
            for j in top:
                ax.plot(t, activity[:, j], label=sensor_names[j], linewidth=1.4)
            ax.plot(t, activity.mean(axis=1), color="black", linewidth=0.8, linestyle="--",
                    alpha=0.6, label="mean (all 29)")
            # Shade phase changes
            phase_colors = {"approach": "#deebf7", "pregrasp": "#fed9a6",
                            "grasp_lift": "#fdae6b", "transit": "#bcbddc",
                            "place": "#a6dba0"}
            curr_ph, curr_start = phases[0], 0
            for t_idx in range(1, T):
                if phases[t_idx] != curr_ph:
                    ax.axvspan(curr_start, t_idx, color=phase_colors.get(curr_ph, "#eee"),
                               alpha=0.45, zorder=-1)
                    curr_ph = phases[t_idx]
                    curr_start = t_idx
            ax.axvspan(curr_start, T, color=phase_colors.get(curr_ph, "#eee"),
                       alpha=0.45, zorder=-1)
            ax.set_xlabel("step")
            ax.set_ylabel("activity")
            success = (name, activity, phases, diag) in example_success
            ax.set_title(f"{name}  {'success' if success else 'failure'}  "
                         f"T={T}, first_held={diag.get('first_held')}, "
                         f"n_held={diag.get('n_held')}",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(0, T)
        # Phase legend
        phase_handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.45)
                         for v in phase_colors.values()]
        fig.legend(phase_handles, list(phase_colors.keys()),
                   loc="lower center", ncol=len(phase_colors), fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out / "sensor_timeline_examples.png", dpi=140)
        plt.close(fig)
        print(f"  wrote {out/'sensor_timeline_examples.png'}")

    # -- Summary JSON
    summary = {
        "n_processed": n_processed,
        "sensor_names": sensor_names,
        "phases": phases_all,
        "phase_step_counts": phase_count_sum.tolist(),
        "phase_mean_activity": phase_mean.tolist(),
        "time_bin_counts": time_count_sum.tolist(),
        "time_mean_activity": time_mean.tolist(),
        # Top sensors per phase
        "top_sensors_per_phase": {
            phases_all[i]: [
                {"sensor": sensor_names[j], "mean_activity": float(phase_mean[i, j])}
                for j in np.argsort(phase_mean[i])[-5:][::-1]
            ] for i in ph_used
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sensor-usage] outputs in {out}")


if __name__ == "__main__":
    main()
