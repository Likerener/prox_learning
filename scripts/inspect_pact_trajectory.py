"""Render a full diagnostic report for one PACT (franka_skin) datagen trajectory.

Covers task, proprioception (qpos/qvel), commanded actions vs. realized joints,
TCP / object trajectories, the 29 proximity skin sensors, manipulation phases,
reward/success, and the collision metric (incl. collision probability).

Usage:
    python scripts/inspect_pact_trajectory.py --h5 <path/to/trajectories_batch_*.h5> \
        [--traj traj_0] [--out diagnostics_output/<name>]

Works on any MolmoSpaces franka_skin pick-and-place h5 (PACT and friends).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PHASE_NAMES = [
    "unknown", "gripper-open", "pregrasp", "grasp", "gripper-close",
    "lift", "preplace", "place", "retreat", "go_home",
]
PHASE_COLORS = plt.cm.tab10(np.linspace(0, 1, len(PHASE_NAMES)))
PROX_CLIP_M = 4.0      # SPAD valid range upper bound, for display
PROX_MIN_VALID = 0.05  # SPAD valid range lower bound


# ----------------------------------------------------------------------------- helpers
def decode_json_row(row: np.ndarray):
    """Decode one (2000,) / (4000,) uint8 JSON-bytes row into a python object."""
    b = bytes(np.asarray(row, dtype=np.uint8)).split(b"\x00", 1)[0]
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def decode_field(group, key, subkey):
    """Stack arm/gripper sub-lists from a (T,N) JSON-bytes dataset into (T, d)."""
    ds = group[key][:]
    rows = [decode_json_row(r) or {} for r in ds]
    out = [np.asarray(r.get(subkey, []), dtype=np.float64) for r in rows]
    width = max((len(x) for x in out), default=0)
    arr = np.full((len(out), width), np.nan)
    for i, x in enumerate(out):
        arr[i, : len(x)] = x
    return arr


def task_info_series(traj, field):
    ds = traj["obs/extra/task_info"][:]
    vals = []
    for r in ds:
        d = decode_json_row(r) or {}
        v = d.get(field)
        vals.append(np.nan if v is None else float(v))
    return np.asarray(vals)


def quat_to_R(q):
    """Rotation matrix from a scalar-first quaternion (qw, qx, qy, qz)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tcp_to_world(tcp_pose, base_pose):
    """Transform per-step TCP poses (robot-base frame) into world frame.

    tcp_pose / base_pose are (T,7) = xyz + scalar-first quat. MolmoSpaces stores
    tcp_pose relative to the robot base, but object poses in world frame, so they
    must be brought into a common frame before any distance is meaningful.
    """
    T = tcp_pose.shape[0]
    out = np.zeros((T, 3))
    for i in range(T):
        R = quat_to_R(base_pose[i, 3:7])
        out[i] = base_pose[i, :3] + R @ tcp_pose[i, :3]
    return out


def shade_phases(ax, phase):
    """Shade the background of a time-axis plot by manipulation phase."""
    if phase is None or len(phase) == 0:
        return
    bounds = np.where(np.diff(phase) != 0)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(phase)]])
    for s, e in zip(starts, ends):
        pid = int(phase[s])
        ax.axvspan(s, e, color=PHASE_COLORS[pid % len(PHASE_COLORS)], alpha=0.12, lw=0)


def phase_legend(fig):
    handles = [plt.Line2D([0], [0], color=PHASE_COLORS[i], lw=6, alpha=0.5, label=n)
               for i, n in enumerate(PHASE_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))


# ----------------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--traj", default="traj_0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    h5_path = Path(args.h5)
    out = Path(args.out) if args.out else Path("diagnostics_output") / f"pact_{h5_path.parent.name}_{args.traj}"
    out.mkdir(parents=True, exist_ok=True)

    f = h5py.File(h5_path, "r")
    t = f[args.traj]
    T = t["rewards"].shape[0]
    steps = np.arange(T)

    # --- scalars / metadata
    scene = json.loads(np.asarray(t["obs_scene"]).item())
    task_desc = scene.get("task_description", "?")
    coll = scene.get("collision_metrics", {})
    per_step_contacts = np.asarray(coll.get("per_step_contacts", [0] * T), dtype=float)
    if len(per_step_contacts) != T:  # be robust
        per_step_contacts = np.resize(per_step_contacts, T)

    # --- proprioception / actions
    qpos_arm = decode_field(t["obs/agent"], "qpos", "arm")      # (T,7)
    qvel_arm = decode_field(t["obs/agent"], "qvel", "arm")      # (T,7)
    grip_q = decode_field(t["obs/agent"], "qpos", "gripper")    # (T,2)
    act_arm = decode_field(t["actions"], "joint_pos", "arm")    # (T,7)
    act_grip = decode_field(t["actions"], "joint_pos", "gripper")  # (T,1)

    # --- extras
    tcp = t["obs/extra/tcp_pose"][:]            # (T,7) xyz + quat, ROBOT-BASE frame
    base_pose = t["obs/extra/robot_base_pose"][:]   # (T,7) world frame
    tcp_w = tcp_to_world(tcp, base_pose)        # (T,3) TCP in WORLD frame
    obj_start = t["obs/extra/obj_start"][:][0, :3]  # world frame
    phase = t["obs/extra/policy_phase"][:].astype(int)
    rewards = t["rewards"][:]
    success = t["success"][:].astype(bool)
    fail = t["fail"][:].astype(bool)
    terminated = t["terminated"][:].astype(bool)
    robot_contact = task_info_series(t, "robot_contact")
    position_error = task_info_series(t, "position_error")

    # --- proximity (29 sensors) -> per-step mean & min (valid) depth
    prox_keys = sorted(t["obs/proximity"].keys(),
                       key=lambda k: (int(k.split("_")[0][4:]), int(k.split("_")[-1])))
    n_sensors = len(prox_keys)
    prox_mean = np.zeros((T, n_sensors))   # mean depth (clipped) per sensor per step
    prox_min = np.full((T, n_sensors), np.nan)  # closest valid return per sensor per step
    prox_raw = {}
    for j, k in enumerate(prox_keys):
        px = t["obs/proximity"][k][:]            # (T,4,8,8)
        prox_raw[k] = px
        flat = px.reshape(T, -1)
        prox_mean[:, j] = np.clip(flat, 0, PROX_CLIP_M).mean(axis=1)
        valid = np.where((flat > PROX_MIN_VALID) & (flat <= PROX_CLIP_M), flat, np.nan)
        with np.errstate(all="ignore"):
            prox_min[:, j] = np.nanmin(valid, axis=1)
    closest_per_step = np.nanmin(prox_min, axis=1)  # nearest surface seen by ANY sensor

    tcp_to_obj = np.linalg.norm(tcp_w - obj_start[None, :], axis=1)  # world-frame distance

    # ---------- collision probability ----------
    env_collision = per_step_contacts > 0           # my metric (excludes held object)
    p_env = float(env_collision.mean())
    p_anycontact = float(np.nanmean(robot_contact > 0)) if np.isfinite(robot_contact).any() else float("nan")
    # per-phase env-collision probability
    per_phase_p = {}
    for pid in range(len(PHASE_NAMES)):
        m = phase == pid
        if m.any():
            per_phase_p[PHASE_NAMES[pid]] = float(env_collision[m].mean())

    plots = []

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        plots.append(name)

    # ---- 01 arm qpos ----
    fig, ax = plt.subplots(figsize=(13, 5))
    shade_phases(ax, phase)
    for j in range(qpos_arm.shape[1]):
        ax.plot(steps, qpos_arm[:, j], label=f"joint {j+1}", lw=1.3)
    ax.set(title=f"Arm joint positions (qpos) — {task_desc}", xlabel="policy step", ylabel="rad")
    ax.legend(ncol=7, fontsize=8); phase_legend(fig); save(fig, "01_qpos_arm.png")

    # ---- 02 arm qvel ----
    fig, ax = plt.subplots(figsize=(13, 4.5))
    shade_phases(ax, phase)
    for j in range(qvel_arm.shape[1]):
        ax.plot(steps, qvel_arm[:, j], lw=1.0, label=f"joint {j+1}")
    ax.set(title="Arm joint velocities (qvel)", xlabel="policy step", ylabel="rad/s")
    ax.legend(ncol=7, fontsize=8); save(fig, "02_qvel_arm.png")

    # ---- 03 gripper ----
    fig, ax = plt.subplots(figsize=(13, 4))
    shade_phases(ax, phase)
    ax.plot(steps, grip_q[:, 0], label="gripper finger 0 (qpos)", lw=1.5)
    if grip_q.shape[1] > 1:
        ax.plot(steps, grip_q[:, 1], label="gripper finger 1 (qpos)", lw=1.0, ls="--")
    if act_grip.shape[1] >= 1:
        ax.plot(steps, act_grip[:, 0], label="gripper command", lw=1.2, color="k", alpha=0.6)
    ax.set(title="Gripper position & command", xlabel="policy step", ylabel="opening")
    ax.legend(); save(fig, "03_gripper.png")

    # ---- 04 commanded vs realized arm joints ----
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    for j in range(7):
        ax = axes[j]; shade_phases(ax, phase)
        ax.plot(steps, qpos_arm[:, j], label="qpos", lw=1.3)
        ax.plot(steps, act_arm[:, j], label="commanded", lw=1.0, ls="--", color="k", alpha=0.7)
        ax.set_title(f"joint {j+1}", fontsize=9); ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8); axes[7].axis("off")
    fig.suptitle("Commanded action vs. realized joint position", y=0.995)
    save(fig, "04_action_vs_qpos.png")

    # ---- 05 TCP 3D + projections ----
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    cols = PHASE_COLORS[np.clip(phase, 0, len(PHASE_COLORS) - 1)]
    ax3d.scatter(tcp_w[:, 0], tcp_w[:, 1], tcp_w[:, 2], c=cols, s=8)
    ax3d.scatter(*obj_start, c="red", s=120, marker="*", label="object start")
    ax3d.set(title="TCP trajectory — world frame (3D)", xlabel="x", ylabel="y", zlabel="z")
    ax3d.legend(fontsize=8)
    for k, (a, b, lab) in enumerate([(0, 1, "x-y"), (0, 2, "x-z")]):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.scatter(tcp_w[:, a], tcp_w[:, b], c=cols, s=8)
        ax.scatter(obj_start[a], obj_start[b], c="red", s=120, marker="*")
        ax.set(title=f"TCP {lab} (world)", xlabel=lab.split("-")[0], ylabel=lab.split("-")[1])
        ax.axis("equal")
    save(fig, "05_tcp_trajectory.png")

    # ---- 06 tcp xyz + distance to object ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    shade_phases(a1, phase)
    for i, lab in enumerate("xyz"):
        a1.plot(steps, tcp_w[:, i], label=f"tcp {lab}", lw=1.3)
    a1.set(title="TCP position (world frame)", ylabel="m"); a1.legend(ncol=3)
    shade_phases(a2, phase)
    a2.plot(steps, tcp_to_obj, color="purple", lw=1.5, label="‖tcp − object_start‖ (world)")
    a2.plot(steps, closest_per_step, color="teal", lw=1.2, alpha=0.8, label="nearest skin reading (any sensor)")
    a2.set(title="Distances", xlabel="policy step", ylabel="m"); a2.legend()
    save(fig, "06_tcp_distances.png")

    # ---- 07 phases timeline + durations ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    a1.step(steps, phase, where="post", color="k", lw=1)
    for s in range(T):
        a1.axvspan(s, s + 1, color=PHASE_COLORS[phase[s] % len(PHASE_COLORS)], alpha=0.5, lw=0)
    a1.set(title="Manipulation phase over time", xlabel="policy step",
           yticks=range(len(PHASE_NAMES)), yticklabels=PHASE_NAMES)
    a1.tick_params(labelsize=8)
    durs = np.array([(phase == i).sum() for i in range(len(PHASE_NAMES))])
    a2.barh(range(len(PHASE_NAMES)), durs, color=PHASE_COLORS)
    a2.set(title="Phase duration (steps)", yticks=range(len(PHASE_NAMES)), yticklabels=PHASE_NAMES)
    a2.tick_params(labelsize=8)
    save(fig, "07_phases.png")

    # ---- 08 proximity per-sensor mean over time (heatmap) ----
    order = np.argsort([ (int(k.split('_')[0][4:]), int(k.split('_')[-1])) for k in prox_keys ])
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(prox_mean.T, aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=[0, T, n_sensors, 0], vmin=0, vmax=PROX_CLIP_M)
    ax.set(title="Per-sensor mean proximity depth (clipped 0–4 m)", xlabel="policy step",
           yticks=np.arange(n_sensors) + 0.5, yticklabels=prox_keys)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, label="depth (m)")
    save(fig, "08_proximity_mean_heatmap.png")

    # ---- 09 proximity closeness (min valid depth) heatmap ----
    fig, ax = plt.subplots(figsize=(13, 7))
    disp = np.where(np.isfinite(prox_min.T), prox_min.T, PROX_CLIP_M)
    im = ax.imshow(disp, aspect="auto", cmap="magma_r", interpolation="nearest",
                   extent=[0, T, n_sensors, 0], vmin=PROX_MIN_VALID, vmax=PROX_CLIP_M)
    ax.set(title="Per-sensor NEAREST surface (min valid depth) — dark = something close",
           xlabel="policy step", yticks=np.arange(n_sensors) + 0.5, yticklabels=prox_keys)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, label="nearest depth (m)")
    save(fig, "09_proximity_closest_heatmap.png")

    # ---- 10 proximity 8x8 montage at closest-approach step ----
    t_close = int(np.nanargmin(closest_per_step)) if np.isfinite(closest_per_step).any() else T // 2
    fig, axes = plt.subplots(5, 6, figsize=(13, 11))
    for idx, k in enumerate(prox_keys):
        ax = axes.ravel()[idx]
        frame = np.clip(prox_raw[k][t_close].mean(axis=0), 0, PROX_CLIP_M)  # mean over 4 substeps
        ax.imshow(frame, cmap="magma_r", vmin=PROX_MIN_VALID, vmax=PROX_CLIP_M)
        ax.set_title(k, fontsize=6); ax.axis("off")
    for j in range(len(prox_keys), axes.size):
        axes.ravel()[j].axis("off")
    fig.suptitle(f"All 29 proximity sensors (8×8 depth) at closest-approach step t={t_close}", y=0.995)
    save(fig, "10_proximity_montage_closest.png")

    # ---- 11 collision metric + probability ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    shade_phases(a1, phase)
    a1.bar(steps, per_step_contacts, width=1.0, color="crimson", label="env contacts (collision_metrics)")
    if np.isfinite(robot_contact).any():
        a1.plot(steps, robot_contact, color="navy", lw=1.0, alpha=0.7, label="task_info.robot_contact (incl. held obj)")
    a1.set(title=f"Collisions over time — env-collision prob = {p_env:.1%}  |  any-contact prob = {p_anycontact:.1%}",
           ylabel="count / flag"); a1.legend()
    shade_phases(a2, phase)
    a2.plot(steps, closest_per_step, color="teal", lw=1.3, label="nearest skin reading (m)")
    a2.axhline(PROX_MIN_VALID, color="red", ls="--", lw=0.8, label="SPAD near limit 0.05 m")
    a2.set(title="Nearest surface seen by the skin (collision proximity)", xlabel="policy step", ylabel="m")
    a2.legend(); save(fig, "11_collision_metric.png")

    # ---- 12 per-phase collision probability ----
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(per_phase_p.keys()); vals = [per_phase_p[n] for n in names]
    ax.bar(names, vals, color=[PHASE_COLORS[PHASE_NAMES.index(n)] for n in names])
    ax.set(title="Env-collision probability by phase", ylabel="P(collision step)")
    ax.set_ylim(0, max(vals + [0.01]) * 1.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    save(fig, "12_collision_prob_by_phase.png")

    # ---- 13 reward / success / position error ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    shade_phases(a1, phase)
    a1.plot(steps, rewards, color="green", lw=1.2, label="reward")
    a1.plot(steps, position_error, color="orange", lw=1.2, label="position_error (m)")
    a1.legend(); a1.set(title="Reward & position error", ylabel="value")
    shade_phases(a2, phase)
    a2.plot(steps, success.astype(int), label="success", lw=1.5)
    a2.plot(steps, fail.astype(int), label="fail", lw=1.0, ls="--")
    a2.plot(steps, terminated.astype(int), label="terminated", lw=1.0, alpha=0.6)
    a2.set(title="Episode flags", xlabel="policy step", ylabel="bool"); a2.legend()
    save(fig, "13_reward_success.png")

    # ---------- summary.json + report.md ----------
    summary = {
        "h5": str(h5_path), "traj": args.traj, "task_description": task_desc,
        "n_steps": int(T), "n_proximity_sensors": int(n_sensors),
        "success_final": bool(success[-1]), "terminated_final": bool(terminated[-1]),
        "fail_any": bool(fail.any()),
        "phase_durations": {PHASE_NAMES[i]: int((phase == i).sum()) for i in range(len(PHASE_NAMES))},
        "collision": {
            "collided": bool(coll.get("collided", env_collision.any())),
            "n_collision_steps": int(env_collision.sum()),
            "total_contacts": int(per_step_contacts.sum()),
            "env_collision_probability": p_env,
            "any_contact_probability": p_anycontact,
            "per_phase_env_collision_probability": per_phase_p,
            "collision_steps_idx": np.where(env_collision)[0].tolist(),
        },
        "proximity": {
            "per_sensor_mean_depth_range": [float(prox_mean.mean(0).min()), float(prox_mean.mean(0).max())],
            "global_nearest_surface_m": float(np.nanmin(closest_per_step)) if np.isfinite(closest_per_step).any() else None,
            "closest_approach_step": int(t_close),
            "nonzero_fraction": float(np.mean([(prox_raw[k] > 0).mean() for k in prox_keys])),
        },
        "tcp_to_object_start_m": {"start": float(tcp_to_obj[0]), "min": float(tcp_to_obj.min()), "end": float(tcp_to_obj[-1])},
        "plots": plots,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        f"# PACT trajectory diagnostics — `{args.traj}`", "",
        f"**Source:** `{h5_path}`  ", f"**Task:** {task_desc}  ",
        f"**Steps:** {T} · **Proximity sensors:** {n_sensors} · "
        f"**Final success:** {bool(success[-1])} · terminated: {bool(terminated[-1])}", "",
        "## Collision",
        f"- **Environment-collision probability** (collision_metrics, excludes held object): "
        f"**{p_env:.1%}** ({int(env_collision.sum())}/{T} steps; collisions at steps {np.where(env_collision)[0].tolist()})",
        f"- **Any-contact probability** (task_info.robot_contact, includes the grasped object): **{p_anycontact:.1%}**",
        f"- Nearest surface seen by the skin over the episode: **{summary['proximity']['global_nearest_surface_m']:.3f} m** "
        f"(at step {t_close})", "",
        "## Proximity health",
        f"- nonzero-pixel fraction: {summary['proximity']['nonzero_fraction']:.4f} (≈1.0 = sensors recording every step)",
        f"- per-sensor mean depth range: {summary['proximity']['per_sensor_mean_depth_range'][0]:.2f}–"
        f"{summary['proximity']['per_sensor_mean_depth_range'][1]:.2f} m (clipped to 4 m for display)", "",
        "## Phase durations (steps)",
        *[f"- {n}: {int((phase==i).sum())}" for i, n in enumerate(PHASE_NAMES) if (phase == i).any()],
        "", "## Figures", *[f"- `{p}`" for p in plots],
    ]
    (out / "report.md").write_text("\n".join(report))

    print(f"[done] wrote {len(plots)} plots + summary.json + report.md to {out}")
    print(f"  env-collision probability: {p_env:.1%} | any-contact probability: {p_anycontact:.1%}")
    print(f"  final success: {bool(success[-1])} | nearest skin reading: {summary['proximity']['global_nearest_surface_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
