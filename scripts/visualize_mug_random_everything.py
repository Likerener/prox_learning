"""Per-folder visualization + wandb upload for mug_house_1_random_everything.

Each timestamp folder in
  assets/datagen/mug_house_1_random_everything/FrankaSkinPickAndPlacePilotMediumConfig/
contains ONE episode (traj_0). For every completed folder we render the same five
plots as ``visualize_skin_test_data.py`` (joint positions, joint velocities,
proximity distances, RGBD tile, proximity heatmap tile), upload them to wandb, and
periodically refresh a cross-episode overlay summary.

Usage:
    python scripts/visualize_mug_random_everything.py              # one-shot pass
    python scripts/visualize_mug_random_everything.py --watch      # poll for new folders
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


DATA_ROOT = Path(
    "/home/jaydv/code/prox_learning/assets/datagen/mug_house_1_random_everything/"
    "FrankaSkinPickAndPlacePilotMediumConfig"
)
OUT_ROOT = Path("/home/jaydv/code/prox_learning/eval_output/mug_random_everything_viz")
SUMMARY_DIR = OUT_ROOT / "_summary"

JOINT_NAMES = [f"joint_{i+1}" for i in range(7)]
SENSOR_NAMES = [
    "link2_sensor_0", "link2_sensor_1", "link2_sensor_2", "link2_sensor_3",
    "link2_sensor_4", "link2_sensor_5", "link2_sensor_6",
    "link3_sensor_0", "link3_sensor_1", "link3_sensor_2", "link3_sensor_3",
    "link3_sensor_4", "link3_sensor_5", "link3_sensor_6", "link3_sensor_7",
    "link5_sensor_0", "link5_sensor_1", "link5_sensor_2", "link5_sensor_3",
    "link5_sensor_4", "link5_sensor_5",
    "link6_sensor_0", "link6_sensor_1", "link6_sensor_2", "link6_sensor_3",
    "link6_sensor_4", "link6_sensor_5", "link6_sensor_6", "link6_sensor_7",
]

DEPTH_VMIN = 0.0
DEPTH_VMAX = 1.0
DEPTH_CMAP = "viridis_r"
SUBSTEP_INDEX = 3

REQUIRED_FILES = [
    "trajectories_batch_1_of_1.h5",
    "episode_00000000_exo_camera_1_batch_1_of_1.mp4",
    "episode_00000000_exo_camera_1_depth_batch_1_of_1.mp4",
    "episode_00000000_wrist_camera_batch_1_of_1.mp4",
    "episode_00000000_wrist_camera_depth_batch_1_of_1.mp4",
]

PLOT_KEYS = ["joint_positions", "joint_velocities", "proximity_distance",
             "rgbd", "proximity_heatmaps"]


# ---------- I/O ----------

@dataclass
class EpisodeBundle:
    folder: str
    h5_path: Path
    house_dir: Path
    qpos: np.ndarray
    qvel: np.ndarray
    proximity: dict[str, np.ndarray]
    T: int


def folder_is_complete(folder: Path) -> bool:
    h1 = folder / "house_1"
    if not h1.is_dir():
        return False
    for fn in REQUIRED_FILES:
        if not (h1 / fn).exists():
            return False
    return True


def list_complete_folders() -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    return sorted(d for d in DATA_ROOT.iterdir() if d.is_dir() and folder_is_complete(d))


def load_episode(folder: Path) -> EpisodeBundle | None:
    h1 = folder / "house_1"
    h5_path = h1 / "trajectories_batch_1_of_1.h5"
    try:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            if not traj_keys:
                return None
            traj_key = traj_keys[0]
            n_steps = f[f"{traj_key}/obs/agent/qpos"].shape[0]
            qpos = np.zeros((n_steps, 9), dtype=np.float32)
            qvel = np.zeros((n_steps, 9), dtype=np.float32)
            for t in range(n_steps):
                raw = f[f"{traj_key}/obs/agent/qpos"][t]
                data = json.loads(raw.tobytes().decode("utf-8").rstrip("\x00"))
                qpos[t, :7] = data["arm"]
                qpos[t, 7:9] = data["gripper"]
                raw_v = f[f"{traj_key}/obs/agent/qvel"][t]
                data_v = json.loads(raw_v.tobytes().decode("utf-8").rstrip("\x00"))
                qvel[t, :7] = data_v["arm"]
                qvel[t, 7:9] = data_v["gripper"]
            prox = {}
            for sn in SENSOR_NAMES:
                key = f"{traj_key}/obs/proximity/{sn}"
                if key in f:
                    prox[sn] = f[key][:]
    except (OSError, KeyError, json.JSONDecodeError) as e:
        print(f"  ! load failed for {folder.name}: {e}")
        return None
    return EpisodeBundle(
        folder=folder.name, h5_path=h5_path, house_dir=h1,
        qpos=qpos, qvel=qvel, proximity=prox, T=n_steps,
    )


def sample_video_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                      else np.zeros((240, 320, 3), dtype=np.uint8))
    cap.release()
    return frames


def closest_distance(arr: np.ndarray) -> np.ndarray:
    sub = arr[:, SUBSTEP_INDEX, :, :]
    valid = sub > 1e-6
    flat = np.where(valid, sub, np.inf)
    closest = flat.reshape(flat.shape[0], -1).min(axis=1)
    return np.where(np.isfinite(closest), closest, np.nan)


# ---------- per-episode plots ----------

def plot_joints_single(ep: EpisodeBundle, key: str, out_path: Path) -> None:
    arr = ep.qpos if key == "qpos" else ep.qvel
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    t = np.arange(arr.shape[0])
    for ji in range(7):
        ax = axes[ji]
        ax.plot(t, arr[:, ji], color="tab:blue", linewidth=1.4)
        ax.set_title(JOINT_NAMES[ji], fontsize=11)
        ax.set_xlabel("timestep"); ax.grid(True, alpha=0.3)
        ax.set_ylabel("position (rad)" if key == "qpos" else "velocity (rad/s)")
    ax = axes[7]
    ax.plot(t, arr[:, 7], color="tab:blue", linewidth=1.4, label="left")
    ax.plot(t, arr[:, 8], color="tab:orange", linewidth=1.2, linestyle="--", label="right")
    ax.set_title("gripper", fontsize=11)
    ax.set_xlabel("timestep"); ax.grid(True, alpha=0.3)
    ax.set_ylabel("position (m)" if key == "qpos" else "velocity (m/s)")
    ax.legend(fontsize=8)
    suffix = "Positions" if key == "qpos" else "Velocities"
    fig.suptitle(f"{ep.folder} — Joint {suffix}  (T={ep.T})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_single(ep: EpisodeBundle, out_path: Path) -> None:
    n_sensors = len(SENSOR_NAMES)
    cols = 6
    rows = (n_sensors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), sharey=True)
    axes = axes.flatten()
    for si, sn in enumerate(SENSOR_NAMES):
        ax = axes[si]
        if sn in ep.proximity:
            closest = closest_distance(ep.proximity[sn])
            ax.plot(np.arange(len(closest)), closest, color="tab:blue", linewidth=1.0)
        ax.set_title(sn.replace("_sensor_", " s"), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 2.0); ax.grid(True, alpha=0.3)
        if si % cols == 0:
            ax.set_ylabel("dist (m)", fontsize=7)
    for si in range(n_sensors, len(axes)):
        axes[si].axis("off")
    fig.suptitle(
        f"{ep.folder} — Closest detected distance per sensor "
        f"(substep {SUBSTEP_INDEX}, zero-pixels masked, T={ep.T})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_rgbd_tile(ep: EpisodeBundle, out_path: Path) -> None:
    n_times = 6
    t_idxs = np.linspace(0, ep.T - 1, n_times, dtype=int).tolist()
    cams = [
        ("Exo RGB",   ep.house_dir / "episode_00000000_exo_camera_1_batch_1_of_1.mp4"),
        ("Exo Depth", ep.house_dir / "episode_00000000_exo_camera_1_depth_batch_1_of_1.mp4"),
        ("Wrist RGB", ep.house_dir / "episode_00000000_wrist_camera_batch_1_of_1.mp4"),
        ("Wrist Depth", ep.house_dir / "episode_00000000_wrist_camera_depth_batch_1_of_1.mp4"),
    ]
    fig, axes = plt.subplots(len(cams), n_times, figsize=(n_times * 2.5, len(cams) * 2.0),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    for ri, (label, path) in enumerate(cams):
        frames = sample_video_frames(path, t_idxs) if path.exists() else [
            np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
        for ci, frame in enumerate(frames):
            ax = axes[ri, ci]
            ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t_idxs[ci]}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, va="center")
    fig.suptitle(f"{ep.folder} — RGBD frames  (T={ep.T})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_heatmap_tile(ep: EpisodeBundle, out_path: Path) -> None:
    n_times = 6
    n_sensors = len(SENSOR_NAMES)
    t_idxs = np.linspace(0, ep.T - 1, n_times, dtype=int)
    camera_h, sensor_h = 3.0, 1.0
    height_ratios = [camera_h, camera_h] + [sensor_h] * n_sensors
    total_h_units = sum(height_ratios)
    fig_w = n_times * 2.0 + 2.5
    fig_h = total_h_units * 0.55 + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=2 + n_sensors, ncols=n_times + 1,
        height_ratios=height_ratios,
        width_ratios=[1.0] * n_times + [0.08],
        wspace=0.06, hspace=0.18,
        left=0.10, right=0.94, top=0.96, bottom=0.02,
    )
    exo_path = ep.house_dir / "episode_00000000_exo_camera_1_batch_1_of_1.mp4"
    wri_path = ep.house_dir / "episode_00000000_wrist_camera_batch_1_of_1.mp4"
    exo_frames = sample_video_frames(exo_path, t_idxs.tolist()) if exo_path.exists() else [
        np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
    wri_frames = sample_video_frames(wri_path, t_idxs.tolist()) if wri_path.exists() else [
        np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
    for ci, frame in enumerate(exo_frames):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t_idxs[ci]}", fontsize=10)
        if ci == 0:
            ax.set_ylabel("Exo RGB", fontsize=10, rotation=90, va="center")
    for ci, frame in enumerate(wri_frames):
        ax = fig.add_subplot(gs[1, ci])
        ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("Wrist RGB", fontsize=10, rotation=90, va="center")
    cmap = mpl.colormaps[DEPTH_CMAP].with_extremes(bad="black")
    norm = mpl.colors.Normalize(vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
    for si, sn in enumerate(SENSOR_NAMES):
        row = 2 + si
        prox = ep.proximity.get(sn)
        short = sn.replace("link", "L").replace("_sensor_", ".s")
        for ti, t in enumerate(t_idxs):
            ax = fig.add_subplot(gs[row, ti])
            if prox is None:
                ax.axis("off")
                continue
            img = prox[t, SUBSTEP_INDEX]
            masked = np.ma.masked_where(img <= 1e-6, img)
            ax.imshow(masked, vmin=DEPTH_VMIN, vmax=DEPTH_VMAX, cmap=cmap,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if ti == 0:
                ax.set_ylabel(short, fontsize=9, rotation=0, ha="right", va="center", labelpad=6)
    cbar_ax = fig.add_subplot(gs[2:, -1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("sensor depth (m)", fontsize=10, labelpad=6)
    cb.ax.tick_params(labelsize=9)
    cbar_ax.text(0.5, 1.02, "close", transform=cbar_ax.transAxes, fontsize=9, ha="center", va="bottom")
    cbar_ax.text(0.5, -0.04, f"far  (>{DEPTH_VMAX}m)", transform=cbar_ax.transAxes,
                 fontsize=9, ha="center", va="top")
    cbar_ax.text(1.6, 0.5, "BLACK = no return (raw 0)", transform=cbar_ax.transAxes,
                 fontsize=9, ha="left", va="center", rotation=90)
    fig.suptitle(f"{ep.folder} — Proximity heatmaps + RGB context  (T={ep.T})", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- cross-episode summary ----------

def plot_summary_joints(eps: list[EpisodeBundle], key: str, out_path: Path) -> None:
    n = len(eps)
    colors = plt.cm.viridis(np.linspace(0, 1, max(n, 2)))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ji in range(7):
        ax = axes[ji]
        for ei, ep in enumerate(eps):
            arr = (ep.qpos if key == "qpos" else ep.qvel)[:, ji]
            ax.plot(np.arange(len(arr)), arr, color=colors[ei], linewidth=0.6, alpha=0.55)
        ax.set_title(JOINT_NAMES[ji], fontsize=11)
        ax.set_xlabel("timestep"); ax.grid(True, alpha=0.3)
        ax.set_ylabel("position (rad)" if key == "qpos" else "velocity (rad/s)")
    ax = axes[7]
    for ei, ep in enumerate(eps):
        arr = ep.qpos if key == "qpos" else ep.qvel
        t = np.arange(arr.shape[0])
        ax.plot(t, arr[:, 7], color=colors[ei], linewidth=0.6, alpha=0.55)
        ax.plot(t, arr[:, 8], color=colors[ei], linewidth=0.5, alpha=0.35, linestyle="--")
    ax.set_title("gripper (solid=left, dashed=right)", fontsize=11)
    ax.set_xlabel("timestep"); ax.grid(True, alpha=0.3)
    ax.set_ylabel("position (m)" if key == "qpos" else "velocity (m/s)")
    suffix = "Positions" if key == "qpos" else "Velocities"
    fig.suptitle(f"Cross-episode summary — Joint {suffix}  ({n} episodes, viridis = chronological order)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_proximity(eps: list[EpisodeBundle], out_path: Path) -> None:
    n = len(eps)
    colors = plt.cm.viridis(np.linspace(0, 1, max(n, 2)))
    n_sensors = len(SENSOR_NAMES)
    cols = 6
    rows = (n_sensors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), sharey=True)
    axes = axes.flatten()
    for si, sn in enumerate(SENSOR_NAMES):
        ax = axes[si]
        for ei, ep in enumerate(eps):
            if sn not in ep.proximity:
                continue
            closest = closest_distance(ep.proximity[sn])
            ax.plot(np.arange(len(closest)), closest, color=colors[ei], linewidth=0.5, alpha=0.5)
        ax.set_title(sn.replace("_sensor_", " s"), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 2.0); ax.grid(True, alpha=0.3)
        if si % cols == 0:
            ax.set_ylabel("dist (m)", fontsize=7)
    for si in range(n_sensors, len(axes)):
        axes[si].axis("off")
    fig.suptitle(
        f"Cross-episode summary — Closest distance per sensor "
        f"({n} episodes overlayed, viridis = chronological order)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------

def episode_done(out_dir: Path) -> bool:
    return all((out_dir / f"{k}.png").exists() for k in PLOT_KEYS)


def render_episode(ep: EpisodeBundle) -> Path:
    out_dir = OUT_ROOT / ep.folder
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_joints_single(ep, "qpos", out_dir / "joint_positions.png")
    plot_joints_single(ep, "qvel", out_dir / "joint_velocities.png")
    plot_proximity_single(ep, out_dir / "proximity_distance.png")
    plot_rgbd_tile(ep, out_dir / "rgbd.png")
    plot_proximity_heatmap_tile(ep, out_dir / "proximity_heatmaps.png")
    return out_dir


def render_summary(eps: list[EpisodeBundle]) -> dict[str, Path]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = SUMMARY_DIR / "joint_positions_overlay.png"
    plot_summary_joints(eps, "qpos", p); paths["summary/joint_positions"] = p
    p = SUMMARY_DIR / "joint_velocities_overlay.png"
    plot_summary_joints(eps, "qvel", p); paths["summary/joint_velocities"] = p
    p = SUMMARY_DIR / "proximity_distance_overlay.png"
    plot_summary_proximity(eps, p); paths["summary/proximity_distance"] = p
    return paths


def init_wandb(args) -> object | None:
    if not args.wandb or not HAS_WANDB:
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="dataset_viz",
            config={
                "data_root": str(DATA_ROOT),
                "out_root": str(OUT_ROOT),
                "config_class": "FrankaSkinPickAndPlacePilotMediumConfig",
                "dataset": "mug_house_1_random_everything",
            },
            resume="allow",
        )
        print(f"[wandb] run url: {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] init failed: {e}; proceeding without wandb")
        return None


def log_episode_to_wandb(run, ep_idx: int, folder: str, out_dir: Path, ep: EpisodeBundle) -> None:
    if run is None:
        return
    images = {f"episode/{k}": wandb.Image(str(out_dir / f"{k}.png")) for k in PLOT_KEYS}
    metrics = {
        "episode_idx": ep_idx,
        "folder": folder,
        "episode_length": ep.T,
        "qpos_arm_range": float(np.ptp(ep.qpos[:, :7])),
        "qpos_gripper_min": float(ep.qpos[:, 7:9].min()),
        "qpos_gripper_max": float(ep.qpos[:, 7:9].max()),
    }
    run.log({**images, **metrics}, step=ep_idx)


def log_summary_to_wandb(run, paths: dict[str, Path], total: int) -> None:
    if run is None:
        return
    payload = {k: wandb.Image(str(v)) for k, v in paths.items()}
    payload["total_episodes"] = total
    run.log(payload)


def process_pending(run, eps_cache: dict[str, EpisodeBundle], summary_every: int,
                    last_summary_count: int) -> tuple[int, int]:
    """Render any folder we haven't rendered yet, refresh summary periodically."""
    folders = list_complete_folders()
    new_count = 0
    for ep_idx, folder in enumerate(folders):
        out_dir = OUT_ROOT / folder.name
        already = episode_done(out_dir) and folder.name in eps_cache
        if already:
            continue
        # Load if we don't have it cached
        if folder.name not in eps_cache:
            ep = load_episode(folder)
            if ep is None:
                continue
            eps_cache[folder.name] = ep
        ep = eps_cache[folder.name]
        if not episode_done(out_dir):
            print(f"[render] ep {ep_idx:03d}  {folder.name}  T={ep.T}")
            render_episode(ep)
            log_episode_to_wandb(run, ep_idx, folder.name, out_dir, ep)
            new_count += 1
    total = len(folders)
    if total - last_summary_count >= summary_every and total > 0:
        eps_in_order = [eps_cache[f.name] for f in folders if f.name in eps_cache]
        print(f"[summary] refreshing across {len(eps_in_order)} eps")
        paths = render_summary(eps_in_order)
        log_summary_to_wandb(run, paths, total)
        last_summary_count = total
    return new_count, last_summary_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Poll for new folders instead of single-pass")
    parser.add_argument("--poll-sec", type=int, default=30,
                        help="Polling interval in watch mode")
    parser.add_argument("--idle-exits-after", type=int, default=0,
                        help="If >0, exit after this many idle polls in watch mode")
    parser.add_argument("--summary-every", type=int, default=10,
                        help="Refresh cross-episode summary every N new episodes")
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Upload to wandb (default on)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", default="prox_learning_dataset_viz")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="mug_house1_random_everything")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"OUT_ROOT  = {OUT_ROOT}")

    run = init_wandb(args)
    eps_cache: dict[str, EpisodeBundle] = {}
    last_summary_count = 0

    if args.watch:
        idle = 0
        while True:
            new_count, last_summary_count = process_pending(
                run, eps_cache, args.summary_every, last_summary_count
            )
            if new_count == 0:
                idle += 1
                print(f"[watch] no new folders (idle {idle}); sleeping {args.poll_sec}s")
                if args.idle_exits_after and idle >= args.idle_exits_after:
                    print("[watch] idle exit threshold reached")
                    break
            else:
                idle = 0
            time.sleep(args.poll_sec)
    else:
        new_count, last_summary_count = process_pending(
            run, eps_cache, args.summary_every, last_summary_count
        )
        # Force a final summary at the end of a one-shot run
        folders = list_complete_folders()
        if folders and last_summary_count < len(folders):
            eps_in_order = [eps_cache[f.name] for f in folders if f.name in eps_cache]
            if eps_in_order:
                paths = render_summary(eps_in_order)
                log_summary_to_wandb(run, paths, len(folders))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
