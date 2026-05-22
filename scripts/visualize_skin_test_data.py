"""Cross-folder visualization of pick_and_place_skin_test_1_house.

Grouped by episode index N (0..9). For each N we collect that trajectory from
every timestamp folder and produce ONE plot per data type, overlaying or tiling
folders together. Output files (flat layout):

  eval_output/skin_test_1_house_viz/
    episode_N_joint_positions.png       (8 subplots, one curve per folder)
    episode_N_joint_velocities.png      (8 subplots, one curve per folder)
    episode_N_proximity_distance.png    (29 subplots, one curve per folder)
    episode_N_rgbd.png                  (folder rows x 6 timesteps, exo+wrist)
    episode_N_proximity_heatmaps.png    (per-sensor tile: folder rows x 6 timesteps,
                                         with shared depth colorbar)

For overlays (joints, proximity distance), each folder gets a unique color and the
legend appears in the first subplot. For tiles (RGBD, heatmaps), folder name is on
the left of each row block.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

DATA_ROOT = Path(
    "assets/datagen/pick_and_place_skin_test_1_house/"
    "FrankaSkinPickAndPlacePilotMediumConfig"
)
OUT_DIR = Path("eval_output/skin_test_1_house_viz")

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

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
DEPTH_CMAP = "viridis_r"  # bright = close, dark = far
SUBSTEP_INDEX = 3


def list_folders() -> list[Path]:
    folders = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    return [d for d in folders if (d / "house_1" / "trajectories_batch_1_of_1.h5").exists()]


def load_episode(h5_path: Path, traj_key: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        n_steps = f[f"{traj_key}/obs/agent/qpos"].shape[0]
        qpos_all = np.zeros((n_steps, 9))
        qvel_all = np.zeros((n_steps, 9))
        for t in range(n_steps):
            raw = f[f"{traj_key}/obs/agent/qpos"][t]
            data = json.loads(raw.tobytes().decode("utf-8").rstrip("\x00"))
            qpos_all[t, :7] = data["arm"]
            qpos_all[t, 7:9] = data["gripper"]
            raw_v = f[f"{traj_key}/obs/agent/qvel"][t]
            data_v = json.loads(raw_v.tobytes().decode("utf-8").rstrip("\x00"))
            qvel_all[t, :7] = data_v["arm"]
            qvel_all[t, 7:9] = data_v["gripper"]
        prox = {}
        for sn in SENSOR_NAMES:
            key = f"{traj_key}/obs/proximity/{sn}"
            if key in f:
                prox[sn] = f[key][:]
    return {"qpos": qpos_all, "qvel": qvel_all, "proximity": prox}


def sample_video_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
    cap.release()
    return frames


def _closest_distance(arr: np.ndarray) -> np.ndarray:
    sub = arr[:, SUBSTEP_INDEX, :, :]
    valid = sub > 1e-6
    flat = np.where(valid, sub, np.inf)
    closest = flat.reshape(flat.shape[0], -1).min(axis=1)
    closest = np.where(np.isfinite(closest), closest, np.nan)
    return closest


def plot_joints_overlay(eps_by_folder: dict[str, dict], key: str, ep_index: int, out_path: Path) -> None:
    n = len(eps_by_folder)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=False)
    axes = axes.flatten()
    for ji in range(7):
        ax = axes[ji]
        for fi, (folder, ep) in enumerate(eps_by_folder.items()):
            arr = ep[key][:, ji]
            ax.plot(np.arange(len(arr)), arr, color=colors[fi], linewidth=1.2, alpha=0.85, label=folder)
        ax.set_title(JOINT_NAMES[ji], fontsize=11)
        ax.set_xlabel("timestep")
        ax.set_ylabel("position (rad)" if key == "qpos" else "velocity (rad/s)")
        ax.grid(True, alpha=0.3)
    ax = axes[7]
    for fi, (folder, ep) in enumerate(eps_by_folder.items()):
        arr_l = ep[key][:, 7]
        arr_r = ep[key][:, 8]
        t = np.arange(len(arr_l))
        ax.plot(t, arr_l, color=colors[fi], linewidth=1.2, alpha=0.85, label=folder)
        ax.plot(t, arr_r, color=colors[fi], linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_title("gripper (solid=left, dashed=right)", fontsize=11)
    ax.set_xlabel("timestep")
    ax.set_ylabel("position (m)" if key == "qpos" else "velocity (m/s)")
    ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best", ncol=1, framealpha=0.85)
    suffix = "Positions" if key == "qpos" else "Velocities"
    fig.suptitle(
        f"Episode {ep_index} — Joint {suffix} (overlay of {n} folders; "
        f"all use seed=2026 so traces are bit-identical → single visible line)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_overlay(eps_by_folder: dict[str, dict], ep_index: int, out_path: Path) -> None:
    n = len(eps_by_folder)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))
    n_sensors = len(SENSOR_NAMES)
    cols = 6
    rows = (n_sensors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), sharey=True)
    axes = axes.flatten()
    for si, sn in enumerate(SENSOR_NAMES):
        ax = axes[si]
        for fi, (folder, ep) in enumerate(eps_by_folder.items()):
            if sn not in ep["proximity"]:
                continue
            data = ep["proximity"][sn]
            closest = _closest_distance(data)
            ax.plot(np.arange(len(closest)), closest, color=colors[fi], linewidth=0.9, alpha=0.8,
                    label=folder if si == 0 else None)
        ax.set_title(sn.replace("_sensor_", " s"), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.3)
        if si % cols == 0:
            ax.set_ylabel("dist (m)", fontsize=7)
    for si in range(n_sensors, len(axes)):
        axes[si].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=7, ncol=2, framealpha=0.85)
    fig.suptitle(
        f"Episode {ep_index} — Closest detected distance per sensor "
        f"(substep {SUBSTEP_INDEX}, zero-pixels masked, ylim 0-2m, overlay of {n} folders; "
        f"all use seed=2026 so traces are bit-identical → single visible line)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_rgbd_tile(folder_data: list[tuple[str, Path, int, int]], ep_index: int, out_path: Path) -> None:
    """All 4 cameras x 6 timesteps for the (representative) first folder.

    Since all folders share seed=2026 and produce bit-identical episodes, we render
    one copy and note this in the title rather than tiling N redundant copies.
    """
    n_folders = len(folder_data)
    folder, h1, traj_idx, T = folder_data[0]
    n_times = 6
    t_idxs = np.linspace(0, T - 1, n_times, dtype=int).tolist()
    cams = [
        ("Exo RGB",   h1 / f"episode_{traj_idx:08d}_exo_camera_1_batch_1_of_1.mp4"),
        ("Exo Depth", h1 / f"episode_{traj_idx:08d}_exo_camera_1_depth_batch_1_of_1.mp4"),
        ("Wrist RGB", h1 / f"episode_{traj_idx:08d}_wrist_camera_batch_1_of_1.mp4"),
        ("Wrist Depth", h1 / f"episode_{traj_idx:08d}_wrist_camera_depth_batch_1_of_1.mp4"),
    ]
    fig, axes = plt.subplots(len(cams), n_times, figsize=(n_times * 2.5, len(cams) * 2.0),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    for ri, (label, path) in enumerate(cams):
        frames = sample_video_frames(path, t_idxs) if path.exists() else [
            np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
        for ci, frame in enumerate(frames):
            ax = axes[ri, ci]
            ax.imshow(frame)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t_idxs[ci]}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, va="center")
    fig.suptitle(
        f"Episode {ep_index} — RGBD frames  ({folder}; "
        f"identical across {n_folders} folders that share seed=2026)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_heatmap_tile(
    eps_by_folder: dict[str, dict], folder_meta: dict[str, tuple[Path, int]],
    ep_index: int, out_path: Path,
) -> None:
    """Cameras (exo+wrist RGB) + 29 sensor heatmaps at 6 timesteps for the first folder.

    Since all folders share seed=2026 and produce bit-identical episodes, we render
    one copy with a note in the title.
    """
    folders = list(eps_by_folder.keys())
    n_folders = len(folders)
    folder = folders[0]
    h1, traj_idx = folder_meta[folder]
    ep = eps_by_folder[folder]
    T = ep["qpos"].shape[0]
    n_times = 6
    n_sensors = len(SENSOR_NAMES)
    t_idxs = np.linspace(0, T - 1, n_times, dtype=int)

    camera_h = 3.0
    sensor_h = 1.0
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

    # Camera rows
    exo_path = h1 / f"episode_{traj_idx:08d}_exo_camera_1_batch_1_of_1.mp4"
    wri_path = h1 / f"episode_{traj_idx:08d}_wrist_camera_batch_1_of_1.mp4"
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

    # Sensor rows
    cmap = mpl.colormaps[DEPTH_CMAP].with_extremes(bad="black")
    norm = mpl.colors.Normalize(vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
    for si, sn in enumerate(SENSOR_NAMES):
        row = 2 + si
        prox = ep["proximity"].get(sn)
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

    fig.suptitle(
        f"Episode {ep_index} — Proximity heatmaps + RGB context  ({folder}; "
        f"identical across {n_folders} folders that share seed=2026)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = list_folders()
    print(f"Found {len(folders)} folders")

    # Discover the set of episode indices common to all (or at least union)
    folder_traj_keys: dict[str, list[str]] = {}
    for folder in folders:
        h5_path = folder / "house_1" / "trajectories_batch_1_of_1.h5"
        with h5py.File(h5_path, "r") as f:
            folder_traj_keys[folder.name] = sorted(k for k in f.keys() if k.startswith("traj_"))
    max_ep = max(len(v) for v in folder_traj_keys.values())
    print(f"Episode count per folder: {[len(v) for v in folder_traj_keys.values()]}; processing 0..{max_ep-1}")

    for ep_index in range(max_ep):
        traj_key = f"traj_{ep_index}"
        eps: dict[str, dict] = {}
        folder_meta: dict[str, tuple[Path, int]] = {}
        folder_data_for_rgbd: list[tuple[str, Path, int, int]] = []
        for folder in folders:
            if traj_key not in folder_traj_keys[folder.name]:
                continue
            h1 = folder / "house_1"
            h5_path = h1 / "trajectories_batch_1_of_1.h5"
            ep = load_episode(h5_path, traj_key)
            eps[folder.name] = ep
            folder_meta[folder.name] = (h1, ep_index)
            folder_data_for_rgbd.append((folder.name, h1, ep_index, ep["qpos"].shape[0]))
        if not eps:
            continue
        print(f"[ep {ep_index}] {len(eps)} folders")

        plot_joints_overlay(eps, "qpos", ep_index, OUT_DIR / f"episode_{ep_index}_joint_positions.png")
        plot_joints_overlay(eps, "qvel", ep_index, OUT_DIR / f"episode_{ep_index}_joint_velocities.png")
        plot_proximity_overlay(eps, ep_index, OUT_DIR / f"episode_{ep_index}_proximity_distance.png")
        plot_rgbd_tile(folder_data_for_rgbd, ep_index, OUT_DIR / f"episode_{ep_index}_rgbd.png")
        plot_proximity_heatmap_tile(eps, folder_meta, ep_index, OUT_DIR / f"episode_{ep_index}_proximity_heatmaps.png")
        print(f"  ep_{ep_index} done")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
