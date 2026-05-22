"""Re-render the 5 episode_0 plots with N extra runs appended as additional series.

Original data: 8 bit-identical folders at seed=2026 in
  assets/datagen/pick_and_place_skin_test_1_house/...
Extra runs (each adds a distinct trace/block to the plots):
  - rand_object_z_offset  (assets/datagen/pick_and_place_skin_test_1_house_rand_object/...)
  - rand_object_textures  (assets/datagen/pick_and_place_skin_test_1_house_rand_object_textures/...)

For overlay plots (joints, proximity_distance): each extra run gets a distinct
bold color overlaid on the 8 originals.

For tile plots (rgbd, proximity_heatmaps): each extra run adds a stacked block
below the previous one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import visualize_skin_test_data as v

ORIG_ROOT = Path(
    "assets/datagen/pick_and_place_skin_test_1_house/"
    "FrankaSkinPickAndPlacePilotMediumConfig"
)
OUT_DIR = Path("eval_output/skin_test_1_house_viz")

EP_INDEX = 0
TRAJ_KEY = f"traj_{EP_INDEX}"


@dataclass
class ExtraRun:
    label: str
    root: Path
    color: str
    folder: Path = None  # set after discovery
    h1: Path = None
    ep: dict = None
    T: int = 0


EXTRA_RUNS = [
    ExtraRun(
        label="rand_object_z_offset",
        root=Path(
            "assets/datagen/pick_and_place_skin_test_1_house_rand_object/"
            "FrankaSkinPickAndPlacePilotMediumConfig"
        ),
        color="black",
    ),
    ExtraRun(
        label="rand_object_textures",
        root=Path(
            "assets/datagen/pick_and_place_skin_test_1_house_rand_object_textures/"
            "FrankaSkinPickAndPlacePilotMediumConfig"
        ),
        color="crimson",
    ),
]
EXTRA_LW = 2.0


def list_folders(root: Path) -> list[Path]:
    folders = sorted(d for d in root.iterdir() if d.is_dir())
    return [d for d in folders if (d / "house_1" / "trajectories_batch_1_of_1.h5").exists()]


# ----- overlay plots -----


def plot_joints(orig_eps: dict, extras: list[ExtraRun], key: str, out_path: Path, summary: str) -> None:
    n = len(orig_eps)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ji in range(7):
        ax = axes[ji]
        for fi, (folder, ep) in enumerate(orig_eps.items()):
            arr = ep[key][:, ji]
            ax.plot(np.arange(len(arr)), arr, color=colors[fi], linewidth=1.0, alpha=0.7,
                    label=folder if ji == 0 else None)
        for ex in extras:
            arr_e = ex.ep[key][:, ji]
            ax.plot(np.arange(len(arr_e)), arr_e, color=ex.color, linewidth=EXTRA_LW, alpha=1.0,
                    label=ex.label if ji == 0 else None)
        ax.set_title(v.JOINT_NAMES[ji], fontsize=11)
        ax.set_xlabel("timestep")
        ax.set_ylabel("position (rad)" if key == "qpos" else "velocity (rad/s)")
        ax.grid(True, alpha=0.3)

    ax = axes[7]
    for fi, (folder, ep) in enumerate(orig_eps.items()):
        t = np.arange(ep[key].shape[0])
        ax.plot(t, ep[key][:, 7], color=colors[fi], linewidth=1.0, alpha=0.7)
        ax.plot(t, ep[key][:, 8], color=colors[fi], linewidth=0.6, alpha=0.4, linestyle="--")
    for ex in extras:
        t = np.arange(ex.ep[key].shape[0])
        ax.plot(t, ex.ep[key][:, 7], color=ex.color, linewidth=EXTRA_LW, alpha=1.0)
        ax.plot(t, ex.ep[key][:, 8], color=ex.color, linewidth=1.2, alpha=0.7, linestyle="--")
    ax.set_title("gripper (solid=left, dashed=right)", fontsize=11)
    ax.set_xlabel("timestep")
    ax.set_ylabel("position (m)" if key == "qpos" else "velocity (m/s)")
    ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best", ncol=1, framealpha=0.85)
    suffix = "Positions" if key == "qpos" else "Velocities"
    fig.suptitle(f"Episode {EP_INDEX} — Joint {suffix}  ({summary})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_distance(orig_eps: dict, extras: list[ExtraRun], out_path: Path, summary: str) -> None:
    n = len(orig_eps)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))
    n_sensors = len(v.SENSOR_NAMES)
    cols = 6
    rows = (n_sensors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), sharey=True)
    axes = axes.flatten()
    for si, sn in enumerate(v.SENSOR_NAMES):
        ax = axes[si]
        for fi, (folder, ep) in enumerate(orig_eps.items()):
            if sn not in ep["proximity"]:
                continue
            closest = v._closest_distance(ep["proximity"][sn])
            ax.plot(np.arange(len(closest)), closest, color=colors[fi], linewidth=0.8, alpha=0.7,
                    label=folder if si == 0 else None)
        for ex in extras:
            if sn in ex.ep["proximity"]:
                closest = v._closest_distance(ex.ep["proximity"][sn])
                ax.plot(np.arange(len(closest)), closest, color=ex.color, linewidth=EXTRA_LW, alpha=1.0,
                        label=ex.label if si == 0 else None)
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
        f"Episode {EP_INDEX} — Closest detected distance per sensor "
        f"(substep {v.SUBSTEP_INDEX}, zero-pixels masked, ylim 0-2m; {summary})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----- tile plots -----


def _camera_paths(h1: Path, traj_idx: int) -> dict[str, Path]:
    return {
        "Exo RGB":   h1 / f"episode_{traj_idx:08d}_exo_camera_1_batch_1_of_1.mp4",
        "Exo Depth": h1 / f"episode_{traj_idx:08d}_exo_camera_1_depth_batch_1_of_1.mp4",
        "Wrist RGB": h1 / f"episode_{traj_idx:08d}_wrist_camera_batch_1_of_1.mp4",
        "Wrist Depth": h1 / f"episode_{traj_idx:08d}_wrist_camera_depth_batch_1_of_1.mp4",
    }


def plot_rgbd(orig_h1: Path, orig_T: int, extras: list[ExtraRun], out_path: Path) -> None:
    n_times = 6
    blocks = [("orig (seed=2026)", orig_h1, orig_T)]
    for ex in extras:
        blocks.append((ex.label, ex.h1, ex.T))
    cams = ["Exo RGB", "Exo Depth", "Wrist RGB", "Wrist Depth"]
    n_rows = len(blocks) * len(cams)
    fig, axes = plt.subplots(n_rows, n_times, figsize=(n_times * 2.5, n_rows * 1.7),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    for bi, (block_name, h1, T) in enumerate(blocks):
        t_idxs = np.linspace(0, T - 1, n_times, dtype=int).tolist()
        cam_paths = _camera_paths(h1, EP_INDEX)
        for ri, label in enumerate(cams):
            path = cam_paths[label]
            frames = v.sample_video_frames(path, t_idxs) if path.exists() else [
                np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
            row = bi * len(cams) + ri
            for ci, frame in enumerate(frames):
                ax = axes[row, ci]
                ax.imshow(frame)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"t={t_idxs[ci]}", fontsize=10)
                if ci == 0:
                    prefix = f"{block_name}\n" if ri == 0 else ""
                    ax.set_ylabel(f"{prefix}{label}", fontsize=9, rotation=90, va="center")
    fig.suptitle(
        f"Episode {EP_INDEX} — RGBD frames  "
        f"(orig block + {len(extras)} extra run block(s) stacked below)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_proximity_heatmaps(orig_ep: dict, orig_h1: Path, orig_T: int,
                             extras: list[ExtraRun], out_path: Path) -> None:
    n_times = 6
    n_sensors = len(v.SENSOR_NAMES)
    blocks = [("orig", orig_ep, orig_h1, orig_T)]
    for ex in extras:
        blocks.append((ex.label, ex.ep, ex.h1, ex.T))

    rows_per_block = 2 + n_sensors
    total_rows = rows_per_block * len(blocks)
    cam_h = 3.0
    sen_h = 1.0
    block_ratios = [cam_h, cam_h] + [sen_h] * n_sensors
    height_ratios = block_ratios * len(blocks)

    fig_w = n_times * 2.0 + 2.5
    fig_h = sum(height_ratios) * 0.55 + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=total_rows, ncols=n_times + 1,
        height_ratios=height_ratios,
        width_ratios=[1.0] * n_times + [0.08],
        wspace=0.06, hspace=0.18,
        left=0.10, right=0.94, top=0.97, bottom=0.01,
    )
    cmap = mpl.colormaps[v.DEPTH_CMAP].with_extremes(bad="black")
    norm = mpl.colors.Normalize(vmin=v.DEPTH_VMIN, vmax=v.DEPTH_VMAX)

    for bi, (block_name, ep, h1, T) in enumerate(blocks):
        row_off = bi * rows_per_block
        t_idxs = np.linspace(0, T - 1, n_times, dtype=int)
        cam_paths = _camera_paths(h1, EP_INDEX)
        exo_frames = v.sample_video_frames(cam_paths["Exo RGB"], t_idxs.tolist()) if cam_paths["Exo RGB"].exists() else [
            np.zeros((240, 320, 3), dtype=np.uint8)] * n_times
        wri_frames = v.sample_video_frames(cam_paths["Wrist RGB"], t_idxs.tolist()) if cam_paths["Wrist RGB"].exists() else [
            np.zeros((240, 320, 3), dtype=np.uint8)] * n_times

        for ci, frame in enumerate(exo_frames):
            ax = fig.add_subplot(gs[row_off + 0, ci])
            ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={t_idxs[ci]}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{block_name}\nExo RGB", fontsize=10, rotation=90, va="center")
        for ci, frame in enumerate(wri_frames):
            ax = fig.add_subplot(gs[row_off + 1, ci])
            ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel("Wrist RGB", fontsize=10, rotation=90, va="center")

        for si, sn in enumerate(v.SENSOR_NAMES):
            row = row_off + 2 + si
            prox = ep["proximity"].get(sn)
            short = sn.replace("link", "L").replace("_sensor_", ".s")
            for ti, t in enumerate(t_idxs):
                ax = fig.add_subplot(gs[row, ti])
                if prox is None:
                    ax.axis("off"); continue
                img = prox[t, v.SUBSTEP_INDEX]
                masked = np.ma.masked_where(img <= 1e-6, img)
                ax.imshow(masked, vmin=v.DEPTH_VMIN, vmax=v.DEPTH_VMAX, cmap=cmap,
                          interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if ti == 0:
                    ax.set_ylabel(short, fontsize=9, rotation=0, ha="right", va="center", labelpad=6)

    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("sensor depth (m)", fontsize=10, labelpad=6)
    cb.ax.tick_params(labelsize=9)
    cbar_ax.text(0.5, 1.005, "close", transform=cbar_ax.transAxes, fontsize=9, ha="center", va="bottom")
    cbar_ax.text(0.5, -0.005, f"far  (>{v.DEPTH_VMAX}m)", transform=cbar_ax.transAxes,
                 fontsize=9, ha="center", va="top")

    fig.suptitle(
        f"Episode {EP_INDEX} — Proximity heatmaps + RGB context  "
        f"(orig block on top, {len(extras)} extra run block(s) below)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    orig_folders = list_folders(ORIG_ROOT)
    print(f"Orig folders: {len(orig_folders)}")

    orig_eps: dict[str, dict] = {}
    for folder in orig_folders:
        h5 = folder / "house_1" / "trajectories_batch_1_of_1.h5"
        orig_eps[folder.name] = v.load_episode(h5, TRAJ_KEY)

    canonical = orig_folders[0].name
    canonical_qpos = orig_eps[canonical]["qpos"]
    identical = [canonical]
    outliers = []
    for f in orig_folders[1:]:
        ep_qp = orig_eps[f.name]["qpos"]
        if ep_qp.shape == canonical_qpos.shape and np.abs(ep_qp - canonical_qpos).max() < 1e-6:
            identical.append(f.name)
        else:
            outliers.append(f.name)
    print(f"Identical to canonical: {len(identical)} folders")
    print(f"Outlier orig folders: {outliers}")

    # Load each extra run's traj_0
    loaded_extras: list[ExtraRun] = []
    for ex in EXTRA_RUNS:
        if not ex.root.exists():
            print(f"Skipping {ex.label}: root {ex.root} not found")
            continue
        folders = list_folders(ex.root)
        if not folders:
            print(f"Skipping {ex.label}: no folders with h5")
            continue
        ex.folder = folders[0]
        ex.h1 = ex.folder / "house_1"
        ex.ep = v.load_episode(ex.h1 / "trajectories_batch_1_of_1.h5", TRAJ_KEY)
        ex.T = ex.ep["qpos"].shape[0]
        loaded_extras.append(ex)
        print(f"Loaded {ex.label}: T={ex.T}")

    orig_first = orig_folders[0]
    orig_h1 = orig_first / "house_1"
    orig_T = orig_eps[orig_first.name]["qpos"].shape[0]

    summary_parts = [f"{len(identical)} bit-identical originals"]
    if outliers:
        summary_parts.append(f"{len(outliers)} outlier orig folder(s)")
    for ex in loaded_extras:
        summary_parts.append(f"1 {ex.label}")
    summary = " + ".join(summary_parts)

    plot_joints(orig_eps, loaded_extras, "qpos",
                OUT_DIR / f"episode_{EP_INDEX}_joint_positions.png", summary)
    plot_joints(orig_eps, loaded_extras, "qvel",
                OUT_DIR / f"episode_{EP_INDEX}_joint_velocities.png", summary)
    plot_proximity_distance(orig_eps, loaded_extras,
                             OUT_DIR / f"episode_{EP_INDEX}_proximity_distance.png", summary)
    plot_rgbd(orig_h1, orig_T, loaded_extras,
              OUT_DIR / f"episode_{EP_INDEX}_rgbd.png")
    plot_proximity_heatmaps(orig_eps[orig_first.name], orig_h1, orig_T,
                             loaded_extras,
                             OUT_DIR / f"episode_{EP_INDEX}_proximity_heatmaps.png")
    print(f"Rewrote episode_{EP_INDEX}_*.png — {summary}")


if __name__ == "__main__":
    main()
