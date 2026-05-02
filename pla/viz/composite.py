"""Composite MP4 using the NEW dataset that contains `proximity_rgb` — real
scene RGB rendered from each of 29 proximity cameras, not the empty-scene
replay hack.

Frame layout:
  ┌─────────────────────┬────────────────────────┐
  │  exo RGB            │  wrist RGB             │
  ├─────────────┬───────┴────────────────────────┤
  │  DEPTH 29× 8×8 │  PROXIMITY RGB 29× 8×8        │
  │ (saved)      │  (saved — REAL scene)         │
  ├─────────────┴────────────────────────────────┤
  │  min-depth per-patch time series + cursor   │
  └──────────────────────────────────────────────┘

Usage:
    python build_composite_v2.py \
        --h5 <path> --traj-idx 0 --out <output.mp4>
"""
from __future__ import annotations
import argparse, gc
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import imageio.v2 as imageio

ZFAR = 4.0

# MJCF camera enumeration order (link6 first) — MUST match axis-1 of proximity/proximity_rgb
MJCF_ORDER = (
    [f'link6_sensor_{i}' for i in range(8)] +
    [f'link5_sensor_{i}' for i in range(6)] +
    [f'link3_sensor_{i}' for i in range(8)] +
    [f'link2_sensor_{i}' for i in range(7)]
)


def _tile_panel(fig, fig_pos, data, title, cmap, vmin, vmax, names, *,
                nrows=4, ncols=8, is_rgb=False):
    sub = fig_pos.subgridspec(nrows, ncols, hspace=0.55, wspace=0.1)
    for i in range(nrows * ncols):
        ax = fig.add_subplot(sub[i // ncols, i % ncols])
        ax.set_xticks([]); ax.set_yticks([])
        if i < 29:
            if is_rgb:
                ax.imshow(data[i], interpolation='nearest')
            else:
                ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest')
            ax.set_title(names[i].replace('_sensor_', '_s'), fontsize=6)
    bbox = fig_pos.get_position(fig)
    fig.text(bbox.x0 + bbox.width / 2, bbox.y1 - 0.005, title,
             ha='center', va='bottom', fontsize=9, fontweight='bold')


def build(h5_path: Path, episode_idx: int, out: Path, fps: int = 8):
    with h5py.File(h5_path, 'r') as f:
        key = f'traj_{episode_idx}'
        depth = f[f'{key}/obs/extra/proximity'][:]         # (T,29,8,8)
        rgb   = f[f'{key}/obs/extra/proximity_rgb'][:]     # (T,29,8,8,3)
    T = depth.shape[0]
    print(f'[composite] depth={depth.shape}  rgb={rgb.shape}  T={T}')

    # Locate scene MP4s for the episode (datagen names them by episode_idx
    # within the house batch; episode_idx==traj_idx here since all 6 in this
    # house were successes).
    ep_num = episode_idx
    base = h5_path.parent
    exo_path = base / f'episode_{ep_num:08d}_exo_camera_1_batch_1_of_1.mp4'
    wrist_path = base / f'episode_{ep_num:08d}_wrist_camera_batch_1_of_1.mp4'
    print(f'[composite] exo={exo_path.exists()}  wrist={wrist_path.exists()}')
    exo = imageio.mimread(str(exo_path), memtest=False) if exo_path.exists() else None
    wrist = imageio.mimread(str(wrist_path), memtest=False) if wrist_path.exists() else None

    frame_count = T
    if exo is not None: frame_count = min(frame_count, len(exo))
    if wrist is not None: frame_count = min(frame_count, len(wrist))

    saved_min = depth.reshape(T, 29, -1).min(axis=-1)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, quality=7)
    try:
        for t in range(frame_count):
            fig = plt.figure(figsize=(16, 9.2), dpi=85)
            gs_outer = gridspec.GridSpec(3, 2, figure=fig,
                                          height_ratios=[1.3, 2.5, 1.0],
                                          hspace=0.3, wspace=0.1)

            ax_exo = fig.add_subplot(gs_outer[0, 0]); ax_wr = fig.add_subplot(gs_outer[0, 1])
            if exo is not None: ax_exo.imshow(exo[t])
            if wrist is not None: ax_wr.imshow(wrist[t])
            ax_exo.set_title(f'exo  t={t}/{frame_count - 1}', fontsize=9)
            ax_wr.set_title('wrist', fontsize=9)
            for ax in (ax_exo, ax_wr): ax.set_xticks([]); ax.set_yticks([])

            _tile_panel(fig, gs_outer[1, 0], depth[t],
                        'PROXIMITY DEPTH (saved, scene)',
                        cmap='viridis_r', vmin=0, vmax=ZFAR, names=MJCF_ORDER)
            _tile_panel(fig, gs_outer[1, 1], rgb[t],
                        'PROXIMITY RGB (saved, scene — real scene content)',
                        cmap=None, vmin=None, vmax=None, names=MJCF_ORDER, is_rgb=True)

            ax_ts = fig.add_subplot(gs_outer[2, :])
            for i in range(29):
                ax_ts.plot(saved_min[:, i], alpha=0.45, lw=0.7)
            ax_ts.axvline(t, color='red', lw=1.2, alpha=0.9)
            ax_ts.axhline(0.30, color='k', ls='--', lw=0.5, alpha=0.4)
            ax_ts.set_xlim(0, frame_count - 1); ax_ts.set_ylim(0, ZFAR * 1.05)
            ax_ts.set_xlabel('step'); ax_ts.set_ylabel('min depth (m)')
            ax_ts.set_title('Per-patch min depth over time', fontsize=9)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)[..., 1:]
            writer.append_data(img)
            plt.close(fig)
            # periodic GC to keep memory flat
            if (t + 1) % 20 == 0:
                gc.collect()
                print(f'  frame {t + 1}/{frame_count}')
        print(f'  frame {frame_count}/{frame_count}')
    finally:
        writer.close()
    print(f'wrote {out} ({out.stat().st_size / 1024:.1f} KB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', type=Path, required=True)
    ap.add_argument('--traj-idx', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=int, default=8)
    args = ap.parse_args()
    build(args.h5, args.traj_idx, args.out, fps=args.fps)


if __name__ == '__main__':
    main()
