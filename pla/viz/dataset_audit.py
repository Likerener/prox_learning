"""Dataset audit visualisations.

Run this on a small pilot collection BEFORE launching the long run.
The numbers from `pla.data.verify` tell you whether the dataset passes
the gate; these plots tell you *what's actually in it* — necessary for
catching:

    * ToF saturation across whole sensors (skin orientation flipped)
    * dead sensors (stuck at one value all episode)
    * frozen RGB (camera stalled mid-trajectory)
    * truncated episodes (TAMP planner timing out)
    * action explosion (joint deltas jumping around)
    * scenes that all look the same (procthor randomisation broken)

Outputs go to ``reports/checks/audit_<task>/`` — one PNG per plot.

Run::

    python -m pla.viz.dataset_audit \
        --data-dir data/raw/near_contact_pilot \
        --out reports/checks/audit_near_contact_pilot
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

PROX_THRESHOLD_MM = 200.0
ZNEAR_MM = 20.0
ZFAR_MM = 4000.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_PLA = "#0173b2"
COLOR_BAD = "#d62728"
COLOR_OK  = "#2ca02c"
COLOR_NEU = "#777777"


# =============================================================================
# Data access helpers
# =============================================================================

def list_h5_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*.h5") if p.name != "_preflight")


def _load_episode(h5_path: Path) -> dict | None:
    """Load one episode's full tensors. Returns None on failure."""
    try:
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                if "observations" not in grp:
                    continue
                obs = grp["observations"]
                rec = {
                    "tof":   obs["tof"][:],
                    "qpos":  obs["qpos"][:],
                    "rgb":   obs["rgb"][:] if "rgb" in obs else None,
                    "actions": grp["actions"][:],
                    "success": bool(grp.attrs.get("success", False)),
                    "n_sensors": int(obs["tof"].shape[1]),
                    "T": int(obs["tof"].shape[0]),
                    "file": str(h5_path),
                    "ep_key": key,
                }
                return rec
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: load failed {h5_path}: {e}", file=sys.stderr)
    return None


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))


# =============================================================================
# Plot 1: ToF heatmap montage at key timesteps
# =============================================================================

def tof_heatmap_montage(eps: list[dict], out: Path, n_eps: int = 4) -> None:
    """For ``n_eps`` sample episodes show ToF at 4 timesteps each.

    Layout: rows = episodes, columns = (t=0, t=T/3, t=2T/3, t=peak),
    each cell shows N sensors arranged in a 4x8 grid (32 sensors) with
    individual 8x8 zone heatmaps. We render each cell as a single
    image of the 32 sensors stacked vertically.

    The "peak" frame is the timestep where any sensor read closest —
    the most informative frame for verifying the dataset captured
    near-contact moments.
    """
    eps = eps[:n_eps]
    if not eps:
        print("  (no episodes; skipping tof montage)")
        return
    T_steps_per_ep = []
    for ep in eps:
        T = ep["T"]
        per_step_min = ep["tof"].reshape(T, -1).min(axis=1)
        peak_t = int(np.argmin(per_step_min))
        T_steps_per_ep.append([0, T // 3, 2 * T // 3, peak_t])

    fig, axes = plt.subplots(len(eps), 4,
                              figsize=(11, 2.6 * len(eps)),
                              gridspec_kw={"wspace": 0.15, "hspace": 0.3})
    if len(eps) == 1:
        axes = axes.reshape(1, -1)

    for r, (ep, ts) in enumerate(zip(eps, T_steps_per_ep)):
        n = ep["n_sensors"]
        # Layout: 8 columns, ceil(n/8) rows of 8x8 patches stacked into one image.
        ncols_grid = 8
        nrows_grid = (n + ncols_grid - 1) // ncols_grid
        for c, t in enumerate(ts):
            tof_t = ep["tof"][t]   # [N, 8, 8]
            # Build a (nrows_grid * 8) x (ncols_grid * 8) tile image.
            tile = np.full((nrows_grid * 8, ncols_grid * 8), ZFAR_MM, dtype=np.float32)
            for i in range(n):
                gr, gc = divmod(i, ncols_grid)
                tile[gr * 8:(gr + 1) * 8, gc * 8:(gc + 1) * 8] = tof_t[i]
            ax = axes[r, c]
            im = ax.imshow(tile, cmap="viridis_r", vmin=ZNEAR_MM, vmax=ZFAR_MM,
                           interpolation="nearest", aspect="equal")
            # Per-sensor box outlines for clarity
            for gr in range(nrows_grid):
                for gc in range(ncols_grid):
                    ax.add_patch(plt.Rectangle(
                        (gc * 8 - 0.5, gr * 8 - 0.5), 8, 8,
                        fill=False, edgecolor="white", linewidth=0.4,
                    ))
            t_label = f"t={t}"
            if c == 3:
                t_label += f"  (peak: {tof_t.min():.0f} mm)"
            ax.set_title(t_label, fontsize=8, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(
                    f"ep{r}: {Path(ep['file']).stem}\n"
                    f"T={ep['T']} success={ep['success']}",
                    fontsize=7,
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.02)
    cbar.set_label("depth (mm)", fontsize=8)

    fig.suptitle(
        "ToF heatmap montage — sample episodes at far / mid / late / peak-contact frames",
        fontsize=10, y=1.0,
    )
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 2: per-sensor depth distribution (32-panel histogram)
# =============================================================================

def per_sensor_distribution(eps: list[dict], out: Path) -> None:
    """One histogram per sensor index across all episodes.

    Catches:
      * dead sensors (single-spike distribution)
      * stuck-at-saturation (mass at ~4000 mm)
      * stuck-at-self-hit (mass at ~20 mm)
      * uneven coverage across the body
    """
    if not eps:
        print("  (no episodes; skipping per-sensor dist)")
        return
    n = max(ep["n_sensors"] for ep in eps)
    # Concat all tof readings: shape [total_steps, N, 8, 8]
    all_tof = np.concatenate([ep["tof"] for ep in eps], axis=0)
    # Per-sensor flatten over (T, 8, 8): [N, total_steps * 64]
    per_sensor = all_tof.transpose(1, 0, 2, 3).reshape(n, -1)

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 1.4 * rows + 0.5),
                              sharex=True, sharey=True,
                              gridspec_kw={"wspace": 0.12, "hspace": 0.4})
    bins = np.linspace(ZNEAR_MM, ZFAR_MM, 40)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        vals = per_sensor[i]
        std = vals.std()
        # Heuristic: stuck if std < 1 mm; never-close if min > 1500 mm.
        stuck = std < 1.0
        never_close = vals.min() > 1500
        color = COLOR_PLA if not (stuck or never_close) else COLOR_BAD
        ax.hist(vals, bins=bins, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.2)
        title = f"s{i}"
        if stuck:
            title += " STUCK"
        elif never_close:
            title += " FAR"
        ax.set_title(title, fontsize=7, pad=1,
                     color=COLOR_BAD if (stuck or never_close) else "black")
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        if i % cols == 0:
            ax.set_ylabel("log count", fontsize=7)
        if i // cols == rows - 1:
            ax.set_xlabel("depth (mm)", fontsize=7)
    fig.suptitle(
        f"Per-sensor depth distribution across {len(eps)} episodes "
        f"({all_tof.shape[0]} total timesteps × 64 zones each)",
        fontsize=10, y=1.0,
    )
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 3: episode traces — depth-min(t), action-norm(t), success markers
# =============================================================================

def episode_traces(eps: list[dict], out: Path, n_eps: int = 12) -> None:
    """Per-episode time-series of min depth and action norm."""
    eps = eps[:n_eps]
    if not eps:
        print("  (no episodes; skipping traces)")
        return
    cols = 4
    rows = (len(eps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.0 * rows),
                              sharey=False,
                              gridspec_kw={"wspace": 0.3, "hspace": 0.45})
    for i, ep in enumerate(eps):
        ax = axes.flat[i]
        T = ep["T"]
        depth_min = ep["tof"].reshape(T, -1).min(axis=1)
        act_norm = np.linalg.norm(ep["actions"], axis=1)

        ax.plot(depth_min, color=COLOR_PLA, label="min ToF (mm)", linewidth=1.2)
        ax.set_ylabel("depth (mm)", color=COLOR_PLA, fontsize=8)
        ax.tick_params(axis="y", labelcolor=COLOR_PLA, labelsize=7)
        ax.set_ylim(0, ZFAR_MM)
        ax.axhline(PROX_THRESHOLD_MM, color=COLOR_PLA, linestyle="--",
                   linewidth=0.6, alpha=0.6)
        ax.set_xlabel("step", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(act_norm, color="#de8f05", label="‖action‖", linewidth=1.0,
                 alpha=0.9)
        ax2.set_ylabel("‖action‖", color="#de8f05", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#de8f05", labelsize=7)
        ax2.spines["top"].set_visible(False)

        succ = "OK" if ep["success"] else "FAIL"
        succ_col = COLOR_OK if ep["success"] else COLOR_BAD
        ax.set_title(f"{Path(ep['file']).stem}  [{succ}]",
                      fontsize=8, color=succ_col, pad=2)
    for i in range(len(eps), rows * cols):
        axes.flat[i].axis("off")
    fig.suptitle(
        "Episode traces — min ToF reading and action norm vs time",
        fontsize=10, y=1.00,
    )
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 4: RGB strip — first / mid / last frame per episode
# =============================================================================

def rgb_strip(eps: list[dict], out: Path, n_eps: int = 6) -> None:
    eps = [e for e in eps if e["rgb"] is not None][:n_eps]
    if not eps:
        print("  (no episodes with RGB; skipping rgb strip)")
        return
    cols = 4
    fig, axes = plt.subplots(len(eps), cols, figsize=(8, 2.2 * len(eps)),
                              gridspec_kw={"wspace": 0.05, "hspace": 0.25})
    if len(eps) == 1:
        axes = axes.reshape(1, -1)
    for r, ep in enumerate(eps):
        rgb = ep["rgb"]   # [T, 3, H, W]
        T = rgb.shape[0]
        ts = [0, T // 3, 2 * T // 3, T - 1]
        for c, t in enumerate(ts):
            img = rgb[t].transpose(1, 2, 0)  # [H, W, 3]
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(f"t={t}", fontsize=7, pad=1)
            if c == 0:
                axes[r, c].set_ylabel(
                    f"{Path(ep['file']).stem}\nsuccess={ep['success']}",
                    fontsize=7,
                )
    fig.suptitle("RGB sanity strip — start / 1/3 / 2/3 / end frames per episode",
                 fontsize=10, y=1.0)
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 5: episode-length distribution
# =============================================================================

def length_distribution(eps: list[dict], out: Path) -> None:
    if not eps:
        return
    Ts = np.asarray([e["T"] for e in eps])
    successes = np.asarray([e["success"] for e in eps])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bins = np.linspace(Ts.min() - 1, Ts.max() + 1, max(20, len(set(Ts.tolist()))))
    ax.hist(Ts[successes], bins=bins, color=COLOR_OK, alpha=0.7,
            edgecolor="white", label=f"success ({int(successes.sum())})")
    ax.hist(Ts[~successes], bins=bins, color=COLOR_BAD, alpha=0.7,
            edgecolor="white", label=f"failure ({int((~successes).sum())})")
    for q, lbl in [(10, "p10"), (50, "p50"), (90, "p90")]:
        ax.axvline(np.percentile(Ts, q), color="black", linestyle=":",
                   linewidth=0.6, alpha=0.5)
        ax.text(np.percentile(Ts, q), ax.get_ylim()[1] * 0.92, lbl,
                fontsize=7, ha="center", color="#444")
    ax.set_xlabel("episode length T")
    ax.set_ylabel("count")
    ax.set_title(
        f"Episode-length distribution  (n = {len(eps)}, "
        f"success rate {100*successes.mean():.0f}%)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 6: action distribution per joint
# =============================================================================

def action_distribution(eps: list[dict], out: Path) -> None:
    if not eps:
        return
    all_acts = np.concatenate([e["actions"] for e in eps], axis=0)  # [T_total, 7]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Per-joint violin/box-style: just plot histograms overlaid
    ax = axes[0]
    bins = np.linspace(all_acts.min(), all_acts.max(), 60)
    for j in range(all_acts.shape[1]):
        ax.hist(all_acts[:, j], bins=bins, alpha=0.4,
                label=f"joint {j}", histtype="step", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(1.0, color=COLOR_BAD, linestyle="--", linewidth=0.8,
               label="|action| = 1 cap")
    ax.axvline(-1.0, color=COLOR_BAD, linestyle="--", linewidth=0.8)
    ax.set_xlabel("action value")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("Per-joint action distribution",
                 fontsize=9, pad=4)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    # Per-joint summary stats heatmap
    ax = axes[1]
    stats = np.stack([
        all_acts.mean(axis=0),
        all_acts.std(axis=0),
        np.abs(all_acts).max(axis=0),
    ], axis=0)
    im = ax.imshow(stats, cmap="coolwarm", vmin=-stats.max(), vmax=stats.max(),
                   aspect="auto")
    ax.set_yticks(range(3), labels=["mean", "std", "|max|"])
    ax.set_xticks(range(all_acts.shape[1]),
                  labels=[f"j{j}" for j in range(all_acts.shape[1])])
    for r in range(3):
        for c in range(all_acts.shape[1]):
            ax.text(c, r, f"{stats[r, c]:.2g}", ha="center", va="center",
                    fontsize=7, color="white" if abs(stats[r, c]) > stats.max() / 2
                    else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("Per-joint action summary", fontsize=9, pad=4)

    fig.suptitle(
        f"Action distribution across {all_acts.shape[0]} timesteps; "
        f"|max|={np.abs(all_acts).max():.3f} (sane bound 1.0)",
        fontsize=10, y=1.02,
    )
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Plot 7: per-sensor coverage heatmap (dead sensor finder)
# =============================================================================

def sensor_coverage(eps: list[dict], out: Path) -> None:
    if not eps:
        return
    n = max(ep["n_sensors"] for ep in eps)
    # Per-sensor min depth across the whole dataset, and per-sensor std.
    all_tof = np.concatenate([ep["tof"] for ep in eps], axis=0)
    per_sensor_min = all_tof.min(axis=(0, 2, 3))
    per_sensor_std = all_tof.std(axis=(0, 2, 3))
    per_sensor_max = all_tof.max(axis=(0, 2, 3))
    per_sensor_mean = all_tof.mean(axis=(0, 2, 3))

    # Layout 4x8 (32 sensors)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.0),
                              gridspec_kw={"hspace": 0.55})

    # Top: bar of per-sensor min
    ax = axes[0]
    bar_colors = []
    for v, s in zip(per_sensor_min, per_sensor_std):
        if s < 1:
            bar_colors.append(COLOR_BAD)
        elif v > 1500:
            bar_colors.append("#f3a35a")
        else:
            bar_colors.append(COLOR_PLA)
    ax.bar(range(n), per_sensor_min, color=bar_colors, edgecolor="black",
           linewidth=0.4)
    ax.axhline(PROX_THRESHOLD_MM, color="green", linestyle="--",
               linewidth=0.8, label=f"{PROX_THRESHOLD_MM:.0f} mm threshold")
    ax.axhline(1500, color="orange", linestyle=":",
               linewidth=0.7, label="1500 mm dead-sensor flag")
    ax.set_xlabel("sensor index"); ax.set_ylabel("min depth (mm)")
    ax.set_title(
        f"Per-sensor minimum reading across dataset  ("
        f"{int((per_sensor_min < PROX_THRESHOLD_MM).sum())}/{n} below threshold; "
        f"{int((per_sensor_std < 1).sum())}/{n} stuck)",
        fontsize=9,
    )
    ax.set_xticks(range(n))
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(frameon=False, fontsize=7)

    # Bottom: per-sensor std
    ax = axes[1]
    bar_colors2 = [COLOR_BAD if s < 1 else COLOR_PLA for s in per_sensor_std]
    ax.bar(range(n), per_sensor_std, color=bar_colors2, edgecolor="black",
           linewidth=0.4)
    ax.axhline(0.5, color=COLOR_BAD, linestyle="--", linewidth=0.7,
               label="stuck threshold (σ < 0.5 mm)")
    ax.set_xlabel("sensor index"); ax.set_ylabel("std (mm)")
    ax.set_title("Per-sensor variation across dataset (stuck sensors flagged red)",
                 fontsize=9)
    ax.set_xticks(range(n))
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Per-sensor coverage diagnostics",
                 fontsize=10, y=1.0)
    _save(fig, out)
    plt.close(fig)
    print(f"  wrote {out}.png/.pdf")


# =============================================================================
# Driver
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output dir for the audit plot bundle")
    p.add_argument("--max-episodes", type=int, default=64,
                   help="cap on episodes loaded (full audit gets slow on huge sets)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = list_h5_files(args.data_dir)
    if not files:
        print(f"no .h5 files in {args.data_dir}")
        sys.exit(1)
    rng = np.random.default_rng(args.seed)
    if len(files) > args.max_episodes:
        idx = rng.choice(len(files), size=args.max_episodes, replace=False)
        files = [files[i] for i in sorted(idx)]
    eps: list[dict] = []
    for f in files:
        rec = _load_episode(f)
        if rec is not None:
            eps.append(rec)
    print(f"loaded {len(eps)} episodes from {args.data_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out
    print()
    print("== generating audit plots ==")
    tof_heatmap_montage(eps, out / "01_tof_montage", n_eps=4)
    per_sensor_distribution(eps, out / "02_per_sensor_dist")
    sensor_coverage(eps, out / "03_sensor_coverage")
    episode_traces(eps, out / "04_episode_traces", n_eps=12)
    rgb_strip(eps, out / "05_rgb_strip", n_eps=6)
    length_distribution(eps, out / "06_length_distribution")
    action_distribution(eps, out / "07_action_distribution")

    # Index file so a reviewer can flip through.
    index = out / "INDEX.md"
    index.write_text(
        f"# Audit plots for `{args.data_dir}`\n\n"
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from "
        f"{len(eps)} episodes.\n\n"
        f"View these in order:\n\n"
        f"1. ![ToF montage](01_tof_montage.png)\n"
        f"2. ![Per-sensor distribution](02_per_sensor_dist.png)\n"
        f"3. ![Sensor coverage](03_sensor_coverage.png)\n"
        f"4. ![Episode traces](04_episode_traces.png)\n"
        f"5. ![RGB strip](05_rgb_strip.png)\n"
        f"6. ![Length distribution](06_length_distribution.png)\n"
        f"7. ![Action distribution](07_action_distribution.png)\n"
    )
    print()
    print(f"== done: see {out}/INDEX.md ==")


if __name__ == "__main__":
    main()
