"""Plot decoder attention to each sensor across normalised episode time.

Uses the existing temporal_per_sensor data in
pact/analysis/attention_outputs/raw_stats.json.

Outputs:
  temporal_attention_heatmap.png — rows=sensors, cols=time buckets
  temporal_attention_top_sensors.png — line plot of top-K sensors over time
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def link_color(name: str) -> str:
    if "link2" in name: return "#1f77b4"
    if "link3" in name: return "#2ca02c"
    if "link5" in name: return "#ff7f0e"
    if "link6" in name: return "#d62728"
    return "#5e5e5e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attn_json", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    attn = json.load(open(args.attn_json))
    sensor_names = list(json.load(open(args.prox_mapping_json))["sensor_names"])
    N = len(sensor_names)

    ts = attn["temporal_per_sensor"]
    fracs = sorted([float(k) for k in ts.keys()])
    M = np.zeros((len(fracs), N), dtype=np.float32)
    for i, f in enumerate(fracs):
        M[i] = np.array(ts[f"{f:.3f}"])

    # ---- Plot 1: heatmap (sensors × time)
    fig, ax = plt.subplots(figsize=(11, 9))
    h = ax.imshow(M.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(N))
    ax.set_yticklabels(sensor_names, fontsize=7)
    ax.set_xticks(range(0, len(fracs), max(1, len(fracs)//10)))
    ax.set_xticklabels([f"{fracs[i]:.2f}" for i in range(0, len(fracs), max(1, len(fracs)//10))])
    ax.set_xlabel("normalised episode time")
    plt.colorbar(h, ax=ax, label="mean decoder cross-attention to sensor")
    ax.set_title("Decoder attention to each sensor across episode time (existing analysis)")
    for boundary in [7, 15, 21]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "temporal_attention_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'temporal_attention_heatmap.png'}")

    # ---- Plot 2: top-K sensors over time (line plot)
    K = 6
    # Pick top-K by max attention over time.
    top_idx = np.argsort(M.max(axis=0))[-K:][::-1]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j in top_idx:
        ax.plot(fracs, M[:, j], marker="o", color=link_color(sensor_names[j]),
                linewidth=1.8, label=sensor_names[j])
    uniform = 1.0 / N
    ax.axhline(uniform, color="grey", linestyle="--", alpha=0.7,
               label=f"uniform 1/N = {uniform:.4f}")
    ax.set_xlabel("normalised episode time")
    ax.set_ylabel("decoder cross-attention")
    ax.set_title(f"Top-{K} sensors by peak attention over time")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "temporal_attention_top_sensors.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'temporal_attention_top_sensors.png'}")

    # ---- Plot 3: per-link mean attention over time
    fig, ax = plt.subplots(figsize=(10, 5))
    by_link = {f"link{l}": [] for l in (2, 3, 5, 6)}
    for j, sn in enumerate(sensor_names):
        for l in (2, 3, 5, 6):
            if f"link{l}_" in sn:
                by_link[f"link{l}"].append(j)
                break
    for link, idx in by_link.items():
        if not idx:
            continue
        link_mean = M[:, idx].mean(axis=1)
        ax.plot(fracs, link_mean, marker="o", linewidth=2, label=f"{link} (n={len(idx)} sensors)",
                color=link_color(f"{link}_sensor_0"))
    ax.axhline(uniform, color="grey", linestyle="--", alpha=0.7, label=f"uniform 1/N")
    ax.set_xlabel("normalised episode time")
    ax.set_ylabel("mean attention per link")
    ax.set_title("Per-link mean decoder attention over time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "temporal_attention_per_link.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'temporal_attention_per_link.png'}")

    print("[temporal-attn] done.")


if __name__ == "__main__":
    main()
