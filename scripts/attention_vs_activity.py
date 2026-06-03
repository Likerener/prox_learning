"""Correlate per-sensor decoder cross-attention with per-sensor physical activity.

Tests whether the model attends MORE to sensors that observe MORE.

Inputs:
  --attn_json  pact/analysis/attention_outputs/raw_stats.json
  --activity_json eval_output/sensor_usage_analysis/summary.json
  --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json

Outputs (--output_dir):
  attn_vs_activity_scatter.png — per-sensor scatter + linear fit
  attn_vs_activity_per_phase.png — one scatter per phase
  attn_vs_activity_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attn_json", required=True)
    p.add_argument("--activity_json", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def link_color(name: str) -> str:
    if "link2" in name: return "#1f77b4"
    if "link3" in name: return "#2ca02c"
    if "link5" in name: return "#ff7f0e"
    if "link6" in name: return "#d62728"
    return "#5e5e5e"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    attn = json.load(open(args.attn_json))
    act = json.load(open(args.activity_json))
    sensor_names = list(json.load(open(args.prox_mapping_json))["sensor_names"])
    N = len(sensor_names)

    attn_per_sensor = np.array(attn["per_sensor"])      # (N,)
    phases = act["phases"]
    # phase_mean_activity is (n_phases, N)
    phase_mean = np.array(act["phase_mean_activity"])   # (P, N)
    # Overall mean activity per sensor (weighted by step counts in each phase)
    counts = np.array(act["phase_step_counts"])         # (P,)
    if counts.sum() > 0:
        overall_act = (phase_mean.T @ counts) / counts.sum()   # (N,)
    else:
        overall_act = np.zeros(N)

    # Pearson + Spearman
    def pearson(x, y):
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
        return float((x * y).mean())

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return pearson(rx, ry)

    r_overall = pearson(attn_per_sensor, overall_act)
    rho_overall = spearman(attn_per_sensor, overall_act)
    print(f"[attn-act] overall: Pearson r = {r_overall:.3f}, Spearman ρ = {rho_overall:.3f}")

    # ---- Plot 1: overall scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [link_color(s) for s in sensor_names]
    ax.scatter(overall_act, attn_per_sensor, c=colors, s=80,
               edgecolor="black", linewidth=0.6, alpha=0.85)
    # Linear fit
    coef = np.polyfit(overall_act, attn_per_sensor, deg=1)
    xs = np.linspace(overall_act.min(), overall_act.max(), 20)
    ax.plot(xs, np.polyval(coef, xs), color="grey", linestyle="--",
            label=f"fit: r={r_overall:.3f}, ρ={rho_overall:.3f}")
    # Label top-3 sensors
    top = np.argsort(-attn_per_sensor)[:3]
    for j in top:
        ax.annotate(sensor_names[j], (overall_act[j], attn_per_sensor[j]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("mean sensor activity (m, weighted by step count)")
    ax.set_ylabel("mean decoder cross-attention to sensor token")
    ax.set_title("Attention vs activity (per sensor)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=link_color(f"link{l}_sensor_0"),
                          markersize=10, label=f"link{l}")
               for l in (2, 3, 5, 6)]
    handles.append(plt.Line2D([0], [0], color="grey", linestyle="--", label="linear fit"))
    ax.legend(handles=handles, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "attn_vs_activity_scatter.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'attn_vs_activity_scatter.png'}")

    # ---- Plot 2: per-phase scatter
    fig, axes = plt.subplots(1, min(4, sum(c > 0 for c in counts)), figsize=(16, 4.5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    plot_phases = [(i, p) for i, p in enumerate(phases) if counts[i] > 0]
    for ax, (i, ph) in zip(axes, plot_phases[:len(axes)]):
        act_i = phase_mean[i]
        r = pearson(attn_per_sensor, act_i)
        rho = spearman(attn_per_sensor, act_i)
        colors = [link_color(s) for s in sensor_names]
        ax.scatter(act_i, attn_per_sensor, c=colors, s=60,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        coef_i = np.polyfit(act_i, attn_per_sensor, deg=1) if act_i.std() > 1e-6 else (0, attn_per_sensor.mean())
        xs = np.linspace(act_i.min(), act_i.max(), 20)
        ax.plot(xs, np.polyval(coef_i, xs), color="grey", linestyle="--", alpha=0.7)
        ax.set_xlabel("mean activity")
        ax.set_title(f"{ph}\nr={r:.2f}, ρ={rho:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("attention")
    fig.suptitle("Per-phase: attention vs activity correlation")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "attn_vs_activity_per_phase.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'attn_vs_activity_per_phase.png'}")

    # Summary
    summary = {
        "overall_pearson_r": r_overall,
        "overall_spearman_rho": rho_overall,
        "per_phase": {
            phases[i]: {
                "pearson_r": pearson(attn_per_sensor, phase_mean[i]),
                "spearman_rho": spearman(attn_per_sensor, phase_mean[i]),
                "n_steps": int(counts[i]),
            }
            for i in range(len(phases)) if counts[i] > 0
        },
    }
    with open(out / "attn_vs_activity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[attn-act] outputs in {out}")


if __name__ == "__main__":
    main()
