"""Joint sensor ranking combining: activity, attention, success-fail diff.

For each sensor, compute three scores:
  - rank by mean physical activity
  - rank by mean decoder attention
  - rank by success-vs-failure activity diff in critical phases (pregrasp + grasp_lift)

Then plot:
  - rank correlation matrix
  - top-K sensors by joint score
  - bottom-K sensors (uninformative ones)

Outputs:
  joint_ranking.png
  joint_ranking_summary.json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    REPO = Path("/home/jaydv/code/prox_learning")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    activity = json.load(open(REPO / "eval_output/sensor_usage_analysis/summary.json"))
    svf      = json.load(open(REPO / "eval_output/sensor_succ_vs_fail/summary.json"))
    attn     = json.load(open(REPO / "pact/analysis/attention_outputs/raw_stats.json"))
    mapping  = json.load(open(REPO / "act_style_data/mug_house1_random_everything/prox_mapping.json"))
    sensor_names = list(mapping["sensor_names"])
    N = len(sensor_names)

    phases = activity["phases"]
    phase_mean = np.array(activity["phase_mean_activity"])
    counts = np.array(activity["phase_step_counts"])
    overall_activity = (phase_mean.T @ counts) / max(counts.sum(), 1)   # (N,)

    diff = np.array(svf["diff_succ_fail"])    # (P, N)
    critical_idx = [phases.index(p) for p in ("pregrasp", "grasp_lift") if p in phases]
    diff_critical = diff[critical_idx].mean(axis=0) if critical_idx else np.zeros(N)

    attn_per_sensor = np.array(attn["per_sensor"])

    # Ranks (1 = highest)
    def rank(x):
        order = np.argsort(-x)
        ranks = np.empty_like(x, dtype=int)
        ranks[order] = np.arange(1, len(x) + 1)
        return ranks

    rank_activity = rank(overall_activity)
    rank_attention = rank(attn_per_sensor)
    rank_diff = rank(diff_critical)

    # Combined score: lower = better. Sum of ranks
    combined = rank_activity + rank_attention + rank_diff

    # Pearson correlation between ranks
    def pearson(x, y):
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
        return float((x * y).mean())

    rA_rB = pearson(rank_activity, rank_attention)
    rA_rD = pearson(rank_activity, rank_diff)
    rB_rD = pearson(rank_attention, rank_diff)

    # ---- Plot: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Rank correlation matrix
    ax = axes[0]
    M = np.array([
        [1.0,     rA_rB,   rA_rD],
        [rA_rB,   1.0,     rB_rD],
        [rA_rD,   rB_rD,   1.0],
    ])
    h = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["activity", "attention", "succ-fail Δ"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(M[i, j]) < 0.5 else "white", fontsize=14)
    plt.colorbar(h, ax=ax, label="Pearson r between ranks")
    ax.set_title("(A) Rank correlation between three signal sources")

    # Panel 2: Top-10 by joint score
    ax = axes[1]
    order = np.argsort(combined)
    K = 10
    top_idx = order[:K]
    bottom_idx = order[-K:][::-1]

    y = np.arange(K)
    names_t = [sensor_names[i] for i in top_idx]
    scores_t = combined[top_idx]
    colors_t = [link_color(n) for n in names_t]
    ax.barh(y, scores_t[::-1], color=colors_t[::-1], edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_t[::-1], fontsize=9)
    ax.set_xlabel("combined rank (lower = more important)")
    ax.set_title(f"(B) Top-{K} sensors by joint ranking")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate the three ranks for the top sensors
    for i, idx in enumerate(top_idx):
        s = f"  A={rank_activity[idx]}, Att={rank_attention[idx]}, Δ={rank_diff[idx]}"
        ax.text(combined[idx] + 0.5, K - 1 - i, s, ha="left", va="center", fontsize=8)

    # Panel 3: Bottom-10 sensors (least important)
    ax = axes[2]
    names_b = [sensor_names[i] for i in bottom_idx]
    scores_b = combined[bottom_idx]
    colors_b = [link_color(n) for n in names_b]
    ax.barh(y, scores_b[::-1], color=colors_b[::-1], edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_b[::-1], fontsize=9)
    ax.set_xlabel("combined rank (higher = less important)")
    ax.set_title(f"(C) Bottom-{K} sensors")
    ax.grid(True, axis="x", alpha=0.3)
    for i, idx in enumerate(bottom_idx):
        s = f"  A={rank_activity[idx]}, Att={rank_attention[idx]}, Δ={rank_diff[idx]}"
        ax.text(combined[idx] + 0.5, K - 1 - i, s, ha="left", va="center", fontsize=8)

    handles = [plt.Rectangle((0,0),1,1, color=link_color(f"link{l}_sensor_0"),
                              label=f"link{l}")
               for l in (2, 3, 5, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Joint sensor ranking — activity × attention × success-vs-failure",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out / "joint_ranking.png", dpi=140)
    plt.close(fig)
    print(f"[joint-rank] saved {out/'joint_ranking.png'}")

    summary = {
        "rank_correlations": {
            "activity_vs_attention": rA_rB,
            "activity_vs_diff": rA_rD,
            "attention_vs_diff": rB_rD,
        },
        "top10": [{"sensor": sensor_names[i],
                   "rank_activity": int(rank_activity[i]),
                   "rank_attention": int(rank_attention[i]),
                   "rank_diff": int(rank_diff[i]),
                   "combined": int(combined[i])} for i in top_idx],
        "bottom10": [{"sensor": sensor_names[i],
                      "rank_activity": int(rank_activity[i]),
                      "rank_attention": int(rank_attention[i]),
                      "rank_diff": int(rank_diff[i]),
                      "combined": int(combined[i])} for i in bottom_idx],
    }
    with open(out / "joint_ranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
