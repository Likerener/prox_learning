"""Master multi-panel paper figure combining all available analyses.

Panels:
  (A) Headline success rates: vanilla ACT vs P+ACT vs P+ACT mask_zero
  (B) Failure taxonomy bar (vanilla vs P+ACT)
  (C) Per-sensor activity by phase (heatmap)
  (D) Per-sensor success - failure activity diff (heatmap)
  (E) Decoder cross-attention per sensor (existing analysis)
  (F) Weight usage: per-modality input projection (P+ACT vs vanilla ACT)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    return p.parse_args()


def link_color(name: str) -> str:
    if "link2" in name: return "#1f77b4"
    if "link3" in name: return "#2ca02c"
    if "link5" in name: return "#ff7f0e"
    if "link6" in name: return "#d62728"
    return "#5e5e5e"


def safe_load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load all inputs.
    REPO = Path("/home/jaydv/code/prox_learning")
    headline_vanilla = safe_load(str(REPO / "eval_output/act_house1_mug_random_v1_aggregate_n50/summary.json")) or {}
    headline_pact    = safe_load(str(REPO / "eval_output/act_prox_mug_v1_aggregate_n50/summary.json")) or {}
    headline_mask    = safe_load(str(REPO / "eval_output/exp1_mask_zero_n50/summary.json")) or {}
    tax              = safe_load(str(REPO / "eval_output/exp3_failure_taxonomy/chi_square.json")) or {}
    activity_summary = safe_load(str(REPO / "eval_output/sensor_usage_analysis/summary.json")) or {}
    svf_summary      = safe_load(str(REPO / "eval_output/sensor_succ_vs_fail/summary.json")) or {}
    attn_stats       = safe_load(str(REPO / "pact/analysis/attention_outputs/raw_stats.json")) or {}
    mapping          = safe_load(str(REPO / "act_style_data/mug_house1_random_everything/prox_mapping.json")) or {}
    sensor_names     = list(mapping.get("sensor_names", []))
    N = len(sensor_names)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.4, 1], hspace=0.5, wspace=0.4)

    # ---------- Panel A: headline ----------
    axA = fig.add_subplot(gs[0, 0])
    names = ["vanilla ACT", "P+ACT", "P+ACT mask=0"]
    rates = [headline_vanilla.get("pooled_success_rate", 0),
             headline_pact.get("pooled_success_rate", 0),
             headline_mask.get("pooled_success_rate", 0)]
    counts = [(headline_vanilla.get("total_successes", 0), headline_vanilla.get("total_episodes", 0)),
              (headline_pact.get("total_successes", 0),    headline_pact.get("total_episodes", 0)),
              (headline_mask.get("total_successes", 0),    headline_mask.get("total_episodes", 0))]
    cis = [headline_vanilla.get("wilson_95_ci", [0, 0]),
           headline_pact.get("wilson_95_ci", [0, 0]),
           headline_mask.get("wilson_95_ci", [0, 0])]
    err_lo = [r - lo for r, (lo, _) in zip(rates, cis)]
    err_hi = [hi - r for r, (_, hi) in zip(rates, cis)]
    color = ["#5e5e5e", "#1f77b4", "#d62728"]
    x = np.arange(len(names))
    axA.bar(x, rates, yerr=np.vstack([err_lo, err_hi]), capsize=4,
            color=color, edgecolor="black", linewidth=0.6)
    for i, ((s, n), r) in enumerate(zip(counts, rates)):
        if n > 0:
            axA.text(x[i], r + 0.04, f"{int(s)}/{int(n)}\n({r:.0%})",
                     ha="center", va="bottom", fontsize=9)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("success rate")
    axA.set_title("(A) Success rate")
    axA.grid(True, axis="y", alpha=0.25)

    # ---------- Panel B: failure taxonomy ----------
    axB = fig.add_subplot(gs[0, 1])
    if tax:
        cats = tax["categories"]
        base_t = tax["table"][0]
        pact_t = tax["table"][1]
        xb = np.arange(len(cats))
        width = 0.38
        axB.bar(xb - width/2, base_t, width, color="#5e5e5e", edgecolor="black",
                linewidth=0.5, label=f"vanilla ({sum(base_t)} fails)")
        axB.bar(xb + width/2, pact_t, width, color="#1f77b4", edgecolor="black",
                linewidth=0.5, label=f"P+ACT ({sum(pact_t)} fails)")
        for i, (b, p) in enumerate(zip(base_t, pact_t)):
            axB.text(xb[i] - width/2, b + 0.2, str(int(b)), ha="center", fontsize=9)
            axB.text(xb[i] + width/2, p + 0.2, str(int(p)), ha="center", fontsize=9)
        axB.set_xticks(xb)
        axB.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
        axB.set_ylabel("# failed rollouts")
        axB.set_title(f"(B) Failure taxonomy  χ²={tax['chi2']:.2f}, V={tax['cramers_v']:.2f}")
        axB.legend(fontsize=8, loc="upper right")
        axB.grid(True, axis="y", alpha=0.25)

    # ---------- Panel E: attention per sensor ----------
    axE = fig.add_subplot(gs[0, 2])
    if attn_stats and sensor_names:
        per_s = np.array(attn_stats["per_sensor"])
        order = np.argsort(per_s)[::-1]
        colors_e = [link_color(sensor_names[i]) for i in order]
        axE.bar(range(len(order)), per_s[order], color=colors_e,
                edgecolor="black", linewidth=0.4)
        axE.set_xticks(range(len(order)))
        axE.set_xticklabels([sensor_names[i] for i in order], rotation=70, ha="right",
                            fontsize=6)
        axE.set_ylabel("mean attention")
        axE.set_title(f"(E) Decoder cross-attn per sensor (1/N={1/len(sensor_names):.4f})")
        axE.axhline(1.0/len(sensor_names), color="grey", linestyle="--", alpha=0.6,
                    label="uniform 1/N")
        axE.grid(True, axis="y", alpha=0.25)
        handles = [plt.Rectangle((0, 0), 1, 1, color=link_color(f"link{l}_sensor_0"), label=f"link{l}")
                   for l in (2, 3, 5, 6)]
        axE.legend(handles=handles, fontsize=8, loc="upper right")

    # ---------- Panel C: per-sensor activity by phase ----------
    axC = fig.add_subplot(gs[1, :2])
    if activity_summary and sensor_names:
        phases = activity_summary["phases"]
        phase_mean = np.array(activity_summary["phase_mean_activity"])
        counts_p = np.array(activity_summary["phase_step_counts"])
        ph_used = [i for i, c in enumerate(counts_p) if c > 0]
        h = axC.imshow(phase_mean[ph_used].T, aspect="auto", cmap="viridis", origin="lower")
        axC.set_yticks(range(N))
        axC.set_yticklabels(sensor_names, fontsize=6)
        axC.set_xticks(range(len(ph_used)))
        axC.set_xticklabels([f"{phases[i]}\n(n={counts_p[i]})" for i in ph_used], fontsize=9)
        plt.colorbar(h, ax=axC, label="activity (m)")
        axC.set_title("(C) Per-sensor mean activity by manipulation phase")
        for boundary in [7, 15, 21]:
            axC.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)

    # ---------- Panel D: success - failure activity heatmap ----------
    axD = fig.add_subplot(gs[1, 2])
    if svf_summary and sensor_names:
        phases_d = svf_summary["phases"]
        diff = np.array(svf_summary["diff_succ_fail"])
        succ_c = np.array([svf_summary.get(f"succ_count_{p}", 0) for p in phases_d])
        # Use the phase_step_counts from svf if available; otherwise use activity_summary
        ph_used_d = [i for i, p in enumerate(phases_d) if np.any(diff[i] != 0)]
        if ph_used_d:
            dmax = max(abs(diff).max(), 0.001)
            h2 = axD.imshow(diff[ph_used_d].T, aspect="auto", cmap="RdBu_r",
                            vmin=-dmax, vmax=dmax, origin="lower")
            axD.set_yticks(range(N))
            axD.set_yticklabels(sensor_names, fontsize=6)
            axD.set_xticks(range(len(ph_used_d)))
            axD.set_xticklabels([phases_d[i] for i in ph_used_d], rotation=15, ha="right",
                                fontsize=9)
            plt.colorbar(h2, ax=axD, label="Δ activity")
            axD.set_title("(D) Success − Failure activity")
            for boundary in [7, 15, 21]:
                axD.axhline(boundary - 0.5, color="black", linewidth=0.4, alpha=0.4)

    # ---------- Panel F: per-modality input weight projection ----------
    axF = fig.add_subplot(gs[2, 0])
    # Use the numbers from the weight_usage analysis.
    modalities = ["image", "qpos", "prox"]
    pact_per_in   = [0.5769, 4.339, 7.647]
    vanilla_per_in = [0.5835, 4.364, 0.0]   # vanilla has no prox
    xf = np.arange(len(modalities))
    width = 0.38
    axF.bar(xf - width/2, pact_per_in, width, color="#1f77b4", edgecolor="black",
            linewidth=0.6, label="P+ACT")
    axF.bar(xf + width/2, vanilla_per_in, width, color="#5e5e5e", edgecolor="black",
            linewidth=0.6, label="vanilla ACT")
    for i, v in enumerate(pact_per_in):
        axF.text(xf[i] - width/2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    axF.set_xticks(xf)
    axF.set_xticklabels(modalities)
    axF.set_ylabel("||W||₂ / √fan_in")
    axF.set_title("(F) Input-projection magnitude per modality")
    axF.legend(fontsize=9)
    axF.grid(True, axis="y", alpha=0.25)

    # ---------- Panel G: top sensors per phase (text annotation) ----------
    axG = fig.add_subplot(gs[2, 1:])
    axG.axis("off")
    if activity_summary:
        text_lines = ["Top-3 most active sensors per phase  (n=50 P+ACT trajectories)"]
        text_lines.append("─" * 70)
        for ph_name, top in activity_summary.get("top_sensors_per_phase", {}).items():
            top_3 = top[:3]
            line = f"  {ph_name:>11s}: "
            line += "  ".join(f"{t['sensor']} ({t['mean_activity']:.3f})" for t in top_3)
            text_lines.append(line)
        text_lines.append("")
        # Add success vs fail key signal
        text_lines.append("Strongest success-vs-failure activity differences:")
        text_lines.append("─" * 70)
        for ph_name, top in (svf_summary or {}).get("top_diff_per_phase", {}).items():
            top_3 = top[:3]
            line = f"  {ph_name:>11s}: "
            line += "  ".join(
                f"{t['sensor']} Δ={t['diff']:+.3f}" for t in top_3
            )
            text_lines.append(line)
        text_lines.append("")
        text_lines.append("Reading: positive Δ means SUCCESS trajectories show MORE activity in that sensor.")
        text_lines.append("Negative Δ means that sensor was spuriously firing in FAILED trajectories.")
        axG.text(0.0, 1.0, "\n".join(text_lines), transform=axG.transAxes,
                 va="top", ha="left", fontsize=10, family="monospace",
                 bbox=dict(facecolor="#f0f0f0", edgecolor="#aaa", boxstyle="round,pad=0.5"))

    fig.suptitle(
        "P+ACT — Multi-panel analysis: success, failure modes, sensor usage, attention, weights",
        fontsize=13,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper-master] saved {out}")


if __name__ == "__main__":
    main()
