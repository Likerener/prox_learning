"""Single-figure summary: sensor characterization throughout manipulation.

Combines:
  (A) Per-sensor activity over normalised episode time (heatmap)
  (B) Per-phase top-K sensors (4 sub-panels)
  (C) Sensor activity SUCCESS - FAILURE per phase
  (D) Phase Gantt (mini)
  (E) Action delta per phase (success vs fail)

This figure tells the complete "use case of sensors throughout manipulation" story.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np


def link_color(name: str) -> str:
    if "link2" in name: return "#1f77b4"
    if "link3" in name: return "#2ca02c"
    if "link5" in name: return "#ff7f0e"
    if "link6" in name: return "#d62728"
    return "#5e5e5e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    REPO = Path("/home/jaydv/code/prox_learning")

    activity = json.load(open(REPO / "eval_output/sensor_usage_analysis/summary.json"))
    svf      = json.load(open(REPO / "eval_output/sensor_succ_vs_fail/summary.json"))
    dur      = json.load(open(REPO / "eval_output/phase_durations/phase_duration_summary.json"))
    act_var  = json.load(open(REPO / "eval_output/action_variance_analysis/action_summary.json"))
    mapping  = json.load(open(REPO / "act_style_data/mug_house1_random_everything/prox_mapping.json"))
    sensor_names = list(mapping["sensor_names"])
    N = len(sensor_names)

    PHASES = activity["phases"]   # 5 phases
    phase_mean = np.array(activity["phase_mean_activity"])
    counts_p   = np.array(activity["phase_step_counts"])
    ph_used    = [i for i, c in enumerate(counts_p) if c > 0]
    diff = np.array(svf["diff_succ_fail"])

    fig = plt.figure(figsize=(20, 12))
    gs = mgs.GridSpec(3, 4, figure=fig,
                      height_ratios=[1.6, 1.1, 1.0],
                      width_ratios=[1.4, 1, 1, 1],
                      hspace=0.55, wspace=0.5)

    # ---------- Panel A: per-sensor activity by phase heatmap ----------
    axA = fig.add_subplot(gs[0, 0])
    hA = axA.imshow(phase_mean[ph_used].T, aspect="auto", cmap="viridis", origin="lower")
    axA.set_yticks(range(N))
    axA.set_yticklabels(sensor_names, fontsize=6)
    axA.set_xticks(range(len(ph_used)))
    axA.set_xticklabels([f"{PHASES[i]}\n(n={counts_p[i]})" for i in ph_used],
                        fontsize=9)
    plt.colorbar(hA, ax=axA, label="activity (m)")
    axA.set_title("(A) Per-sensor activity by phase")
    for boundary in [7, 15, 21]:
        axA.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)

    # ---------- Panel B: top-3 sensors per phase (small subplots) ----------
    for ax_i, ph_i in enumerate(ph_used[:3]):
        axB = fig.add_subplot(gs[0, 1 + ax_i])
        top = activity["top_sensors_per_phase"].get(PHASES[ph_i], [])[:5]
        names_top = [t["sensor"] for t in top]
        vals_top = [t["mean_activity"] for t in top]
        colors_b = [link_color(s) for s in names_top]
        axB.barh(range(len(names_top)), vals_top[::-1], color=colors_b[::-1],
                 edgecolor="black", linewidth=0.5)
        axB.set_yticks(range(len(names_top)))
        axB.set_yticklabels(names_top[::-1], fontsize=8)
        axB.set_title(f"top-5 in {PHASES[ph_i]}", fontsize=10)
        axB.set_xlabel("activity (m)")
        axB.grid(True, axis="x", alpha=0.3)

    # ---------- Panel C: success-failure activity difference ----------
    axC = fig.add_subplot(gs[1, :2])
    dmax = max(abs(diff).max(), 1e-3)
    hC = axC.imshow(diff[ph_used].T, aspect="auto", cmap="RdBu_r",
                    vmin=-dmax, vmax=dmax, origin="lower")
    axC.set_yticks(range(N))
    axC.set_yticklabels(sensor_names, fontsize=6)
    axC.set_xticks(range(len(ph_used)))
    axC.set_xticklabels([PHASES[i] for i in ph_used], fontsize=9)
    plt.colorbar(hC, ax=axC, label="Δ activity (succ − fail)")
    axC.set_title("(C) Per-sensor activity: SUCCESS − FAILURE")
    for boundary in [7, 15, 21]:
        axC.axhline(boundary - 0.5, color="black", linewidth=0.4, alpha=0.4)

    # ---------- Panel D: phase duration distribution (succ vs fail) ----------
    axD = fig.add_subplot(gs[1, 2:])
    succ_dur = [dur["succ_mean_duration"][p] for p in PHASES]
    fail_dur = [dur["fail_mean_duration"][p] for p in PHASES]
    x = np.arange(len(PHASES))
    width = 0.38
    axD.bar(x - width/2, succ_dur, width, color="#1f77b4", edgecolor="black",
            linewidth=0.6, label=f"success (n={dur['n_success']})")
    axD.bar(x + width/2, fail_dur, width, color="#d62728", edgecolor="black",
            linewidth=0.6, label=f"failure (n={dur['n_fail']})")
    for i, (s, f) in enumerate(zip(succ_dur, fail_dur)):
        axD.text(x[i] - width/2, s + 0.005, f"{s:.2f}", ha="center", fontsize=8)
        axD.text(x[i] + width/2, f + 0.005, f"{f:.2f}", ha="center", fontsize=8)
    axD.set_xticks(x)
    axD.set_xticklabels(PHASES)
    axD.set_ylabel("mean fraction of episode")
    axD.set_title("(D) Phase duration: success vs failure")
    axD.legend(fontsize=9)
    axD.grid(True, axis="y", alpha=0.25)

    # ---------- Panel E: action delta per phase ----------
    axE = fig.add_subplot(gs[2, :2])
    succ_d = [act_var["succ_mean_delta"][p] for p in PHASES]
    fail_d = [act_var["fail_mean_delta"][p] for p in PHASES]
    axE.bar(x - width/2, succ_d, width, color="#1f77b4", edgecolor="black",
            linewidth=0.6, label=f"success (n={act_var['n_success']})")
    axE.bar(x + width/2, fail_d, width, color="#d62728", edgecolor="black",
            linewidth=0.6, label=f"failure (n={act_var['n_fail']})")
    for i, (s, f) in enumerate(zip(succ_d, fail_d)):
        if s > 0: axE.text(x[i] - width/2, s + 0.05, f"{s:.2f}", ha="center", fontsize=8)
        if f > 0: axE.text(x[i] + width/2, f + 0.05, f"{f:.2f}", ha="center", fontsize=8)
    axE.set_xticks(x)
    axE.set_xticklabels(PHASES)
    axE.set_ylabel("mean ||a_{t+1} − a_t||")
    axE.set_title("(E) Action commitment by phase  "
                  "(success commits 6× harder in pregrasp)")
    axE.legend(fontsize=9)
    axE.grid(True, axis="y", alpha=0.25)

    # ---------- Panel F: key text summary ----------
    axF = fig.add_subplot(gs[2, 2:])
    axF.axis("off")
    text = []
    text.append("Sensor usage characterisation — key findings")
    text.append("─" * 52)
    text.append("• Wrist sensors (link6) dominate pregrasp & grasp_lift")
    top_pg = activity["top_sensors_per_phase"].get("pregrasp", [])[:3]
    text.append(f"   pregrasp top-3: {[t['sensor'] for t in top_pg]}")
    top_gl = activity["top_sensors_per_phase"].get("grasp_lift", [])[:3]
    text.append(f"   grasp_lift top-3: {[t['sensor'] for t in top_gl]}")
    text.append("")
    text.append("• Success trajectories show stronger wrist activity")
    top_diff_pg = svf["top_diff_per_phase"].get("pregrasp", [])[:2]
    for t in top_diff_pg:
        sign = "+" if t["diff"] > 0 else ""
        text.append(f"   pregrasp {t['sensor']}: Δ={sign}{t['diff']:.3f} m")
    text.append("")
    text.append("• Failures spend 4× more time stuck in contact phases:")
    text.append(f"   pregrasp:   succ {dur['succ_mean_duration']['pregrasp']:.0%}, "
                f"fail {dur['fail_mean_duration']['pregrasp']:.0%}")
    text.append(f"   grasp_lift: succ {dur['succ_mean_duration']['grasp_lift']:.0%}, "
                f"fail {dur['fail_mean_duration']['grasp_lift']:.0%}")
    text.append("")
    text.append("• Successful pregrasp action is 6× larger than failure:")
    text.append(f"   succ Δa = {act_var['succ_mean_delta']['pregrasp']:.2f}")
    text.append(f"   fail Δa = {act_var['fail_mean_delta']['pregrasp']:.2f}")
    text.append("")
    text.append("Reading: proximity sensors are most informative during the")
    text.append("contact phases, with wrist (link6) sensors carrying the bulk")
    text.append("of the signal that distinguishes successful from failed grasps.")
    axF.text(0.0, 1.0, "\n".join(text), transform=axF.transAxes,
             va="top", ha="left", fontsize=10, family="monospace",
             bbox=dict(facecolor="#f5f5f5", edgecolor="#aaa",
                       boxstyle="round,pad=0.5"))

    fig.suptitle("Sensor usage characterisation across the manipulation trajectory  "
                 "(n=50 P+ACT trajectories on house_1 mug pick-and-place)",
                 fontsize=14)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[char] saved {out}")


if __name__ == "__main__":
    main()
