"""Clean, simple paper figure built from v2 (n=50 policy_last) analyses.

No abbreviations, big labels, no jargon. Each panel answers one question.

Panels:
  A) Success rate at n=50 (PACT vs baseline) — bar with Wilson CIs
  B) Failure types (counts) — grouped bars
  C) Sensor activity by phase (mean activity, top sensors highlighted)
  D) Success vs failure activity difference at pregrasp/grasp_lift
  E) Phase duration: success vs failure
  F) Attention × activity correlation per phase
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np


def link_color(n: str) -> str:
    if "link2" in n: return "#4C72B0"
    if "link3" in n: return "#55A868"
    if "link5" in n: return "#DD8452"
    if "link6" in n: return "#C44E52"
    return "#777"


def wilson(p, n, z=1.96):
    if n == 0: return 0, 0
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = (z/denom) * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return centre - half, centre + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    REPO = Path("/home/jaydv/code/prox_learning")

    # ---------- Load data ----------
    PA = REPO/"eval_output/paper_analysis_2026_05_27"
    base = json.load(open(REPO/"eval_output/act_house1_mug_random_v1_aggregate_n50/summary.json"))
    pact = json.load(open(REPO/"eval_output/best_epoch_headline_n50/summary.json"))
    mask = json.load(open(REPO/"eval_output/best_epoch_mask_zero_n50/summary.json"))
    mmean = json.load(open(REPO/"eval_output/mask_mean_n50/summary.json"))
    tax  = json.load(open(PA/"failure_taxonomy_v2/chi_square.json"))
    act  = json.load(open(PA/"sensor_usage_v2/summary.json"))
    svf  = json.load(open(PA/"sensor_succ_vs_fail_v2/summary.json"))
    dur  = json.load(open(PA/"phase_durations_v2/phase_duration_summary.json"))
    attn = json.load(open(PA/"attn_vs_activity_v2/attn_vs_activity_summary.json"))
    mapping = json.load(open(REPO/"act_style_data/mug_house1_random_everything/prox_mapping.json"))
    sensor_names = mapping["sensor_names"]
    PHASES = act["phases"]

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig = plt.figure(figsize=(18, 11))
    gs = mgs.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---------- A. Success rate ----------
    axA = fig.add_subplot(gs[0, 0])
    labels = ["Vanilla ACT\n(no proximity)", "PACT\n(+ proximity)",
              "PACT\n(prox→zero)", "PACT\n(prox→train mean)"]
    rates = [base["pooled_success_rate"], pact["pooled_success_rate"],
             mask["pooled_success_rate"], mmean["pooled_success_rate"]]
    ns    = [base["total_episodes"], pact["total_episodes"],
             mask["total_episodes"], mmean["total_episodes"]]
    cis   = [base["wilson_95_ci"], pact["wilson_95_ci"],
             mask["wilson_95_ci"], mmean["wilson_95_ci"]]
    errs  = [[r-ci[0] for r,ci in zip(rates,cis)],
             [ci[1]-r for r,ci in zip(rates,cis)]]
    x = np.arange(4)
    bars = axA.bar(x, rates, color=["#888", "#C44E52", "#E0A0A0", "#E8C0C0"],
                   edgecolor="black", linewidth=0.8, width=0.62)
    axA.errorbar(x, rates, yerr=errs, fmt="none", color="black",
                 capsize=5, linewidth=1.1)
    for i, (r, n) in enumerate(zip(rates, ns)):
        axA.text(i, r + 0.025, f"{r*100:.0f}%\n({int(round(r*n))}/{n})",
                 ha="center", fontsize=9, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylim(0, 1.05); axA.set_ylabel("success rate")
    axA.set_title("A. Success + masking ablations (n=50 each)")
    axA.axhline(0.5, color="gray", linestyle=":", linewidth=0.6)
    axA.text(0.5, 0.04,
             "zero hurts (OOD shock), mean does not → weak/redundant reliance",
             transform=axA.transAxes, ha="center", fontsize=8, color="#555",
             style="italic")

    # ---------- B. Failure types ----------
    axB = fig.add_subplot(gs[0, 1])
    cats = tax["categories"]
    base_counts, pact_counts = tax["table"][0], tax["table"][1]
    x = np.arange(len(cats))
    w = 0.38
    axB.bar(x - w/2, base_counts, w, color="#888", edgecolor="black",
            label=f"Baseline ({sum(base_counts)} fails)")
    axB.bar(x + w/2, pact_counts, w, color="#C44E52", edgecolor="black",
            label=f"PACT ({sum(pact_counts)} fails)")
    for i, (b, p) in enumerate(zip(base_counts, pact_counts)):
        if b: axB.text(i-w/2, b+0.3, str(b), ha="center", fontsize=10)
        if p: axB.text(i+w/2, p+0.3, str(p), ha="center", fontsize=10)
    axB.set_xticks(x)
    axB.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=10)
    axB.set_ylabel("number of failed trajectories")
    axB.set_title("B. How failures break down")
    axB.legend(loc="upper right", fontsize=10)
    pp_base = base_counts[1]/sum(base_counts)*100
    pp_pact = pact_counts[1]/sum(pact_counts)*100
    axB.text(0.02, 0.97,
             f"Failures at pregrasp:\n  Baseline {pp_base:.0f}%\n  PACT {pp_pact:.0f}%",
             transform=axB.transAxes, va="top", fontsize=10,
             bbox=dict(facecolor="#fff8dc", edgecolor="#bb9", pad=4))

    # ---------- C. Sensor activity per phase (top-5 each) ----------
    axC = fig.add_subplot(gs[0, 2])
    contact_phases = ["pregrasp", "grasp_lift"]
    bar_labels, bar_vals, bar_colors = [], [], []
    for ph in contact_phases:
        top = act["top_sensors_per_phase"].get(ph, [])[:4]
        for t in top:
            bar_labels.append(f"{t['sensor']}\n({ph})")
            bar_vals.append(t["mean_activity"])
            bar_colors.append(link_color(t["sensor"]))
    y = np.arange(len(bar_labels))
    axC.barh(y, bar_vals[::-1], color=bar_colors[::-1],
             edgecolor="black", linewidth=0.5)
    axC.set_yticks(y); axC.set_yticklabels(bar_labels[::-1], fontsize=9)
    axC.set_xlabel("mean activity (metres)")
    axC.set_title("C. Top sensors during contact phases")
    handles = [plt.Rectangle((0,0),1,1, color=link_color(f"link{l}_sensor_0"),
                              label=f"link{l}") for l in (2, 5, 6)]
    axC.legend(handles=handles, loc="lower right", fontsize=9)

    # ---------- D. Succ vs fail activity diff (pregrasp + grasp_lift) ----------
    axD = fig.add_subplot(gs[1, 0])
    diff = np.array(svf["diff_succ_fail"])
    P = svf["phases"]
    pg_i = P.index("pregrasp"); gl_i = P.index("grasp_lift")
    diff_contact = (diff[pg_i] + diff[gl_i]) / 2
    order = np.argsort(-np.abs(diff_contact))[:10]
    labels_d = [sensor_names[i] for i in order]
    vals_d   = diff_contact[order]
    colors_d = [link_color(n) for n in labels_d]
    y = np.arange(len(labels_d))
    axD.barh(y, vals_d[::-1], color=colors_d[::-1],
             edgecolor="black", linewidth=0.5)
    axD.axvline(0, color="black", linewidth=0.8)
    axD.set_yticks(y); axD.set_yticklabels(labels_d[::-1], fontsize=9)
    axD.set_xlabel("Δ activity, success − failure (metres)")
    axD.set_title("D. Which sensors differ in success vs failure?\n"
                  "(averaged over pregrasp + grasp_lift)")

    # ---------- E. Phase duration ----------
    axE = fig.add_subplot(gs[1, 1])
    succ_d = [dur["succ_mean_duration"][p] for p in PHASES]
    fail_d = [dur["fail_mean_duration"][p] for p in PHASES]
    x = np.arange(len(PHASES))
    w = 0.38
    axE.bar(x - w/2, succ_d, w, color="#4C72B0", edgecolor="black",
            label=f"success (n={dur['n_success']})")
    axE.bar(x + w/2, fail_d, w, color="#C44E52", edgecolor="black",
            label=f"failure (n={dur['n_fail']})")
    for i, (s, f) in enumerate(zip(succ_d, fail_d)):
        if s > 0.01: axE.text(i-w/2, s+0.01, f"{s*100:.0f}%", ha="center", fontsize=9)
        if f > 0.01: axE.text(i+w/2, f+0.01, f"{f*100:.0f}%", ha="center", fontsize=9)
    axE.set_xticks(x); axE.set_xticklabels(PHASES, fontsize=10)
    axE.set_ylabel("fraction of episode")
    axE.set_title("E. How long does each phase last?")
    axE.legend(fontsize=10)

    # ---------- F. Attention × activity correlation ----------
    axF = fig.add_subplot(gs[1, 2])
    phs = list(attn["per_phase"].keys())
    rs = [attn["per_phase"][p]["pearson_r"] for p in phs]
    ns_p = [attn["per_phase"][p]["n_steps"] for p in phs]
    colors_f = ["#888" if r < 0.1 else "#C44E52" for r in rs]
    x = np.arange(len(phs))
    bars = axF.bar(x, rs, color=colors_f, edgecolor="black", width=0.6)
    for i, (r, n) in enumerate(zip(rs, ns_p)):
        axF.text(i, r + (0.005 if r >= 0 else -0.015),
                 f"r={r:+.2f}\n(n={n:,})", ha="center", fontsize=9)
    axF.axhline(0, color="black", linewidth=0.8)
    axF.set_xticks(x); axF.set_xticklabels(phs, fontsize=10)
    axF.set_ylabel("Pearson r")
    axF.set_title("F. Does the model attend more to active sensors?\n"
                  "(per-step correlation, attention × sensor reading)")
    axF.set_ylim(min(rs)-0.05, max(rs)+0.07)

    fig.suptitle("PACT vs vanilla ACT — pick-and-place (mug, house_1, n=50 each)\n"
                 "Headline: tied success rate; the modality reshapes failures, not the rate.",
                 fontsize=14, y=0.995)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper-fig-v2] saved {out}")


if __name__ == "__main__":
    main()
