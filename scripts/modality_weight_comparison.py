"""Modality weight comparison for PACT — RGB vs proximity vs robot state.

Two views:
  (A) Per-input-element weight magnitude (||W||_2 / sqrt(fan_in)). Tells you
      how strongly each *single element* of an input is projected into the
      residual stream. This is the "13x" comparison.
  (B) Total weight magnitude (raw ||W||_2). Tells you the cumulative size of
      each modality's input projection.

Depth panel is empty + annotated to be honest: the ACT model in this repo
ingests RGB only (conv1 is (64, 3, 7, 7)) — depth videos are recorded for
debug but not fed to the policy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def l2(t):
    return float(t.detach().to(torch.float32).norm(p=2).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # ---- Collect weights ----
    W_resnet_conv1 = sd["model.backbones.0.0.body.conv1.weight"]   # (64, 3, 7, 7) — RGB
    W_img_proj     = sd["model.input_proj.weight"]                  # (512, 512, 1, 1) — ResNet feat → hidden
    W_qpos         = sd["model.input_proj_robot_state.weight"]      # (512, 9)
    W_prox         = sd["model.input_proj_proximity.weight"]        # (512, 3)

    modalities = [
        ("RGB\n(post-ResNet projection)", W_img_proj.flatten(1), "#4C72B0"),
        ("Depth\n(NOT INPUT)",            None,                  "#CCCCCC"),
        ("Robot state\n(qpos)",           W_qpos,                "#55A868"),
        ("Proximity\n(per sensor, 3 D)",  W_prox,                "#C44E52"),
    ]

    per_in = []
    total  = []
    fan_in = []
    for name, W, c in modalities:
        if W is None:
            per_in.append(0.0); total.append(0.0); fan_in.append(0)
        else:
            n = l2(W); fi = W.shape[1]
            per_in.append(n / np.sqrt(fi))
            total.append(n)
            fan_in.append(fi)

    # ---- Plot ----
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels  = [m[0] for m in modalities]
    colors  = [m[2] for m in modalities]
    x = np.arange(len(modalities))

    # (A) per-element weight
    ax = axes[0]
    bars = ax.bar(x, per_in, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
    for i, v in enumerate(per_in):
        if v > 0:
            ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
        else:
            ax.text(i, 0.4, "not used by\nthe model", ha="center", fontsize=10, color="#555",
                    style="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("‖W‖₂ / √fan_in   (per-element weight magnitude)")
    ax.set_title("(A) Per-element weight on each input\n"
                 "How strongly is one scalar projected into the network?")
    ax.set_ylim(0, max(per_in) * 1.18)
    ax.grid(True, axis="y", alpha=0.25)
    # Annotate the 13× story
    if per_in[0] > 0 and per_in[3] > 0:
        ratio = per_in[3] / per_in[0]
        ax.annotate(
            f"proximity gets {ratio:.1f}× the weight of\n"
            "post-ResNet image features\n"
            "(per-element basis)",
            xy=(3, per_in[3]), xytext=(1.0, max(per_in) * 0.85),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=10, ha="left",
            bbox=dict(facecolor="#fff8dc", edgecolor="#bb9", pad=4))

    # (B) Total weight magnitude (raw L2)
    ax = axes[1]
    bars = ax.bar(x, total, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
    for i, (v, fi) in enumerate(zip(total, fan_in)):
        if v > 0:
            ax.text(i, v + max(total) * 0.02,
                    f"{v:.1f}\n(fan-in={fi})", ha="center", fontsize=10)
        else:
            ax.text(i, max(total) * 0.05, "not used", ha="center",
                    fontsize=10, color="#555", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("‖W‖₂   (total weight magnitude)")
    ax.set_title("(B) Total weight on each input projection\n"
                 "Image projection is largest in absolute terms because fan-in is 512")
    ax.set_ylim(0, max(total) * 1.18)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("PACT input-projection weights — what the model assigns to each modality\n"
                 "Source: runs/act_prox_mug_v1/policy_last.ckpt",
                 fontsize=13, y=0.99)

    # Note about depth at the bottom
    fig.text(0.5, -0.02,
             "Note: The vision encoder is a single ResNet18 with conv1 = (64, 3, 7, 7) — RGB only. "
             "Depth videos are recorded for analysis but never reach the policy.",
             ha="center", fontsize=9.5, color="#444", style="italic",
             bbox=dict(facecolor="#f6f6f6", edgecolor="#ccc", pad=4))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[modality-weight] saved {out}")
    print(f"\nPer-element weight ‖W‖₂/√fan_in:")
    for (name, _, _), v, fi in zip(modalities, per_in, fan_in):
        nm = name.split('\n')[0]
        print(f"  {nm:25s} : {v:.4f}   (fan_in={fi})")
    print(f"\nRatios (per-element):")
    if per_in[0] > 0:
        print(f"  proximity / image     = {per_in[3]/per_in[0]:6.2f}×")
        print(f"  qpos      / image     = {per_in[2]/per_in[0]:6.2f}×")
        print(f"  proximity / qpos      = {per_in[3]/per_in[2]:6.2f}×")


if __name__ == "__main__":
    main()
