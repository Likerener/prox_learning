"""Weight-usage plot for P+ACT (and optionally vanilla ACT).

Three complementary views of "how the network uses its weights":

  (A) Input-projection norms by modality
        Compares the L2 norm of `input_proj`        (image features → hidden_dim)
                            `input_proj_robot_state` (qpos      → hidden_dim)
                            `input_proj_proximity`  (3-D prox  → K*hidden_dim)
      after normalising for the fan-in of each layer (i.e. average per-input-
      element norm). Tells us how strongly each modality is initially projected
      into the residual stream.

  (B) Per-layer total parameter norm
        Aggregate L2 norm of all trainable params in each module (encoder
        layer i, decoder layer i, action_head, input_proj_proximity, ...).
      Shows where parameter "mass" sits in the network.

  (C) Per-sensor input_proj_proximity row norm
        For the proximity input layer, the row norm of each sensor's 3-D →
        K*hidden_dim projection. Tells us whether the model weights some
        sensors (e.g. wrist link6) more than others at the input stage,
        a static counterpart to the attention-mass evidence.

Run:
    /opt/conda/envs/mlspaces/bin/python scripts/plot_weight_usage.py \\
        --ckpt runs/act_prox_mug_v1/policy_best.ckpt \\
        --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \\
        --out eval_output/weight_usage.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="P+ACT policy_best.ckpt")
    p.add_argument("--prox_mapping_json", required=True,
                   help="To label per-sensor rows in panel (C)")
    p.add_argument("--out", required=True,
                   help="output PNG")
    p.add_argument("--vanilla_ckpt", default="",
                   help="Optional vanilla ACT ckpt for side-by-side input-norm comparison")
    return p.parse_args()


def _l2(t: torch.Tensor) -> float:
    return float(t.detach().to(torch.float32).norm(p=2).item())


def _per_row_l2(W: torch.Tensor) -> np.ndarray:
    """Row-wise L2 of (out, in) weight matrix → returns (out,)."""
    return W.detach().to(torch.float32).norm(p=2, dim=1).cpu().numpy()


def collect_input_projection_norms(sd: dict, label: str) -> dict:
    """Compute L2 norms for input_proj, input_proj_robot_state, input_proj_proximity
    AND normalised-by-fan-in scores."""
    out = {"label": label}

    def grab(key: str):
        if key in sd:
            W = sd[key]
            n = _l2(W)
            fan_in = W.shape[1] if W.dim() >= 2 else W.numel()
            return n, fan_in, W
        return None, None, None

    # image: input_proj is a 1×1 conv (out, in_ch, 1, 1) for ACT
    n_img, fi_img, W_img = grab("model.input_proj.weight")
    n_qpos, fi_qpos, W_qpos = grab("model.input_proj_robot_state.weight")
    n_prox, fi_prox, W_prox = grab("model.input_proj_proximity.weight")

    out["image"] = {"norm": n_img, "fan_in": fi_img, "per_in": (n_img / np.sqrt(fi_img)) if n_img else None}
    out["qpos"]  = {"norm": n_qpos, "fan_in": fi_qpos, "per_in": (n_qpos / np.sqrt(fi_qpos)) if n_qpos else None}
    out["prox"]  = {"norm": n_prox, "fan_in": fi_prox, "per_in": (n_prox / np.sqrt(fi_prox)) if n_prox else None}
    return out


def collect_layer_norms(sd: dict) -> dict:
    """Aggregate L2 norm of params within each named module (e.g.
    transformer.encoder.layers.0, decoder.layers.3, action_head, ...).
    Returns a dict {module_name: norm}.
    """
    buckets: dict[str, float] = {}
    for name, t in sd.items():
        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            continue
        # Group by top-3 path segments, e.g. model.transformer.encoder.layers.0
        parts = name.split(".")
        if len(parts) >= 5 and parts[1] == "transformer" and parts[3] == "layers":
            bucket = ".".join(parts[1:5])
        elif name.startswith("model.input_proj") or name.startswith("input_proj"):
            bucket = parts[1] if name.startswith("model.") else parts[0]
        elif name.startswith("model.additional_pos_embed"):
            bucket = "additional_pos_embed"
        elif name.startswith("model.pos_table"):
            bucket = "pos_table"
        elif name.startswith("model.backbones"):
            bucket = ".".join(parts[1:3])
        elif name.startswith("model.action_head") or name.startswith("model.is_pad_head"):
            bucket = parts[1]
        elif name.startswith("model.query_embed"):
            bucket = "query_embed"
        elif name.startswith("model.latent_proj") or name.startswith("model.latent_out_proj"):
            bucket = "latent"
        elif name.startswith("model.encoder_action_proj") or name.startswith("model.encoder_joint_proj") or name.startswith("model.cls_embed"):
            bucket = "vae_encoder_proj"
        else:
            bucket = name.split(".")[0] if not name.startswith("model.") else parts[1]
        n2 = float(t.detach().to(torch.float32).pow(2).sum().item())
        buckets[bucket] = buckets.get(bucket, 0.0) + n2
    return {k: np.sqrt(v) for k, v in buckets.items()}


def per_sensor_prox_proj_norms(sd: dict, sensor_names: list[str]) -> np.ndarray:
    """For the proximity input projection (shape: (K*hidden_dim, 3)), the per-sensor
    norm is the row-norm of the 3-D → K*hidden_dim weight. But ACT applies the SAME
    Linear(3 → K*hidden_dim) to every sensor — so per-sensor differences arise from
    `additional_pos_embed` (K extra entries per sensor) + the prox tokens' contribution
    to subsequent layers. We surface BOTH:
      * input_proj_proximity row-norm (shared across sensors)
      * additional_pos_embed[2 + sensor_offset : 2 + sensor_offset + K] norm per sensor.
    """
    # Linear weight is (K*hidden_dim, 3) — row L2.
    proj_row_norms = _per_row_l2(sd["model.input_proj_proximity.weight"])
    # additional_pos_embed weight is (2 + n_prox_tokens, hidden_dim)
    pe = sd["model.additional_pos_embed.weight"].detach().to(torch.float32)   # (2+P, H)
    P = pe.shape[0] - 2
    K = P // len(sensor_names)
    # Per-sensor: sum of K consecutive position-embedding rows' L2 norms.
    sensor_pe = pe[2:].reshape(len(sensor_names), K, -1)                       # (N, K, H)
    sensor_norms = sensor_pe.norm(p=2, dim=(1, 2)).cpu().numpy()
    return proj_row_norms, sensor_norms


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt  # Some ACT checkpoints are state_dicts directly.
    print(f"[weight-usage] loaded ckpt {args.ckpt}, {len(sd)} entries")

    with open(args.prox_mapping_json) as f:
        mp = json.load(f)
    sensor_names = list(mp["sensor_names"])

    pact = collect_input_projection_norms(sd, "P+ACT")
    print(f"[weight-usage] P+ACT input-proj norms:")
    for k in ("image", "qpos", "prox"):
        v = pact[k]
        print(f"  {k}: ||W||={v['norm']}, fan_in={v['fan_in']}, ||W||/√fan_in={v['per_in']}")

    layer_norms = collect_layer_norms(sd)
    print(f"[weight-usage] {len(layer_norms)} layer buckets")

    proj_rows, sensor_pe = per_sensor_prox_proj_norms(sd, sensor_names)
    print(f"[weight-usage] input_proj_proximity row-norms: "
          f"min={proj_rows.min():.3f}, median={np.median(proj_rows):.3f}, max={proj_rows.max():.3f}")
    print(f"[weight-usage] additional_pos_embed per-sensor norms: "
          f"min={sensor_pe.min():.3f}, median={np.median(sensor_pe):.3f}, max={sensor_pe.max():.3f}")

    # Optional vanilla comparison.
    vanilla = None
    if args.vanilla_ckpt and Path(args.vanilla_ckpt).exists():
        sd_v = torch.load(args.vanilla_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd_v, dict) and "model" in sd_v:
            sd_v = sd_v["model"]
        vanilla = collect_input_projection_norms(sd_v, "vanilla ACT")
        print(f"[weight-usage] vanilla ACT input-proj norms:")
        for k in ("image", "qpos"):
            v = vanilla[k]
            print(f"  {k}: ||W||={v['norm']}, fan_in={v['fan_in']}, ||W||/√fan_in={v['per_in']}")

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Panel A: input-projection norms by modality
    ax = fig.add_subplot(gs[0, 0])
    modalities = ["image", "qpos", "prox"]
    pact_per_in = [pact[m]["per_in"] for m in modalities]
    x = np.arange(len(modalities))
    width = 0.36 if vanilla else 0.55
    bars1 = ax.bar(x - (width / 2 if vanilla else 0), pact_per_in, width,
                   label="P+ACT", color="#1f77b4", edgecolor="black", linewidth=0.6)
    if vanilla:
        v_per_in = [vanilla[m]["per_in"] if vanilla[m]["per_in"] else 0.0 for m in modalities]
        bars2 = ax.bar(x + width / 2, v_per_in, width,
                       label="vanilla ACT", color="#5e5e5e", edgecolor="black", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=9)
    for i, val in enumerate(pact_per_in):
        ax.text(x[i] - (width / 2 if vanilla else 0), val * 1.02 if val else 0.0,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("||W||₂ / √fan_in")
    ax.set_title("(A) Per-modality input-projection magnitude (normalised by fan-in)")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel B: per-layer total norm
    ax = fig.add_subplot(gs[0, 1])
    # Sort buckets by name with encoder/decoder grouped.
    items = sorted(layer_norms.items(), key=lambda kv: kv[0])
    names_b = [k for k, _ in items]
    vals_b  = [v for _, v in items]
    color_for = lambda n: ("#1f77b4" if "encoder" in n else
                           "#ff7f0e" if "decoder" in n else
                           "#9467bd" if "prox" in n else
                           "#5e5e5e")
    colors = [color_for(n) for n in names_b]
    ax.barh(range(len(names_b)), vals_b, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(names_b)))
    ax.set_yticklabels(names_b, fontsize=8)
    ax.set_xlabel("L2 norm of all trainable params in module")
    ax.set_title("(B) Per-module weight mass")
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()

    # Panel C: per-sensor position-embedding norm (a per-sensor "weight" signal).
    ax = fig.add_subplot(gs[1, :])
    order = np.argsort(sensor_pe)
    sensors_sorted = [sensor_names[i] for i in order]
    norms_sorted   = sensor_pe[order]
    # Color by link group.
    def link_color(name):
        if "link2" in name: return "#1f77b4"
        if "link3" in name: return "#2ca02c"
        if "link5" in name: return "#ff7f0e"
        if "link6" in name: return "#d62728"
        return "#5e5e5e"
    colors_c = [link_color(s) for s in sensors_sorted]
    ax.bar(range(len(sensors_sorted)), norms_sorted, color=colors_c,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sensors_sorted)))
    ax.set_xticklabels(sensors_sorted, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("||additional_pos_embed||₂ per sensor")
    ax.set_title("(C) Per-sensor positional embedding magnitude — "
                 f"sorted; min {norms_sorted[0]:.2f}, max {norms_sorted[-1]:.2f}")
    ax.grid(True, axis="y", alpha=0.25)
    # Legend handles
    handles = [plt.Rectangle((0,0),1,1, color=link_color(f"link{l}_sensor_0"), label=f"link{l}")
               for l in (2,3,5,6)]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.suptitle(
        f"P+ACT weight usage — input-proj magnitudes, per-module norms, per-sensor PE norms",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[weight-usage] saved {out}")


if __name__ == "__main__":
    main()
