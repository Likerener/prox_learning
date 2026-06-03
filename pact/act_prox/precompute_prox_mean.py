"""Precompute the per-sensor mean predicted 3-D position over the training set.

Runs the FROZEN prox-encoder on training proximity windows sampled from
the `prox_mapping.json`-linked source h5s and averages the predicted
positions over many (B*N) samples. Outputs a `.npy` of shape
`(n_sensors, 3)` in metres.

This is used by `eval_act_with_prox_encoder.py --mask_proximity mean` as a
sanity-check replacement for prox_pos (vs. the harsher `zero` baseline).

Run:
    /opt/conda/envs/mlspaces/bin/python pact/act_prox/precompute_prox_mean.py \\
        --act_dataset_dir act_style_data/mug_house1_random_everything \\
        --prox_mapping_json act_style_data/mug_house1_random_everything/prox_mapping.json \\
        --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \\
        --output pact/outputs_prox/runs/prox_encoder_v1/prox_pos_mean.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Make the pact + act submodule paths importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "submodules" / "act") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "submodules" / "act"))

from pact.act_prox.prox_features import FrozenProxFeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--act_dataset_dir", required=True,
                   help="ACT episodes dir (e.g. act_style_data/mug_house1_random_everything)")
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--prox_encoder_ckpt", required=True)
    p.add_argument("--output", required=True,
                   help="Output .npy path; shape will be (n_sensors, 3) in metres.")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Number of (episode, start_ts) samples to draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    with open(args.prox_mapping_json, "r") as f:
        mapping = json.load(f)
    sensor_names = list(mapping["sensor_names"])
    n_sensors = int(mapping["n_sensors"])
    eps = mapping["episodes"]
    ep_keys = sorted(eps.keys(), key=lambda x: int(x))
    print(f"[prox-mean] mapping has {len(ep_keys)} episodes, {n_sensors} sensors")

    extractor = FrozenProxFeatureExtractor(args.prox_encoder_ckpt, device=device)
    W = int(extractor.window)
    prox_mean = extractor.prox_mean.cpu().numpy().astype(np.float32)   # (4, 8, 8)
    prox_std  = extractor.prox_std.cpu().numpy().astype(np.float32)    # (4, 8, 8)

    # Sample (episode, t) pairs uniformly.
    sample_indices = rng.choice(len(ep_keys), size=args.n_samples, replace=True)

    accum = np.zeros((n_sensors, 3), dtype=np.float64)
    count = 0
    # Open source files lazily.
    src_handles: dict[str, h5py.File] = {}

    def _open(p: str) -> h5py.File:
        h = src_handles.get(p)
        if h is None:
            h = h5py.File(p, "r")
            src_handles[p] = h
        return h

    for k, ep_idx in enumerate(sample_indices):
        ek = ep_keys[ep_idx]
        entry = eps[ek]
        src = _open(entry["source_h5"])
        traj = src[entry["traj_key"]]
        # Pick a step.
        T = traj[f"obs/proximity/{sensor_names[0]}"].shape[0]
        t = int(rng.integers(0, T))
        lo = max(0, t - W + 1)
        hi = t + 1
        n_real = hi - lo
        n_pad = W - n_real

        window = np.empty((n_sensors, W, 4, 8, 8), dtype=np.float32)
        for i, sn in enumerate(sensor_names):
            full = traj[f"obs/proximity/{sn}"][lo:hi]
            if n_pad > 0:
                pad = np.repeat(full[:1], n_pad, axis=0)
                full = np.concatenate([pad, full], axis=0)
            window[i] = full
        # z-score with encoder stats.
        window = (window - prox_mean[None, None]) / prox_std[None, None]
        window = window.reshape(n_sensors, W * 4, 8, 8)

        x = torch.from_numpy(window).unsqueeze(0).to(device)            # (1, N, W*4, 8, 8)
        with torch.no_grad():
            pred = extractor(x)                                          # (1, N, 3) metres
        accum += pred.squeeze(0).cpu().numpy().astype(np.float64)
        count += 1
        if (k + 1) % 64 == 0:
            print(f"[prox-mean] processed {k+1}/{args.n_samples}", flush=True)

    mean = (accum / count).astype(np.float32)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mean)

    norms = np.linalg.norm(mean, axis=1)
    print(f"[prox-mean] saved {out}  shape={mean.shape}  "
          f"||mean|| min/median/max = {norms.min():.3f}/{np.median(norms):.3f}/{norms.max():.3f} m")
    print("[prox-mean] mean (per-sensor xyz, metres):")
    for i, sn in enumerate(sensor_names):
        print(f"  {sn}: x={mean[i,0]:+.3f}  y={mean[i,1]:+.3f}  z={mean[i,2]:+.3f}")


if __name__ == "__main__":
    main()
