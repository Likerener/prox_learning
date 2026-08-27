"""Collect episodes from one datagen config, single worker.

Single worker on purpose: FrankaSkin* configs seed the task sampler once per
worker from a fixed config seed, so concurrent workers inside one run replay
identical episode streams. Run several of these processes instead, each with its
own seed and its own slice of houses.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import importlib
from pathlib import Path

from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

p = argparse.ArgumentParser()
p.add_argument("--config", default="FrankaSkinFumehoodSmokeConfig")
p.add_argument("--config_module",
               default="molmo_spaces.data_generation.config.object_manipulation_datagen_configs",
               help="module holding the config class")
p.add_argument("--houses", required=True, help="comma-separated house indices")
p.add_argument("--samples", type=int, default=20)
p.add_argument("--seed", type=int, default=None,
               help="omit to keep the config's own seed")
p.add_argument("--output_dir", required=True)
args = p.parse_args()

mod = importlib.import_module(args.config_module)
cfg = getattr(mod, args.config)()
if args.seed is not None:
    cfg.seed = args.seed
cfg.num_workers = 1
cfg.filter_for_successful_trajectories = True
cfg.output_dir = Path(args.output_dir).resolve()
cfg.output_dir.mkdir(parents=True, exist_ok=True)
cfg.task_sampler_config.house_inds = [int(h) for h in args.houses.split(",")]
cfg.task_sampler_config.samples_per_house = args.samples

cfg.save_config()
print(f"[collect] {args.config} houses={cfg.task_sampler_config.house_inds} "
      f"samples={args.samples} seed={args.seed} -> {cfg.output_dir}")
success, total = ParallelRolloutRunner(cfg).run()
print(f"[collect] success {success}/{total}")
