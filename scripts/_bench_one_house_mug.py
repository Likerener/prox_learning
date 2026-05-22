"""Quick datagen benchmark for FrankaSkinPickAndPlaceOneHouseMugConfig.

Runs a small N-episode collection so we can:
  1) confirm per-episode visual diversity (no longer the 250x-duplicate problem)
  2) measure wall-clock throughput before launching the full 250-ep run

Run:
  cd /home/jaydv/code/prox_learning
  export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  /opt/conda/envs/mlspaces/bin/python scripts/_bench_one_house_mug.py --n 5 --workers 4
"""
from __future__ import annotations
import argparse, datetime, os, time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="samples_per_house")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--multiplier", type=int, default=25)
    p.add_argument("--tag", type=str, default="bench")
    args = p.parse_args()

    # Import AFTER env vars are set by the launcher.
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaSkinPickAndPlaceOneHouseMugConfig,
    )
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    cfg = FrankaSkinPickAndPlaceOneHouseMugConfig()
    cfg.task_sampler_config.samples_per_house = args.n
    cfg.task_sampler_config.max_total_attempts_multiplier = args.multiplier
    cfg.num_workers = args.workers

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = Path("/home/jaydv/code/prox_learning/assets/datagen/_bench_one_house_mug") / f"{args.tag}_{ts}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_config()
    print(f"[bench] writing to {cfg.output_dir}")
    print(f"[bench] n={args.n} workers={args.workers} multiplier={args.multiplier}")

    t0 = time.time()
    runner = ParallelRolloutRunner(cfg)
    success, total = runner.run()
    dt = time.time() - t0
    per_ep = dt / max(success, 1)
    print(f"[bench] DONE in {dt:.1f}s  success={success}/{total}  per_success={per_ep:.1f}s")
    print(f"[bench] extrapolated 250 successes @ {args.workers}w: {(per_ep*250/3600):.2f}h")


if __name__ == "__main__":
    main()
