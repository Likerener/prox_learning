"""Launch N parallel datagen subprocesses for FrankaSkinPickAndPlaceHouse10CupConfig.

The locked-down cup-pickup-in-house10 task. Each subprocess gets its own
`output_dir` and `seed` so workers don't collide on `house_10/episode_*.mp4`
(setup_house_dirs keys on house_id only).

Usage:
  cd /home/jaydv/code/prox_learning
  source /opt/conda/etc/profile.d/conda.sh && conda activate mlspaces
  export MLSPACES_ASSETS_DIR=/home/jaydv/code/prox_learning/assets
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  python scripts/run_house10_cup_parallel.py --jobs 4 --samples_per_job 63

  # Smoke (1 job × 2 ep, ~10 min):
  python scripts/run_house10_cup_parallel.py --jobs 1 --samples_per_job 2 --tag smoke
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_BASE = REPO_ROOT / "assets" / "datagen" / "pick_and_place_house10_cup_v1"

CHILD_INLINE = """
import sys, os
from pathlib import Path
os.environ.setdefault('MLSPACES_ASSETS_DIR', '{assets_dir}')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPickAndPlaceHouse10CupConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

cfg = FrankaSkinPickAndPlaceHouse10CupConfig()
cfg.task_sampler_config.samples_per_house = {samples_per_job}
cfg.num_workers = 1
cfg.seed = {seed}
cfg.output_dir = Path(r"{output_dir}")
cfg.output_dir.mkdir(parents=True, exist_ok=True)
cfg.save_config()
print(f"[w{worker_id}] output_dir={{cfg.output_dir}} seed={{cfg.seed}} target={{cfg.task_sampler_config.samples_per_house}}", flush=True)

runner = ParallelRolloutRunner(cfg)
success, total = runner.run()
print(f"[w{worker_id}] DONE success={{success}}/{{total}}", flush=True)
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4, help="number of parallel processes")
    p.add_argument(
        "--samples_per_job", type=int, default=63,
        help="samples_per_house per process; total dataset = jobs * samples_per_job",
    )
    p.add_argument("--base_seed", type=int, default=2026)
    p.add_argument("--tag", type=str, default="parallel")
    p.add_argument(
        "--output_base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="parent dir; each job writes to <base>/run_w<i>",
    )
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_base) / f"{args.tag}_{ts}"
    base.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] writing all jobs under {base}", flush=True)
    print(
        f"[launcher] jobs={args.jobs} samples_per_job={args.samples_per_job} "
        f"total_target={args.jobs * args.samples_per_job}",
        flush=True,
    )

    procs: list[tuple[int, subprocess.Popen, Path]] = []
    t0 = time.time()
    for i in range(args.jobs):
        out = base / f"run_w{i}"
        out.mkdir(parents=True, exist_ok=True)
        log_path = base / f"run_w{i}.log"
        inline = CHILD_INLINE.format(
            assets_dir=os.environ.get("MLSPACES_ASSETS_DIR", str(REPO_ROOT / "assets")),
            samples_per_job=args.samples_per_job,
            seed=args.base_seed + i,
            output_dir=str(out),
            worker_id=i,
        )
        log_fh = open(log_path, "w")
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", inline],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs.append((i, proc, log_path))
        print(f"[launcher] spawned w{i} pid={proc.pid} log={log_path}", flush=True)
        time.sleep(1.5)

    rc_total = 0
    for i, proc, log_path in procs:
        proc.wait()
        rc = proc.returncode
        rc_total |= rc
        dt = time.time() - t0
        print(f"[launcher] w{i} exited rc={rc} after {dt:.1f}s  log={log_path}", flush=True)

    dt = time.time() - t0
    print(
        f"[launcher] ALL DONE in {dt:.1f}s ({dt/3600:.2f}h) — "
        f"check logs under {base}",
        flush=True,
    )
    return 0 if rc_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
