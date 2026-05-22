"""Merge the per-worker outputs from `run_v3_parallel.py` into a single
ACT-style training dataset under `act_style_data/`.

Source layout (after run_v3_parallel.py finishes):
  assets/datagen/pick_and_place_one_house_mug_v3/<tag>_<ts>/
    run_w0/house_1/{trajectories_batch_1_of_1.h5, episode_NNNNNNNN_*.mp4}
    run_w1/house_1/{trajectories_batch_1_of_1.h5, episode_NNNNNNNN_*.mp4}
    ...

The convert_pla_to_act.py converter operates on ONE source h5 + its sibling
MP4s. This wrapper runs the converter once per `run_w<i>/house_1/` and
writes into one destination dir, applying a global episode-index offset so
episode_<i>.hdf5 files don't collide.

Usage:
  python scripts/merge_v3_to_act_style.py \
      --src_base /home/jaydv/code/prox_learning/assets/datagen/pick_and_place_one_house_mug_v3/parallel_20260516_XXXXXX \
      --dst /home/jaydv/code/prox_learning/act_style_data/pla_house1_mug_v3 \
      --image_h 240 --image_w 320
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_converter(
    src_h5: Path,
    dst: Path,
    image_h: int,
    image_w: int,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.convert_pla_to_act",
        "--src",
        str(src_h5),
        "--dst",
        str(dst),
        "--image_h",
        str(image_h),
        "--image_w",
        str(image_w),
    ]
    print(f"[merge] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src_base", required=True, type=Path,
                   help="Parent dir written by run_v3_parallel.py")
    p.add_argument("--dst", required=True, type=Path,
                   help="Final act_style_data/<set> dir")
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    args = p.parse_args()

    src_base: Path = args.src_base.resolve()
    dst: Path = args.dst.resolve()
    if not src_base.is_dir():
        raise SystemExit(f"src_base not a directory: {src_base}")

    run_dirs = sorted(
        d for d in src_base.iterdir() if d.is_dir() and d.name.startswith("run_w")
    )
    if not run_dirs:
        raise SystemExit(f"no run_w* subdirs under {src_base}")

    # Stage each run into a separate temp dst, then rename episode_*.hdf5
    # files with a running offset so they don't collide in the final dst.
    dst.mkdir(parents=True, exist_ok=True)
    offset = 0
    for run_dir in run_dirs:
        house_dir = run_dir / "house_1"
        src_h5 = house_dir / "trajectories_batch_1_of_1.h5"
        if not src_h5.exists():
            print(f"[merge] skip {run_dir.name}: no trajectories h5", flush=True)
            continue
        # Convert into a per-run staging dir; then move into dst with offset.
        stage = src_base / f".stage_{run_dir.name}"
        if stage.exists():
            shutil.rmtree(stage)
        rc = _run_converter(src_h5, stage, args.image_h, args.image_w)
        if rc != 0:
            raise SystemExit(f"converter failed on {src_h5}")

        ep_files = sorted(
            stage.glob("episode_*.hdf5"),
            key=lambda p: int(p.stem.split("_", 1)[1]),
        )
        for src_ep in ep_files:
            local_idx = int(src_ep.stem.split("_", 1)[1])
            global_idx = offset + local_idx
            dst_ep = dst / f"episode_{global_idx}.hdf5"
            shutil.move(str(src_ep), str(dst_ep))
        offset += len(ep_files)
        shutil.rmtree(stage, ignore_errors=True)
        print(f"[merge] {run_dir.name}: +{len(ep_files)} episodes (total now {offset})",
              flush=True)

    print(f"[merge] DONE — {offset} episodes total under {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
