"""Convert the `mug_house_1_random_everything` datagen output to ACT-style data.

Source layout (one-episode-per-folder):
  assets/datagen/mug_house_1_random_everything/FrankaSkinPickAndPlacePilotMediumConfig/
    <timestamp>/house_1/
      trajectories_batch_1_of_1.h5            (contains a single traj_0)
      episode_00000000_exo_camera_1_batch_1_of_1.mp4
      episode_00000000_wrist_camera_batch_1_of_1.mp4
      episode_00000000_exo_camera_1_depth_batch_1_of_1.mp4   (ignored)
      episode_00000000_wrist_camera_depth_batch_1_of_1.mp4   (ignored)

Each folder is fed to `scripts.convert_pla_to_act`. That converter writes one
`episode_0.hdf5` (since each h5 only has traj_0). We rename it to a unique
`episode_<global_idx>.hdf5` in the destination as we go, so all 356 folders
land side-by-side in one ACT-ready directory.

Usage:
  python scripts/convert_mug_random_to_act.py \\
      --dst /home/jaydv/code/prox_learning/act_style_data/mug_house1_random_everything \\
      --image_h 240 --image_w 320
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SRC = (
    REPO_ROOT
    / "assets/datagen/mug_house_1_random_everything/FrankaSkinPickAndPlacePilotMediumConfig"
)

REQUIRED_FILES = [
    "trajectories_batch_1_of_1.h5",
    "episode_00000000_exo_camera_1_batch_1_of_1.mp4",
    "episode_00000000_wrist_camera_batch_1_of_1.mp4",
]


def folder_is_complete(folder: Path) -> bool:
    h1 = folder / "house_1"
    return h1.is_dir() and all((h1 / fn).exists() for fn in REQUIRED_FILES)


def list_complete_folders(src: Path) -> list[Path]:
    return sorted(d for d in src.iterdir() if d.is_dir() and folder_is_complete(d))


def _run_converter(src_h5: Path, dst: Path, image_h: int, image_w: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.convert_pla_to_act",
        "--src", str(src_h5),
        "--dst", str(dst),
        "--image_h", str(image_h),
        "--image_w", str(image_w),
    ]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="root holding the per-timestamp folders")
    p.add_argument("--dst", type=Path, required=True,
                   help="final act_style_data/<name>/ dir")
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--max_folders", type=int, default=None,
                   help="if set, only convert the first N folders (smoke test)")
    p.add_argument("--resume", action="store_true",
                   help="skip folders whose target episode_<idx>.hdf5 already exists")
    args = p.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"src not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    folders = list_complete_folders(src)
    if args.max_folders is not None:
        folders = folders[: args.max_folders]
    print(f"[convert] {len(folders)} complete folders under {src}", flush=True)

    n_done = 0
    n_skipped_resume = 0
    n_failed = 0
    for global_idx, folder in enumerate(folders):
        dst_ep = dst / f"episode_{global_idx}.hdf5"
        if args.resume and dst_ep.exists():
            n_skipped_resume += 1
            continue

        src_h5 = folder / "house_1" / "trajectories_batch_1_of_1.h5"
        stage = dst / f".stage_{folder.name}"
        if stage.exists():
            shutil.rmtree(stage)

        rc = _run_converter(src_h5, stage, args.image_h, args.image_w)
        if rc != 0:
            print(f"[convert] ! converter rc={rc} for {folder.name}", flush=True)
            shutil.rmtree(stage, ignore_errors=True)
            n_failed += 1
            continue

        produced = sorted(stage.glob("episode_*.hdf5"))
        if not produced:
            print(f"[convert] ! no episode produced for {folder.name}", flush=True)
            shutil.rmtree(stage, ignore_errors=True)
            n_failed += 1
            continue
        if len(produced) > 1:
            print(f"[convert] ! {folder.name} produced {len(produced)} episodes "
                  f"(expected 1); taking the first", flush=True)

        shutil.move(str(produced[0]), str(dst_ep))
        shutil.rmtree(stage, ignore_errors=True)
        n_done += 1
        if (global_idx + 1) % 25 == 0 or global_idx == len(folders) - 1:
            print(f"[convert] {global_idx + 1}/{len(folders)}  "
                  f"done={n_done} resumed_skip={n_skipped_resume} fail={n_failed}",
                  flush=True)

    print(f"[convert] DONE — wrote {n_done} new episodes, "
          f"{n_skipped_resume} resumed skips, {n_failed} failures into {dst}",
          flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
