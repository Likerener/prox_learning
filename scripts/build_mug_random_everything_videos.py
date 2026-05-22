"""First/last-frame videos for mug_house_1_random_everything dataset.

Walks every complete folder under
  assets/datagen/mug_house_1_random_everything/FrankaSkinPickAndPlacePilotMediumConfig/
in chronological order. For each one, samples the first frame (t=0) and the last
frame (t=T-1) from both cameras. Produces two side-by-side videos (exo|wrist):

  - eval_output/mug_random_everything_viz/_summary/first_frames.mp4
  - eval_output/mug_random_everything_viz/_summary/last_frames.mp4

A small overlay shows "ep <idx> / <total>   <folder_name>".

Optionally uploads the videos to the same wandb run created by
``visualize_mug_random_everything.py`` (project=prox_learning_dataset_viz,
name=mug_house1_random_everything) under keys summary/first_frames_video and
summary/last_frames_video.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


DATA_ROOT = Path(
    "/home/jaydv/code/prox_learning/assets/datagen/mug_house_1_random_everything/"
    "FrankaSkinPickAndPlacePilotMediumConfig"
)
OUT_DIR = Path(
    "/home/jaydv/code/prox_learning/eval_output/mug_random_everything_viz/_summary"
)

EXO_NAME = "episode_00000000_exo_camera_1_batch_1_of_1.mp4"
WRIST_NAME = "episode_00000000_wrist_camera_batch_1_of_1.mp4"
H5_NAME = "trajectories_batch_1_of_1.h5"
REQUIRED = [EXO_NAME, WRIST_NAME, H5_NAME]


def folder_is_complete(folder: Path) -> bool:
    h1 = folder / "house_1"
    return h1.is_dir() and all((h1 / fn).exists() for fn in REQUIRED)


def list_complete_folders() -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    return sorted(d for d in DATA_ROOT.iterdir() if d.is_dir() and folder_is_complete(d))


def grab_frame(path: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    target = max(0, min(idx if idx >= 0 else n + idx, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def grab_first_last(video_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok0, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    okL, last = cap.read()
    cap.release()
    return (first if ok0 else None), (last if okL else None)


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_side_by_side(exo: np.ndarray, wrist: np.ndarray, label: str) -> np.ndarray:
    if exo is None:
        exo = np.zeros((352, 624, 3), dtype=np.uint8)
    if wrist is None:
        wrist = np.zeros((352, 624, 3), dtype=np.uint8)
    if exo.shape != wrist.shape:
        wrist = cv2.resize(wrist, (exo.shape[1], exo.shape[0]))
    combo = np.concatenate([exo, wrist], axis=1)
    return annotate(combo, label)


def build_video(folders: list[Path], which: str, out_path: Path, fps: float) -> int:
    assert which in ("first", "last")
    writer = None
    n_written = 0
    total = len(folders)
    for ep_idx, folder in enumerate(folders):
        h1 = folder / "house_1"
        exo_first, exo_last = grab_first_last(h1 / EXO_NAME)
        wri_first, wri_last = grab_first_last(h1 / WRIST_NAME)
        exo = exo_first if which == "first" else exo_last
        wri = wri_first if which == "first" else wri_last
        if exo is None and wri is None:
            print(f"  ! skip ep {ep_idx:03d} {folder.name}: no decodable frames")
            continue
        label = f"ep {ep_idx:03d} / {total:03d}   {folder.name}   ({which})"
        combo = make_side_by_side(exo, wri, label)
        if writer is None:
            h, w = combo.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"  -> writing {out_path}  ({w}x{h} @ {fps} fps)")
        writer.write(combo)
        n_written += 1
        if (ep_idx + 1) % 50 == 0:
            print(f"    {which}: {ep_idx + 1}/{total} processed")
    if writer is not None:
        writer.release()
    return n_written


def maybe_log_wandb(args, first_path: Path, last_path: Path, n_first: int, n_last: int) -> None:
    if not args.wandb or not HAS_WANDB:
        return
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="dataset_viz",
            resume="allow",
        )
        print(f"[wandb] resuming run: {run.url}")
        payload = {
            "summary/first_frames_video": wandb.Video(str(first_path), fps=int(args.fps)),
            "summary/last_frames_video": wandb.Video(str(last_path), fps=int(args.fps)),
            "summary/first_frames_count": n_first,
            "summary/last_frames_count": n_last,
        }
        run.log(payload)
        run.finish()
    except Exception as e:
        print(f"[wandb] log failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Output video frame rate (each frame = one episode)")
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="prox_learning_dataset_viz")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default="mug_house1_random_everything")
    args = ap.parse_args()

    folders = list_complete_folders()
    print(f"found {len(folders)} complete folders in {DATA_ROOT}")
    if not folders:
        print("nothing to do")
        return

    first_path = OUT_DIR / "first_frames.mp4"
    last_path = OUT_DIR / "last_frames.mp4"

    print(f"\n[first frames] building {first_path}")
    n_first = build_video(folders, "first", first_path, args.fps)
    print(f"  wrote {n_first} frames")

    print(f"\n[last frames] building {last_path}")
    n_last = build_video(folders, "last", last_path, args.fps)
    print(f"  wrote {n_last} frames")

    print()
    maybe_log_wandb(args, first_path, last_path, n_first, n_last)
    print("done")


if __name__ == "__main__":
    main()
