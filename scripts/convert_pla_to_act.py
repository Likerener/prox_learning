"""Convert a molmospaces PLA dataset (h5 + sibling MP4s) into the per-episode
HDF5 layout expected by the upstream ACT trainer (`submodules/act/utils.py`).

Source layout (one batch file under the datagen dir):
  - trajectories_batch_1_of_1.h5: groups traj_0, traj_1, ... each with
      obs/agent/qpos      (T, 2000) uint8  — JSON byte rows {"arm":[7], "gripper":[2]}
      obs/agent/qvel      (T, 2000) uint8  — JSON byte rows {"arm":[7], "gripper":[2]}
      actions/joint_pos   (T, 2000) uint8  — JSON byte rows {"arm":[7], "gripper":[1]}
  - episode_<i:08d>_<cam>_batch_1_of_1.mp4   — RGB frames per camera, T frames each

Target layout (one file per episode, ACT-compatible):
  episode_<i>.hdf5
    attrs['sim'] = True
    /action                                 (T, 8)         arm(7)+gripper_cmd(1)
    /observations/qpos                       (T, 9)         arm(7)+gripper(2)
    /observations/qvel                       (T, 9)         arm(7)+gripper(2)
    /observations/images/exo_camera_1        (T, H, W, 3)   uint8
    /observations/images/wrist_camera        (T, H, W, 3)   uint8

Action layout note: the last timestep's joint_pos row in the source is `{}` (no
command issued), so we drop it. This makes the per-episode T be `T_h5 - 1`.

Run:
    python -m scripts.convert_pla_to_act \
        --src /home/jaydv/code/prox_learning/assets/datagen/pick_and_place_one_house_mug_dup250_v1/FrankaSkinPickAndPlaceOneHouseMugDup250Config/20260515_010000/house_1/trajectories_batch_1_of_1.h5 \
        --dst /home/jaydv/code/prox_learning/act_style_data/pla_house1_mug_v1 \
        --image_h 240 --image_w 320
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CAM_NAMES = ("exo_camera_1", "wrist_camera")
ARM_DIM = 7
GRIP_DIM_OBS = 2     # qpos/qvel: two finger joints
GRIP_DIM_ACT = 1     # joint_pos action: one gripper command
QPOS_DIM = ARM_DIM + GRIP_DIM_OBS
ACTION_DIM = ARM_DIM + GRIP_DIM_ACT


def _decode_jsonrow(blob: np.ndarray) -> dict:
    raw = bytes(blob).split(b"\x00", 1)[0]
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _decode_qpos_qvel(blob: np.ndarray) -> np.ndarray:
    """Decode an arm+gripper JSON row into a 9-d vector."""
    out = np.zeros(QPOS_DIM, dtype=np.float32)
    try:
        d = _decode_jsonrow(blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return out
    arm = d.get("arm") or []
    grip = d.get("gripper") or []
    out[: min(len(arm), ARM_DIM)] = arm[:ARM_DIM]
    out[ARM_DIM : ARM_DIM + min(len(grip), GRIP_DIM_OBS)] = grip[:GRIP_DIM_OBS]
    return out


def _decode_action(blob: np.ndarray) -> tuple[np.ndarray, bool]:
    """Decode a joint_pos action row into an 8-d vector.

    Returns (vec, is_valid). The trailing row in each trajectory is `{}` and
    must be filtered by the caller.
    """
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    try:
        d = _decode_jsonrow(blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return out, False
    arm = d.get("arm") or []
    if not arm:
        return out, False
    out[: min(len(arm), ARM_DIM)] = arm[:ARM_DIM]
    grip = d.get("gripper") or []
    if grip:
        out[ARM_DIM] = float(grip[0])
    return out, True


def _video_frames(path: Path, image_h: int | None, image_w: int | None) -> np.ndarray:
    """Decode every frame of an MP4 into an (N, H, W, 3) uint8 array (RGB)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()  # BGR
            if not ok:
                break
            if image_h is not None and image_w is not None and frame.shape[:2] != (image_h, image_w):
                frame = cv2.resize(frame, (image_w, image_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{path} decoded to 0 frames")
    return np.stack(frames, axis=0)


def convert(
    src_h5: Path,
    dst_dir: Path,
    image_h: int | None,
    image_w: int | None,
    max_episodes: int | None,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    h5_parent = src_h5.parent

    with h5py.File(src_h5, "r") as src:
        traj_keys = sorted(src.keys(), key=lambda k: int(k.split("_", 1)[1]))
        if max_episodes is not None:
            traj_keys = traj_keys[:max_episodes]

        for traj_key in tqdm(traj_keys, desc="episodes"):
            ep_idx = int(traj_key.split("_", 1)[1])
            grp = src[traj_key]
            T_full = grp["actions/joint_pos"].shape[0]

            # Decode actions; drop trailing invalid rows.
            actions = np.zeros((T_full, ACTION_DIM), dtype=np.float32)
            valid = np.zeros(T_full, dtype=bool)
            for t in range(T_full):
                a, ok = _decode_action(grp["actions/joint_pos"][t])
                actions[t] = a
                valid[t] = ok
            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"[skip] {traj_key}: no valid action rows")
                continue
            # ACT trains on aligned (qpos_t, action_t..t+k) so keep the same prefix
            # for qpos/qvel/images.
            T = n_valid
            actions = actions[:T]

            qpos = np.stack([_decode_qpos_qvel(grp["obs/agent/qpos"][t]) for t in range(T)])
            qvel = np.stack([_decode_qpos_qvel(grp["obs/agent/qvel"][t]) for t in range(T)])

            # Load camera videos.
            images = {}
            for cam in CAM_NAMES:
                mp4 = h5_parent / f"episode_{ep_idx:08d}_{cam}_batch_1_of_1.mp4"
                if not mp4.exists():
                    raise FileNotFoundError(f"missing video for {traj_key}: {mp4}")
                frames = _video_frames(mp4, image_h, image_w)
                if frames.shape[0] < T:
                    raise ValueError(
                        f"{traj_key}/{cam}: only {frames.shape[0]} frames < required {T}"
                    )
                images[cam] = frames[:T]

            out_path = dst_dir / f"episode_{ep_idx}.hdf5"
            with h5py.File(out_path, "w") as dst:
                dst.attrs["sim"] = True
                dst.create_dataset("action", data=actions, dtype="float32")
                obs = dst.create_group("observations")
                obs.create_dataset("qpos", data=qpos.astype(np.float32))
                obs.create_dataset("qvel", data=qvel.astype(np.float32))
                imgs = obs.create_group("images")
                for cam, arr in images.items():
                    # chunk per-frame to keep random-access reads cheap during training
                    imgs.create_dataset(
                        cam,
                        data=arr,
                        dtype="uint8",
                        chunks=(1, arr.shape[1], arr.shape[2], 3),
                        compression="gzip",
                        compression_opts=4,
                    )

    print(f"wrote {len(traj_keys)} episodes to {dst_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="trajectories_batch_1_of_1.h5")
    p.add_argument("--dst", type=Path, required=True, help="output dir for episode_*.hdf5")
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--max_episodes", type=int, default=None)
    args = p.parse_args()
    convert(args.src, args.dst, args.image_h, args.image_w, args.max_episodes)


if __name__ == "__main__":
    main()
