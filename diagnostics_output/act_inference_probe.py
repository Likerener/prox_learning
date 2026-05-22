"""Direct inference probe for the v1 ACT checkpoint.

Loads the model EXACTLY as eval_act_house1.py:ACTInferencePolicy.prepare_model
does (including the _detr_argv shim), then runs three experiments:
  1) Training inputs at t in [0, 30, 46, 80, 130] -> compare predicted vs recorded.
  2) OOD constant-grey image with training qpos[0] -> what does model predict?
  3) Eval-initial qpos with training image[0] -> predicted action + gripper chunk.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch


CKPT_DIR = "/home/jaydv/code/prox_learning/submodules/act/ckpts/act_house1_mug_v1"
CKPT_PATH = str(Path(CKPT_DIR) / "policy_best.ckpt")
STATS_PATH = str(Path(CKPT_DIR) / "dataset_stats.pkl")
EPISODE_PATH = "/home/jaydv/code/prox_learning/act_style_data/pla_house1_mug_v1/episode_0.hdf5"

CAMERA_NAMES = ("exo_camera_1", "wrist_camera")
IMAGE_H = 240
IMAGE_W = 320
CHUNK_SIZE = 100
SEED = 0


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "act_inference_probe.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug",
        "--seed", str(seed),
        "--num_epochs", "1",
    ]
    try:
        yield
    finally:
        sys.argv = orig


def build_policy():
    # We must run with cwd = submodules/act so `from policy import ACTPolicy`
    # picks up the ACT-side module rather than pla.policy.
    from policy import ACTPolicy  # type: ignore
    from utils import set_seed  # type: ignore
    set_seed(SEED)
    policy_config = {
        "lr": 1e-5,
        "num_queries": CHUNK_SIZE,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": list(CAMERA_NAMES),
        "state_dim": 9,
        "action_dim": 8,
    }
    with _detr_argv(CKPT_DIR, SEED):
        policy = ACTPolicy(policy_config)
    sd = torch.load(CKPT_PATH, map_location="cuda")
    policy.load_state_dict(sd)
    policy.cuda()
    policy.eval()
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    print(f"[probe] loaded {CKPT_PATH}")
    print(f"[probe] stats keys: {list(stats.keys())}")
    print(f"[probe] qpos_mean: {stats['qpos_mean']}")
    print(f"[probe] qpos_std:  {stats['qpos_std']}")
    print(f"[probe] action_mean: {stats['action_mean']}")
    print(f"[probe] action_std:  {stats['action_std']}")
    return policy, stats


def load_step(h5, t: int):
    """Return raw qpos (9,), raw image dict (camera->(H,W,3) uint8), action (8,)."""
    qpos = np.asarray(h5["/observations/qpos"][t], dtype=np.float32)
    imgs = {}
    for cam in CAMERA_NAMES:
        imgs[cam] = np.asarray(h5[f"/observations/images/{cam}"][t])  # uint8 H,W,3
    action = np.asarray(h5["/action"][t], dtype=np.float32)
    return qpos, imgs, action


def preprocess(qpos_raw: np.ndarray, imgs: dict, stats: dict):
    """Apply eval-side preprocessing.

    qpos: subtract mean / divide std -> (1, 9) cuda
    image: uint8 -> /255 -> (C,H,W) -> stack cameras -> (1, num_cam, 3, H, W) cuda
    """
    qpos_norm = (qpos_raw - stats["qpos_mean"]) / stats["qpos_std"]
    qpos_t = torch.from_numpy(qpos_norm.astype(np.float32)).cuda().unsqueeze(0)

    cams = []
    for cam in CAMERA_NAMES:
        img = imgs[cam]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        # No resize needed: training data is already 240x320.
        cams.append(img.astype(np.float32) / 255.0)
    image = np.stack(cams, axis=0)                # (num_cam, H, W, 3)
    image = np.transpose(image, (0, 3, 1, 2))      # (num_cam, 3, H, W)
    image_t = torch.from_numpy(image).float().cuda().unsqueeze(0)
    return qpos_t, image_t


def main():
    policy, stats = build_policy()
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    print("\n=== EXPERIMENT 1: training inputs at t=[0,30,46,80,130] ===")
    with h5py.File(EPISODE_PATH, "r") as h5:
        T = h5["/observations/qpos"].shape[0]
        print(f"[probe] episode_0 length = {T}")
        cached_inputs_at_0 = None  # for experiments 2 and 3
        for t in [0, 30, 46, 80, 130]:
            if t >= T:
                print(f"  t={t}: out of range (T={T})")
                continue
            qpos_raw, imgs, action_recorded = load_step(h5, t)
            qpos_t, image_t = preprocess(qpos_raw, imgs, stats)
            with torch.no_grad():
                a_hat = policy(qpos_t, image_t)  # (1, chunk, 8)
            pred_norm = a_hat[0, 0].cpu().numpy()
            pred = pred_norm * action_std + action_mean
            diff = np.abs(pred - action_recorded)
            print(f"\n-- t={t} --")
            print(f"  qpos_raw       : {np.array2string(qpos_raw, precision=4, suppress_small=True)}")
            print(f"  recorded action: {np.array2string(action_recorded, precision=4, suppress_small=True)}")
            print(f"  predicted [0]  : {np.array2string(pred, precision=4, suppress_small=True)}")
            print(f"  abs diff       : {np.array2string(diff, precision=4, suppress_small=True)}")
            print(f"  max abs diff   : {diff.max():.4f}    mean abs diff: {diff.mean():.4f}")
            # gripper across the chunk
            chunk_grip_norm = a_hat[0, :, 7].cpu().numpy()
            chunk_grip = chunk_grip_norm * action_std[7] + action_mean[7]
            print(f"  gripper chunk min/max/mean: {chunk_grip.min():.2f} / {chunk_grip.max():.2f} / {chunk_grip.mean():.2f}")
            # which frame in the chunk first crosses 127.5?
            cross = np.where(chunk_grip > 127.5)[0]
            if cross.size > 0:
                print(f"  gripper>127.5 first crossing in chunk at offset k={cross[0]} (absolute frame t+k={t+cross[0]})")
            else:
                print(f"  gripper>127.5 NEVER in this chunk (chunk_size={chunk_grip.size})")
            if t == 0:
                cached_inputs_at_0 = (qpos_raw.copy(), {k: v.copy() for k, v in imgs.items()}, action_recorded.copy())

        # ----- EXPERIMENT 2: OOD grey image, training qpos[0] -----
        print("\n=== EXPERIMENT 2: OOD constant-grey image + training qpos[0] ===")
        qpos_raw_0, _, _ = cached_inputs_at_0
        grey = np.full((IMAGE_H, IMAGE_W, 3), 128, dtype=np.uint8)
        imgs_grey = {cam: grey for cam in CAMERA_NAMES}
        qpos_t, image_t = preprocess(qpos_raw_0, imgs_grey, stats)
        with torch.no_grad():
            a_hat = policy(qpos_t, image_t)
        pred_norm = a_hat[0, 0].cpu().numpy()
        pred = pred_norm * action_std + action_mean
        print(f"  predicted [0]  : {np.array2string(pred, precision=4, suppress_small=True)}")
        chunk_grip = a_hat[0, :, 7].cpu().numpy() * action_std[7] + action_mean[7]
        print(f"  gripper chunk min/max/mean: {chunk_grip.min():.2f} / {chunk_grip.max():.2f} / {chunk_grip.mean():.2f}")
        cross = np.where(chunk_grip > 127.5)[0]
        if cross.size > 0:
            print(f"  gripper>127.5 first at k={cross[0]}")
        else:
            print(f"  gripper>127.5 NEVER")

        # ----- EXPERIMENT 3: eval-initial qpos + training image[0] -----
        print("\n=== EXPERIMENT 3: eval-initial qpos + training image[0] ===")
        eval_qpos = np.array([0.0073, -0.832, -0.0455, -2.2608, 0.0494, 1.5034, -0.0251, 0.003, 0.003], dtype=np.float32)
        _, imgs_train_0, _ = cached_inputs_at_0
        qpos_t, image_t = preprocess(eval_qpos, imgs_train_0, stats)
        with torch.no_grad():
            a_hat = policy(qpos_t, image_t)
        pred_norm = a_hat[0, 0].cpu().numpy()
        pred = pred_norm * action_std + action_mean
        print(f"  eval qpos      : {np.array2string(eval_qpos, precision=4, suppress_small=True)}")
        print(f"  train qpos[0]  : {np.array2string(qpos_raw_0, precision=4, suppress_small=True)}")
        print(f"  predicted [0]  : {np.array2string(pred, precision=4, suppress_small=True)}")
        chunk_grip = a_hat[0, :, 7].cpu().numpy() * action_std[7] + action_mean[7]
        print(f"  gripper chunk full trajectory (every 5):")
        for i in range(0, chunk_grip.size, 5):
            print(f"    k={i:3d}  gripper={chunk_grip[i]:8.2f}")
        cross = np.where(chunk_grip > 127.5)[0]
        if cross.size > 0:
            print(f"  gripper>127.5 first at k={cross[0]}")
        else:
            print(f"  gripper>127.5 NEVER in chunk")
        # compare arm portion to training pred at t=0
        qpos_t_train, image_t_train = preprocess(qpos_raw_0, imgs_train_0, stats)
        with torch.no_grad():
            a_hat_train = policy(qpos_t_train, image_t_train)
        pred_train = a_hat_train[0, 0].cpu().numpy() * action_std + action_mean
        print(f"  pred(train qpos, train img) [0]: {np.array2string(pred_train, precision=4, suppress_small=True)}")
        print(f"  delta vs eval-qpos pred         : {np.array2string(pred - pred_train, precision=4, suppress_small=True)}")


if __name__ == "__main__":
    main()
