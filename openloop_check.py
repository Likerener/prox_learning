"""Open-loop sanity check: does the trained policy reproduce demo actions?

No simulator, no rendering. Feeds recorded observations from the ACT-style
dataset through the checkpoint and compares the predicted action chunk against
the recorded actions. If the error is small, the model learned the task and any
rollout failure is a closed-loop problem (horizon, compounding error). If the
error is large, the problem is the model or how it is being loaded.
"""
import glob
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

PL = Path(__file__).resolve().parent
sys.path.insert(0, str(PL / "submodules" / "act"))
sys.path.insert(0, str(PL / "submodules" / "act" / "detr"))

import ckpt_dims  # noqa: E402


@contextmanager
def detr_argv(ckpt_dir, n_prox=0):
    orig = sys.argv
    sys.argv = [orig[0], "--ckpt_dir", str(ckpt_dir), "--policy_class", "ACT",
                "--task_name", "openloop", "--seed", "0", "--num_epochs", "1"]
    if n_prox:
        sys.argv += ["--n_proximity_sensors", str(n_prox)]
    try:
        yield
    finally:
        sys.argv = orig


ckpt_dir = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
dims = ckpt_dims.infer(str(ckpt_dir / "policy_best.ckpt"))
print("dims:", dims)
if dims["n_proximity_sensors"]:
    print("NOTE: proximity checkpoint — zero proximity is used here, so this is a "
          "lower bound on its accuracy.")

from policy import ACTPolicy  # noqa: E402

cfg = dict(lr=1e-5, num_queries=dims["chunk_size"], kl_weight=10,
           hidden_dim=dims["hidden_dim"], dim_feedforward=dims["dim_feedforward"],
           lr_backbone=1e-5, backbone="resnet18", enc_layers=dims["enc_layers"],
           dec_layers=dims["dec_layers"], nheads=8,
           camera_names=["exo_camera_1", "wrist_camera"],
           state_dim=dims["state_dim"], action_dim=dims["action_dim"])
with detr_argv(ckpt_dir, dims["n_proximity_sensors"]):
    pol = ACTPolicy(cfg)
pol.load_state_dict(torch.load(ckpt_dir / "policy_best.ckpt", map_location="cuda"))
pol.cuda().eval()
stats = pickle.load(open(ckpt_dir / "dataset_stats.pkl", "rb"))

eps = sorted(glob.glob(str(data_dir / "episode_*.hdf5")))[:12]
errs, first_errs = [], []
for p in eps:
    with h5py.File(p) as f:
        cams = list(f["/observations/images"].keys())
        acts = f["/action"][:]
        qpos = f["/observations/qpos"][:]
        T = acts.shape[0]
        for t in range(0, max(1, T - dims["chunk_size"]), 5):
            q = (qpos[t] - stats["qpos_mean"]) / stats["qpos_std"]
            imgs = []
            for c in ["exo_camera_1", "wrist_camera"]:
                im = f[f"/observations/images/{c}"][t]
                if im.shape[:2] != (240, 320):
                    im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
                imgs.append(im.astype(np.float32) / 255.0)
            img_t = torch.from_numpy(
                np.transpose(np.stack(imgs), (0, 3, 1, 2))).float().cuda().unsqueeze(0)
            q_t = torch.from_numpy(q).float().cuda().unsqueeze(0)
            kw = {}
            if dims["n_proximity_sensors"]:
                kw["proximity_positions"] = torch.zeros(
                    1, dims["n_proximity_sensors"], 3).cuda()
            with torch.no_grad():
                pred = pol(q_t, img_t, **kw).squeeze(0).cpu().numpy()
            pred = pred * stats["action_std"] + stats["action_mean"]
            tgt = acts[t:t + dims["chunk_size"]]
            errs.append(np.abs(pred[:len(tgt)] - tgt).mean(0))
            first_errs.append(np.abs(pred[0] - acts[t]))
    print(f"  {Path(p).name}: cams={cams} T={T}")

errs = np.array(errs)
first = np.array(first_errs)
print()
print("mean |error| over the whole chunk, per action dim:")
print("  ", np.round(errs.mean(0), 4))
print("mean |error| for the immediate next action:")
print("  ", np.round(first.mean(0), 4))
print("action_std (for scale):")
print("  ", np.round(stats["action_std"], 4))
print()
ratio = first.mean(0)[:7] / stats["action_std"][:7]
print("arm error / action_std =", np.round(ratio, 3), " (<<1 means the policy tracks demos)")
print("gripper: mean |error| =", round(float(first.mean(0)[7]), 1), "on a 0..255 scale")
