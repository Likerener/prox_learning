"""Training-time visual degradation, applied per sample in the dataset.

Modes (TRAIN_DEGRADE env var), with strength from TRAIN_DEGRADE_P / _SIGMA:
    none            passthrough
    blur            Gaussian blur on every RGB frame (FACTR-style constant blur)
    camera_dropout  zero a whole camera with probability p (exo 2x more often
                    than wrist) — removes the sightline rather than the detail,
                    which is what makes proximity non-redundant
    occlusion       paste a random rectangle over part of the exo frame

Applied identically to the baseline and proximity policies, so the comparison
between them is unaffected by the choice of mode.
"""
import os

import cv2
import numpy as np

_MODE = os.environ.get("TRAIN_DEGRADE", "none")
_P = float(os.environ.get("TRAIN_DEGRADE_P", "0.3"))
_SIGMA = float(os.environ.get("TRAIN_DEGRADE_SIGMA", "4"))


def degrade(img, cam_name, rng=np.random):
    if _MODE == "none":
        return img
    if _MODE == "blur":
        k = int(_SIGMA * 3) | 1
        return cv2.GaussianBlur(img, (k, k), _SIGMA)
    if _MODE == "camera_dropout":
        p = _P if "exo" in cam_name else _P / 2
        return np.zeros_like(img) if rng.random() < p else img
    if _MODE == "occlusion":
        if "exo" not in cam_name or rng.random() >= _P:
            return img
        h, w = img.shape[:2]
        bh, bw = rng.randint(h // 4, h // 2), rng.randint(w // 4, w // 2)
        y, x = rng.randint(0, h - bh), rng.randint(0, w - bw)
        out = img.copy()
        out[y:y + bh, x:x + bw] = 0
        return out
    raise ValueError(f"unknown TRAIN_DEGRADE={_MODE}")
