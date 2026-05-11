"""molmospaces-compatible inference wrapper for the PLA / VLM-only ACT policy.

`PLAInferencePolicy` plugs into `molmo_spaces.evaluation.run_evaluation` via a
`BasePolicyConfig` subclass (see `pla/eval.py`). It loads a checkpoint saved
by `pla.train`, reads the standard franka_skin observations exposed by the
sim env, and returns per-step `{"arm": [7], "gripper": [1]}` actions.

Action chunking (chunk_size=100) is buffered: we predict a fresh chunk each
time the buffer drains, then replay it one step at a time. This is the
standard ACT inference recipe — temporal aggregation is intentionally NOT
used here (matches upstream ACTPolicy default).

Gripper handling: the TODO spec is `action [T, 7]` (arm only). A pick-and-
place rollout still needs gripper open/close, so we expose
`gripper_schedule`:
  - "open"     : always-open (debug; will not grasp anything)
  - "qpos"     : echo the current qpos gripper value (no-op control)
  - "binary_t" : open until step `gripper_close_step`, then close
This is a known limitation; once we extend the action head to 8 dims, the
network predicts gripper directly.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from molmo_spaces.policy.base_policy import InferencePolicy

from pla.dataset import SENSOR_NAMES
from pla.policy import PLAConfig, PLAPolicy

log = logging.getLogger(__name__)


class PLAInferencePolicy(InferencePolicy):
    """ACT-style learned policy with optional 29-sensor proximity context."""

    def __init__(self, exp_config, task=None) -> None:
        # setup_policy() in pipeline.py:132 calls policy_cls(exp_config, task),
        # but InferencePolicy.__init__ drops the task arg. Accept it here and set
        # via BasePolicy directly so self.task is preserved for the rollout loop.
        super().__init__(exp_config)
        self.task = task
        self.policy_config = exp_config.policy_config
        self.checkpoint_path = self.policy_config.checkpoint_path
        self.use_proximity = self.policy_config.use_proximity
        self.image_h = self.policy_config.image_h
        self.image_w = self.policy_config.image_w
        self.depth_max_m = self.policy_config.depth_max_m
        self.gripper_schedule = self.policy_config.gripper_schedule
        self.gripper_close_step = self.policy_config.gripper_close_step
        self.gripper_threshold = getattr(self.policy_config, "gripper_threshold", 0.5)

        self.model: PLAPolicy | None = None
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0
        self._step = 0

    # ------------------------------------------------------------------
    # InferencePolicy abstract methods
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._chunk = None
        self._chunk_idx = 0
        self._step = 0

    def prepare_model(self, model_name: str | None = None) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location="cuda")
        cfg_dict = ckpt.get("policy_cfg", {})
        cfg_dict["use_proximity"] = self.use_proximity  # respect eval-time override
        cfg = PLAConfig(**cfg_dict)
        model = PLAPolicy(cfg).cuda()
        # tolerate state-dict tag mismatch (e.g. proximity_encoder absent)
        msg = model.load_state_dict(ckpt["model"], strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            log.warning(
                "load_state_dict mismatch — missing=%d unexpected=%d (continuing)",
                len(msg.missing_keys), len(msg.unexpected_keys),
            )
        model.eval()
        self.model = model
        log.info(
            "[pla.eval] loaded %s (use_proximity=%s, chunk=%d)",
            self.checkpoint_path, cfg.use_proximity, cfg.chunk_size,
        )

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        return obs

    def inference_model(self, obs):
        if self.model is None:
            self.prepare_model()

        # Refill chunk if drained
        if self._chunk is None or self._chunk_idx >= len(self._chunk):
            qpos_t, image_t, prox_t = self._tensors_from_obs(obs)
            with torch.no_grad():
                a_hat = self.model(qpos_t, image_t, prox_t)  # (1, k, 7)
            self._chunk = a_hat.squeeze(0).cpu().numpy()
            self._chunk_idx = 0

        action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action

    def model_output_to_action(self, model_output):
        # model_output is (action_dim,). With action_dim>=8 we predict the
        # gripper directly (normalized to [0,1], rescale to {0,255} space).
        # With action_dim==7 we fall back to the heuristic schedule.
        arm = np.asarray(model_output[:7], dtype=np.float32)
        if len(model_output) >= 8:
            g_norm = float(np.clip(model_output[7], 0.0, 1.0))
            # Snap to binary command consistent with how datagen records
            # gripper actions (raw value is exactly 0.0 or 255.0).
            gripper = 0.0 if g_norm < self.gripper_threshold else 255.0
        else:
            gripper = self._gripper_value()
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _gripper_value(self) -> float:
        if self.gripper_schedule == "open":
            return 0.0
        if self.gripper_schedule == "qpos":
            # Caller will overwrite with current qpos in env; keep 0 as no-op.
            return 0.0
        if self.gripper_schedule == "binary_t":
            return 0.0 if self._step < self.gripper_close_step else 255.0
        raise ValueError(f"unknown gripper_schedule={self.gripper_schedule!r}")

    def _tensors_from_obs(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # qpos -> (1, 7)
        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        qpos = torch.from_numpy(arm).cuda().unsqueeze(0)

        # images -> (1, num_cam, 3, H, W) in [0, 1]
        cams = []
        for cam_name in ("exo_camera_1", "wrist_camera"):
            img = obs[cam_name]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
            cams.append(img.astype(np.float32) / 255.0)
        image = torch.from_numpy(np.stack(cams, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).cuda()

        prox = None
        if self.use_proximity:
            stack = np.zeros((29, 8, 8), dtype=np.float32)
            for i, sname in enumerate(SENSOR_NAMES):
                arr = np.asarray(obs[sname], dtype=np.float32)
                if arr.ndim == 3:  # (n_substeps, 8, 8)
                    arr = arr.mean(axis=0)
                stack[i] = arr
            stack = np.clip(stack / float(self.depth_max_m), 0.0, 1.0)
            prox = torch.from_numpy(stack).unsqueeze(0).cuda()

        return qpos, image, prox

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "pla" if self.use_proximity else "vlm_only_act"
        info["policy_checkpoint"] = self.checkpoint_path
        info["use_proximity"] = self.use_proximity
        info["gripper_schedule"] = self.gripper_schedule
        return info
