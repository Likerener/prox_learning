"""Rollout-time eval for ACT + frozen prox-encoder (P+ACT).

Mirrors `submodules/act/eval_act_mug_random.py` (so the env config and
ParallelRolloutRunner plumbing are unchanged) but:
  * Loads `FrozenProxFeatureExtractor` from the encoder checkpoint.
  * Maintains a per-sensor ring buffer of the last W=8 control-step substep
    readings (each step's `(4, 8, 8)` block) → assembles `(1, 29, W*4, 8, 8)`
    each control step, z-scores with the encoder's `prox_mean`/`prox_std`, and
    runs the extractor → `(1, 29, 3)` metres.
  * Passes the resulting positions into the policy via `proximity_positions=`.

The new eval does NOT touch the existing `submodules/act/eval_act_with_prox.py`
(which depended on the now-deleted residual-head module).

Run from the repo root with the conda env that has molmospaces installed:

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:/home/jaydv/code/prox_learning:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python /home/jaydv/code/prox_learning/pact/act_prox/eval_act_with_prox_encoder.py \
        --ckpt_dir <runs/act_prox_mug_v1> \
        --prox_encoder_ckpt /home/jaydv/code/prox_learning/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
        --prox_mapping_json /home/jaydv/code/prox_learning/act_style_data/mug_house1_random_everything/prox_mapping.json \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_prox_mug_v1
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import json
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

_WANDB_RUN = None
_ROLLOUT_INDEX = 0

# Make ACT modules importable.
_ACT_DIR = Path(__file__).resolve().parents[2] / "submodules" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
# Make pact modules importable.
_PACT_PARENT = Path(__file__).resolve().parents[2]
if str(_PACT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACT_PARENT))


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int, n_proximity_sensors: int,
               prox_tokens_per_sensor: int = 1):
    """Hide this script's CLI flags from detr/main.py's nested argparse."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_with_prox_encoder.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug_random",
        "--seed", str(seed),
        "--num_epochs", "1",
        "--n_proximity_sensors", str(n_proximity_sensors),
        "--prox_tokens_per_sensor", str(prox_tokens_per_sensor),
    ]
    try:
        yield
    finally:
        sys.argv = orig


from policy import ACTPolicy                                              # noqa: E402
from utils import set_seed                                                # noqa: E402

from molmo_spaces.configs.policy_configs import BasePolicyConfig          # noqa: E402
from molmo_spaces.configs.task_sampler_configs import (                     # noqa: E402
    PickAndPlaceTaskSamplerConfig,
)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinPickAndPlacePilotConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner    # noqa: E402
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR                  # noqa: E402
from molmo_spaces.policy.base_policy import InferencePolicy                # noqa: E402
from molmo_spaces.tasks.pick_and_place_task_sampler import (                # noqa: E402
    PickAndPlaceTaskSampler,
)
from molmo_spaces.utils.constants.object_constants import (                 # noqa: E402
    PICK_AND_PLACE_OBJECTS,
)


# ----------------------------------------------------------------------
# Inline reproduction of the original `FrankaSkinPickAndPlacePilotMediumConfig`
# that the `mug_house1_random_everything` dataset was collected with (see
# molmospaces commit 15a748b). The upstream class has since been renamed to
# `PACT` and its randomization stripped, so we recreate the dataset-time
# settings here to keep eval distribution-matched. DO NOT MODIFY.
# ----------------------------------------------------------------------
class _DatagenMediumMatchedConfig(FrankaSkinPickAndPlacePilotConfig):
    seed: int | None = 2026
    num_workers: int = 1
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=1,
        house_inds=[1],
        max_allowed_sequential_irrecoverable_failures=10000,
        robot_object_z_offset_random_min=-np.random.uniform(0.0, 1.0),
        robot_object_z_offset_random_max=np.random.uniform(0.0, 1.0),
        robot_placement_rotation_range_rad=0.52,
        randomize_lighting=True,
    )
    output_dir: Path = ASSETS_DIR / "datagen" / "mug_house_1_random_everything"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_and_place_pilot_medium"

from pact.act_prox.prox_features import FrozenProxFeatureExtractor         # noqa: E402


# ----------------------------------------------------------------------
# Inference policy.
# ----------------------------------------------------------------------
class ACTWithProxEncoderInferencePolicy(InferencePolicy):
    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: "ACTProxEncoderPolicyConfig" = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")

        with open(pc.prox_mapping_json, "r") as f:
            mp = json.load(f)
        self.sensor_names: list[str] = list(mp["sensor_names"])
        self.n_sensors: int = int(mp["n_sensors"])

        self._step: int = 0
        self._pending_chunks: list[Tuple[int, np.ndarray]] = []
        self._policy: Optional[ACTPolicy] = None
        self._stats = None
        self._extractor: Optional[FrozenProxFeatureExtractor] = None

        # Ring buffer (per sensor) of last W control steps, each (4, 8, 8).
        # Lazy-built on first inference call so we know the encoder's window.
        self._buffer: Optional[np.ndarray] = None   # (W, N_sensors, 4, 8, 8)
        self._buffer_filled: int = 0
        self._window: Optional[int] = None
        self._prox_mean: Optional[np.ndarray] = None
        self._prox_std: Optional[np.ndarray] = None

        # Masking & phase state (Experiments 1 & 2).
        self._prox_mean_pos: Optional[torch.Tensor] = None      # (1, n_sensors, 3) loaded mean
        self._phase_log: list[dict] = []                        # per-step phase / dist / gripper
        self._phase_first_close_step: Optional[int] = None
        self._phase_first_lift_step: Optional[int] = None
        self._phase_first_transit_step: Optional[int] = None
        self._phase_first_near_target_step: Optional[int] = None
        self._obj_z_start: Optional[float] = None
        self._obj_xy_target: Optional[np.ndarray] = None        # (2,) world xy goal
        self._cached_tcp_obj_dist: Optional[float] = None
        self._cached_obj_z: Optional[float] = None
        self._cached_gripper_closed: bool = False
        self._mask_counts: dict = {p: 0 for p in
                                   ("none", "approach", "pregrasp",
                                    "grasp_lift", "transit", "place", "unknown")}

    def reset(self) -> None:
        global _ROLLOUT_INDEX
        if _WANDB_RUN is not None and self._step > 0:
            _WANDB_RUN.log(
                {"rollout/episode_idx": _ROLLOUT_INDEX,
                 "rollout/episode_length": int(self._step)},
                step=_ROLLOUT_INDEX,
            )
            _ROLLOUT_INDEX += 1
        # Dump per-step phase log for this rollout before clearing.
        if self.pc.phase_log_path and self._phase_log:
            try:
                p = Path(self.pc.phase_log_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    for row in self._phase_log:
                        f.write(json.dumps(row) + "\n")
            except Exception as e:
                print(f"[act-prox-eval] failed to write phase log: {e}")
        self._step = 0
        self._pending_chunks.clear()
        self._buffer = None
        self._buffer_filled = 0
        self._phase_log.clear()
        self._phase_first_close_step = None
        self._phase_first_lift_step = None
        self._phase_first_transit_step = None
        self._phase_first_near_target_step = None
        self._obj_z_start = None
        self._obj_xy_target = None
        for k in self._mask_counts:
            self._mask_counts[k] = 0

    def prepare_model(self, model_name: str | None = None) -> None:
        pc = self.pc
        # Load encoder extractor.
        self._extractor = FrozenProxFeatureExtractor(pc.prox_encoder_ckpt, device=torch.device("cuda"))
        self._window = int(self._extractor.window)
        self._prox_mean = self._extractor.prox_mean.cpu().numpy().astype(np.float32)   # (4, 8, 8)
        self._prox_std  = self._extractor.prox_std.cpu().numpy().astype(np.float32)    # (4, 8, 8)
        print(f"[act-prox-eval] loaded prox encoder (window={self._window}, n_sensors={self.n_sensors})")

        # Build the ACT policy with prox tokens enabled.
        policy_config = {
            "lr": pc.lr,
            "num_queries": pc.chunk_size,
            "kl_weight": pc.kl_weight,
            "hidden_dim": pc.hidden_dim,
            "dim_feedforward": pc.dim_feedforward,
            "lr_backbone": pc.lr_backbone,
            "backbone": pc.backbone,
            "enc_layers": pc.enc_layers,
            "dec_layers": pc.dec_layers,
            "nheads": pc.nheads,
            "camera_names": list(pc.camera_names),
            "state_dim": pc.state_dim,
            "action_dim": pc.action_dim,
            "n_proximity_sensors": self.n_sensors,
            "prox_tokens_per_sensor": int(getattr(pc, "prox_tokens_per_sensor", 1) or 1),
        }
        with _detr_argv(self.pc.ckpt_dir, self.pc.seed, self.n_sensors,
                        prox_tokens_per_sensor=policy_config["prox_tokens_per_sensor"]):
            policy = ACTPolicy(policy_config)
        sd = torch.load(self.ckpt_path, map_location="cuda")
        policy.load_state_dict(sd)
        policy.cuda().eval()
        self._policy = policy

        with open(self.stats_path, "rb") as f:
            self._stats = pickle.load(f)
        print(f"[act-prox-eval] loaded ACT {self.ckpt_path}")

        # Load proximity-mean replacement tensor (used when --mask_proximity mean).
        if pc.mask_proximity == "mean":
            if not pc.prox_mean_path or not Path(pc.prox_mean_path).exists():
                raise RuntimeError(
                    f"--mask_proximity mean requires --prox_mean_path pointing at a .npy "
                    f"of shape ({self.n_sensors}, 3); got '{pc.prox_mean_path}'"
                )
            arr = np.load(pc.prox_mean_path).astype(np.float32)
            if arr.shape != (self.n_sensors, 3):
                raise RuntimeError(
                    f"prox_mean file shape {arr.shape} != ({self.n_sensors}, 3)"
                )
            self._prox_mean_pos = torch.from_numpy(arr).unsqueeze(0).cuda()
            print(f"[act-prox-eval] loaded prox mean from {pc.prox_mean_path}: "
                  f"||mean||={float(np.linalg.norm(arr.reshape(-1))):.3f}")
        print(f"[act-prox-eval] mask_proximity={pc.mask_proximity!r}  "
              f"mask_phase={pc.mask_phase!r}")

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        return obs

    def _push_proximity(self, obs) -> None:
        """Push the current control step's (N_sensors, 4, 8, 8) block onto the ring buffer."""
        block = np.zeros((self.n_sensors, 4, 8, 8), dtype=np.float32)
        for i, sname in enumerate(self.sensor_names):
            arr = np.asarray(obs[sname], dtype=np.float32)
            if arr.shape == (4, 8, 8):
                block[i] = arr
            elif arr.shape == (8, 8):
                # Simulator returned a per-step mean-pool — replicate across substeps.
                block[i] = np.broadcast_to(arr[None], (4, 8, 8)).copy()
            else:
                raise ValueError(f"unexpected proximity shape {arr.shape} for {sname}")
        W = self._window
        if self._buffer is None:
            # Cold-start: fill all W slots with the first block.
            self._buffer = np.broadcast_to(block[None], (W, self.n_sensors, 4, 8, 8)).copy()
            self._buffer_filled = 1
        else:
            # Roll left by one control step and place the new block at the end.
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = block
            self._buffer_filled = min(W, self._buffer_filled + 1)

    def _build_prox_window_tensor(self) -> torch.Tensor:
        """Assemble the (1, N_sensors, W*4, 8, 8) z-scored input for the encoder."""
        # buf shape: (W, N_sensors, 4, 8, 8) -> normalise across substep stats:
        buf = self._buffer.astype(np.float32, copy=False)
        normed = (buf - self._prox_mean[None, None]) / self._prox_std[None, None]
        # Transpose to (N_sensors, W, 4, 8, 8) and fold (W*4) for the encoder.
        normed = np.transpose(normed, (1, 0, 2, 3, 4))                     # (N, W, 4, 8, 8)
        normed = normed.reshape(self.n_sensors, self._window * 4, 8, 8)    # (N, W*4, 8, 8)
        return torch.from_numpy(normed).unsqueeze(0).cuda()                # (1, N, W*4, 8, 8)

    # ------------------------------------------------------------------
    # Phase classifier  (Exp 2)
    # ------------------------------------------------------------------
    # Five phases:
    #   approach   : gripper open, TCP > ~10 cm from object xy
    #   pregrasp   : gripper open, TCP within ~10 cm of object xy
    #   grasp_lift : gripper closing/closed AND object hasn't been lifted past
    #                lift_threshold (5 cm above start z)
    #   transit    : object is lifted (> lift_threshold) and we're not yet
    #                close to the target (xy distance > ~10 cm)
    #   place      : object lifted AND close to target xy
    # Falls back to "unknown" if any required obs key is missing.
    _PHASES = ("approach", "pregrasp", "grasp_lift", "transit", "place")
    PREGRASP_DIST_M  = 0.10
    LIFT_THRESHOLD_M = 0.05
    PLACE_DIST_M     = 0.10
    # Inverted gripper-qpos convention in molmospaces: tiny value (~0.003) = OPEN,
    # large value (~0.76) = CLOSED. Empirical threshold from a successful rollout:
    # held cases all have gripper_qpos > 0.5; open all have < 0.05.
    GRIPPER_CLOSED_THRESH = 0.10

    def _get_obj_world_xyz(self, obs) -> Optional[np.ndarray]:
        """Best-effort extraction of object world xyz from inference obs dict.
        Key is `obj_start_pose` in molmospaces."""
        for k in ("obj_start_pose", "obj_start", "pickup_obj_pose",
                  "pickup_obj_position", "obj_pose"):
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32)
                if v.size >= 3:
                    return v.reshape(-1)[:3]
        return None

    def _get_target_world_xyz(self, obs) -> Optional[np.ndarray]:
        """World xyz of the target placement (obj_end_pose / place_receptacle)."""
        for k in ("obj_end_pose", "place_receptacle_pose",
                  "obj_target_pose", "target_pose"):
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32)
                if v.size >= 3 and float(np.linalg.norm(v[:3])) > 1e-6:
                    return v.reshape(-1)[:3]
        return None

    def _get_tcp_world_xyz(self, obs) -> Optional[np.ndarray]:
        """tcp_pose is in ROBOT frame (Franka base). Transform to world via
        robot_base_pose (world-frame xyz + quat). Returns world xyz."""
        # Locate raw tcp pose (robot frame) first.
        tcp_robot = None
        for k in ("tcp_pose", "ee_pose", "tcp_world", "ee_position"):
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32)
                if v.size >= 3:
                    tcp_robot = v.reshape(-1)
                    break
        if tcp_robot is None:
            extras = obs.get("extra") if isinstance(obs, dict) else None
            if isinstance(extras, dict):
                for k in ("tcp_pose", "ee_pose"):
                    if k in extras:
                        v = np.asarray(extras[k], dtype=np.float32)
                        if v.size >= 3:
                            tcp_robot = v.reshape(-1)
                            break
        if tcp_robot is None:
            return None

        # Find robot_base_pose (world frame).
        base_pose = None
        for k in ("robot_base_pose", "base_pose"):
            if k in obs:
                v = np.asarray(obs[k], dtype=np.float32)
                if v.size >= 7:
                    base_pose = v.reshape(-1)[:7]
                    break
        if base_pose is None:
            # No base pose available — return raw tcp_robot (caller should be
            # tolerant; the calling site will compare to obj_world which is also
            # incorrectly placed in that case).
            return tcp_robot[:3]

        # Transform: world_xyz = base_xyz + R(base_quat) @ tcp_robot_xyz
        base_xyz = base_pose[:3]
        base_q   = base_pose[3:7]   # (qw, qx, qy, qz)
        qw, qx, qy, qz = base_q
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float32)
        return base_xyz + R @ tcp_robot[:3]

    def _get_target_xy(self, obs) -> Optional[np.ndarray]:
        v = self._get_target_world_xyz(obs)
        if v is not None:
            return v[:2]
        return None

    def _gripper_is_closed(self, obs) -> bool:
        try:
            grip = obs["qpos"].get("gripper") or [0.0, 0.0]
            grip = np.asarray(grip[:2], dtype=np.float32)
            # Inverted convention: tiny qpos (~0.003) = OPEN, large (~0.76) = CLOSED.
            return bool(np.mean(grip) > self.GRIPPER_CLOSED_THRESH)
        except Exception:
            return False

    def _classify_phase(self, obs) -> str:
        tcp = self._get_tcp_world_xyz(obs)
        obj = self._get_obj_world_xyz(obs)
        # Cache for logging.
        self._cached_tcp_obj_dist = None
        self._cached_obj_z = None
        self._cached_gripper_closed = self._gripper_is_closed(obs)

        if tcp is None or obj is None:
            return "unknown"

        if self._obj_z_start is None:
            self._obj_z_start = float(obj[2])

        d_xy = float(np.linalg.norm(tcp[:2] - obj[:2]))
        d_xyz = float(np.linalg.norm(tcp - obj))
        self._cached_tcp_obj_dist = d_xyz
        self._cached_obj_z = float(obj[2])

        lift = float(obj[2] - self._obj_z_start)
        closed = self._cached_gripper_closed

        # Cache milestone steps for plotting / failure taxonomy.
        if closed and self._phase_first_close_step is None:
            self._phase_first_close_step = self._step
        if lift > self.LIFT_THRESHOLD_M and self._phase_first_lift_step is None:
            self._phase_first_lift_step = self._step

        # Determine phase.
        if lift > self.LIFT_THRESHOLD_M:
            tgt = self._get_target_xy(obs)
            if tgt is not None and float(np.linalg.norm(obj[:2] - tgt)) < self.PLACE_DIST_M:
                return "place"
            return "transit"
        if closed:
            return "grasp_lift"
        # Gripper still open
        if d_xy < self.PREGRASP_DIST_M:
            return "pregrasp"
        return "approach"

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()

        # First-call diagnostic: dump obs keys so we can confirm tcp_pose / obj_start /
        # robot_base_pose presence (Exp 2 phase classifier depends on them).
        if self._step == 0 and self.pc.phase_log_path:
            try:
                keys = sorted(list(obs.keys()))
                top_summary = {k: (str(type(obs[k]).__name__) + ":" +
                                     (str(getattr(obs[k], 'shape', '?')) if hasattr(obs[k], 'shape')
                                      else str(len(obs[k]) if hasattr(obs[k], '__len__') else '?')))
                               for k in keys[:60]}
                p = Path(self.pc.phase_log_path).with_suffix(".obs_keys.txt")
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w") as f:
                        f.write("# Captured on first inference step (step 0)\n")
                        for k in keys:
                            f.write(f"{k}\n")
                        f.write("\n# Type summary:\n")
                        for k, v in top_summary.items():
                            f.write(f"  {k}: {v}\n")
            except Exception:
                pass

        pc = self.pc
        stats = self._stats

        # qpos: arm + gripper.
        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos = np.concatenate([arm, grip], axis=0).astype(np.float32)
        qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # Image stack.
        cams = []
        for cam in pc.camera_names:
            img = obs[cam]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[:2] != (pc.image_h, pc.image_w):
                img = cv2.resize(img, (pc.image_w, pc.image_h), interpolation=cv2.INTER_AREA)
            cams.append(img.astype(np.float32) / 255.0)
        image = np.stack(cams, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))

        # Optional vision degradation (Exp 4 stress test):
        if self.pc.degrade_vision != "none":
            mode = self.pc.degrade_vision
            if mode == "blackout":
                # Zero out both camera streams entirely.
                image[:] = 0.0
            elif mode == "dim":
                # Reduce brightness to 30%.
                image *= 0.3
            elif mode == "noise":
                # Add Gaussian noise (σ=0.2) to RGB then clip.
                image = np.clip(image + np.random.randn(*image.shape).astype(np.float32) * 0.2, 0.0, 1.0)
            elif mode == "wrist_off":
                # Zero out the wrist camera only (keep exo).
                # Identify which camera index is "wrist_camera".
                cam_idx = [i for i, c in enumerate(pc.camera_names) if "wrist" in c]
                for i in cam_idx:
                    image[i] = 0.0
            elif mode == "exo_off":
                cam_idx = [i for i, c in enumerate(pc.camera_names) if "exo" in c]
                for i in cam_idx:
                    image[i] = 0.0
            else:
                raise ValueError(f"unknown degrade_vision={mode}")

        image_t = torch.from_numpy(image).float().cuda().unsqueeze(0)

        # Proximity ring buffer → encoder → predicted positions.
        self._push_proximity(obs)
        prox_window = self._build_prox_window_tensor()                     # (1, N, W*4, 8, 8)
        prox_pos = self._extractor(prox_window)                            # (1, N, 3)
        prox_pos_raw = prox_pos  # keep raw for logging

        # ---- Phase classification (used by Exp 2 and per-step logging) ----
        phase = self._classify_phase(obs)
        self._mask_counts[phase] = self._mask_counts.get(phase, 0) + 1

        # ---- Apply masking ----
        mask_mode = self.pc.mask_proximity        # "none", "zero", "mean", "noise", "shuffle"
        mask_phase = self.pc.mask_phase           # "none" or one of the phases
        apply_mask_now = (mask_mode != "none" and
                          (mask_phase == "none" or mask_phase == phase))
        if apply_mask_now:
            if mask_mode == "zero":
                prox_pos = torch.zeros_like(prox_pos)
            elif mask_mode == "mean":
                if self._prox_mean_pos is None:
                    raise RuntimeError("mask=mean but prox_mean_pos was not loaded")
                prox_pos = self._prox_mean_pos.expand_as(prox_pos).clone()
            elif mask_mode == "noise":
                # Gaussian noise with similar magnitude to typical prox_pos.
                # Use the magnitude observed in the current sample to keep noise plausible.
                with torch.no_grad():
                    std = prox_pos.std() if prox_pos.numel() > 1 else torch.tensor(0.3, device=prox_pos.device)
                    if std.item() < 1e-3:
                        std = torch.tensor(0.3, device=prox_pos.device)
                prox_pos = torch.randn_like(prox_pos) * std
            elif mask_mode == "shuffle":
                # Shuffle sensor identities — preserves marginal distribution but
                # destroys per-sensor semantics.
                perm = torch.randperm(prox_pos.shape[1], device=prox_pos.device)
                prox_pos = prox_pos.index_select(1, perm)
            else:
                raise ValueError(f"unknown mask_proximity={mask_mode}")

        # Diagnostics: log per-step prox stats periodically.
        if _WANDB_RUN is not None and self._step % 25 == 0:
            _WANDB_RUN.log(
                {
                    "rollout/prox_pos_x_mean": float(prox_pos[..., 0].mean().item()),
                    "rollout/prox_pos_y_mean": float(prox_pos[..., 1].mean().item()),
                    "rollout/prox_pos_z_mean": float(prox_pos[..., 2].mean().item()),
                    "rollout/prox_buffer_filled": int(self._buffer_filled),
                    "rollout/step_in_episode": int(self._step),
                    "rollout/phase": phase,
                    "rollout/mask_applied": int(apply_mask_now),
                },
                step=_ROLLOUT_INDEX,
            )

        # Per-step phase log (append-only, immediately flushed).
        if self.pc.phase_log_path:
            row = {
                "rollout": int(_ROLLOUT_INDEX),
                "step": int(self._step),
                "phase": phase,
                "mask_applied": bool(apply_mask_now),
                "prox_norm": float(prox_pos_raw.norm(dim=-1).mean().item()),
                "tcp_obj_dist": float(self._cached_tcp_obj_dist) if self._cached_tcp_obj_dist is not None else None,
                "obj_z": float(self._cached_obj_z) if self._cached_obj_z is not None else None,
                "gripper_closed": bool(self._cached_gripper_closed),
            }
            try:
                p = Path(self.pc.phase_log_path)
                if not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception:
                pass

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t, proximity_positions=prox_pos)  # (1, chunk, A) normalised

        new_chunk = a_hat.squeeze(0).cpu().numpy()
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]

        if pc.temp_agg_off:
            self._pending_chunks = [(self._step, new_chunk)]
            return new_chunk[0]
        H = new_chunk.shape[0]
        self._pending_chunks.append((self._step, new_chunk))
        self._pending_chunks = [
            (s, c) for (s, c) in self._pending_chunks if self._step - s < H
        ]
        preds, weights = [], []
        for (start, chunk) in self._pending_chunks:
            k = self._step - start
            if 0 <= k < H:
                preds.append(chunk[k])
                weights.append(np.exp(-pc.temp_agg_m * k))
        preds_a = np.stack(preds, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        return (preds_a * w[:, None]).sum(axis=0).astype(np.float32)

    def model_output_to_action(self, model_output):
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action


# ----------------------------------------------------------------------
# Policy + env configs.
# ----------------------------------------------------------------------
class ACTProxEncoderPolicyConfig(BasePolicyConfig):
    policy_cls: type = ACTWithProxEncoderInferencePolicy
    policy_type: str = "learned"

    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
    prox_encoder_ckpt: str = ""
    prox_mapping_json: str = ""

    image_h: int = 240
    image_w: int = 320
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")
    chunk_size: int = 20
    temp_agg_m: float = 0.01
    temp_agg_off: bool = False
    kl_weight: int = 10
    hidden_dim: int = 256
    dim_feedforward: int = 2048
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    state_dim: int = 9
    action_dim: int = 8
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    seed: int = 0
    prox_tokens_per_sensor: int = 1

    # Masking / phase (Experiments 1 & 2). "none" -> pass prox_pos as-is.
    mask_proximity: str = "none"          # one of {none, zero, mean, noise, shuffle}
    mask_phase: str = "none"              # none|approach|pregrasp|grasp_lift|transit|place
    prox_mean_path: str = ""              # .npy file with mean (n_sensors, 3)
    phase_log_path: str = ""              # if set, append per-step phase to JSONL
    degrade_vision: str = "none"          # none|blackout|dim|noise|wrist_off|exo_off


class ACTProxEncoderMugRandomEvalConfig(_DatagenMediumMatchedConfig):
    """ACT+prox-encoder eval against the same env used to collect
    mug_house_1_random_everything. Inherits all task sampler knobs verbatim
    and only attaches a learned policy."""

    policy_config: ACTProxEncoderPolicyConfig = ACTProxEncoderPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory containing policy_best.ckpt + dataset_stats.pkl")
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_encoder_ckpt", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=20)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--prox_tokens_per_sensor", type=int, default=1,
                   help="Must match the value used at training time (default 1).")
    p.add_argument("--mask_proximity", choices=("none", "zero", "mean", "noise", "shuffle"),
                   default="none",
                   help="Experiment 1: how to replace prox_pos when masking. "
                        "'none' = pass through; 'zero' = zeros; 'mean' = training-mean; "
                        "'noise' = Gaussian noise scaled to typical magnitude; "
                        "'shuffle' = permute sensor identities.")
    p.add_argument("--mask_phase",
                   choices=("none", "approach", "pregrasp", "grasp_lift", "transit", "place"),
                   default="none",
                   help="Experiment 2: only apply --mask_proximity during this phase. "
                        "'none' = apply at every step (i.e., the Exp 1 full-mask condition).")
    p.add_argument("--prox_mean_path", type=str, default="",
                   help="Required if --mask_proximity mean: .npy file of shape (n_sensors, 3).")
    p.add_argument("--phase_log_path", type=str, default="",
                   help="If set, append per-step (rollout, step, phase, mask_applied, ...) to this JSONL.")
    p.add_argument("--degrade_vision",
                   choices=("none", "blackout", "dim", "noise", "wrist_off", "exo_off"),
                   default="none",
                   help="Stress test: degrade the image stream at inference. "
                        "'wrist_off'/'exo_off' zero only one camera; 'dim' multiplies by 0.3; "
                        "'noise' adds Gaussian σ=0.2; 'blackout' zeroes both cameras.")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="pact-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="act_prox_mug_v1")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-prox-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()

    eval_cfg = ACTProxEncoderMugRandomEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(eval_cfg.seed) if eval_cfg.seed is not None else 2026)

    pc = eval_cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.prox_encoder_ckpt = str(Path(args.prox_encoder_ckpt).resolve())
    pc.prox_mapping_json = str(Path(args.prox_mapping_json).resolve())
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m
    pc.prox_tokens_per_sensor = int(args.prox_tokens_per_sensor)
    pc.mask_proximity = str(args.mask_proximity)
    pc.mask_phase = str(args.mask_phase)
    pc.prox_mean_path = str(Path(args.prox_mean_path).resolve()) if args.prox_mean_path else ""
    pc.phase_log_path = str(Path(args.phase_log_path).resolve()) if args.phase_log_path else ""
    pc.degrade_vision = str(args.degrade_vision)

    eval_cfg.save_config()
    print(f"[act-prox-eval] writing rollouts to {eval_cfg.output_dir}")
    print(f"[act-prox-eval] env config seed={eval_cfg.seed}  "
          f"samples_per_house={eval_cfg.task_sampler_config.samples_per_house}  "
          f"house_inds={list(eval_cfg.task_sampler_config.house_inds)}")

    global _WANDB_RUN, _ROLLOUT_INDEX
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("--use_wandb but wandb is not installed.")
        run_name = args.wandb_run_name or f"eval_act_prox_mug_{int(time.time())}"
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config={
                "ckpt_dir": pc.ckpt_dir,
                "prox_encoder_ckpt": pc.prox_encoder_ckpt,
                "prox_mapping_json": pc.prox_mapping_json,
                "task_horizon": args.task_horizon,
                "chunk_size": pc.chunk_size,
                "kl_weight": pc.kl_weight,
                "hidden_dim": pc.hidden_dim,
                "dim_feedforward": pc.dim_feedforward,
                "env_config_class": "FrankaSkinPickAndPlacePilotMediumConfig",
                "env_seed": eval_cfg.seed,
                "mask_proximity": pc.mask_proximity,
                "mask_phase": pc.mask_phase,
                "prox_mean_path": pc.prox_mean_path,
            },
            tags=["pact", "act_prox", "house1_mug_random", "eval",
                  f"mask_prox={pc.mask_proximity}", f"mask_phase={pc.mask_phase}"],
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-prox-eval] wandb run: {_WANDB_RUN.url}  (group={args.wandb_group})")

    try:
        runner = ParallelRolloutRunner(eval_cfg)
        success, total = runner.run()
        print(f"[act-prox-eval] success {success}/{total}")
        if _WANDB_RUN is not None:
            rate = (success / total) if total > 0 else 0.0
            _WANDB_RUN.log({"eval/success": int(success), "eval/total": int(total),
                            "eval/success_rate": float(rate)})
            _WANDB_RUN.summary["success"] = int(success)
            _WANDB_RUN.summary["total"] = int(total)
            _WANDB_RUN.summary["success_rate"] = float(rate)
    finally:
        if _WANDB_RUN is not None:
            _WANDB_RUN.finish()
            _WANDB_RUN = None


if __name__ == "__main__":
    main()
