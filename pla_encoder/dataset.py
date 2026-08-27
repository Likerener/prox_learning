"""HDF5 dataset for the franka_skin pick-and-place trajectories.

Reads HDF5 files produced by `molmo_spaces.data_generation.main` for the
`FrankaSkinPickAndPlace*` configs.  Each file holds N trajectories under
`traj_<i>/`; per-timestep tensors live under `obs/proximity/<sensor>` (29
sensors of shape `(T, n_substeps, 8, 8)` float32 meters), `obs/agent/qpos`
and `actions/joint_pos` (JSON blobs in uint8 datasets — see schema notes
below), and a per-trajectory `obs_scene` JSON+pickle blob carrying the
language `task_description`.

Schema notes (verified against the 20260508 pilot, but generic):
- `obs/agent/qpos[t]`: bytes → JSON `{"arm": [7], "base": [], "gripper": [2]}`.
  We extract the 7-dim arm vector.
- `actions/joint_pos[t]`: bytes → JSON `{"arm": [7], "gripper": [1]}`. We
  return an 8-dim action: 7 arm joints (rad) + 1 gripper command normalized
  to [0, 1] (raw is binary {0.0, 255.0}; we divide by 255.0). action_dim=8
  is the default; set to 7 to drop gripper.
- `obs/proximity/<sensor>`: `(T, n_substeps, 8, 8)` float32 in meters
  (verified: p50≈1.4m, p99≈7.7m, with rare overflow spikes >1000m). We
  mean-pool the substep dim, divide by `depth_max_m` (default 4.0 m), and
  clip to [0, 1]. Note: the TODO.md text "divide by 4000.0" was written
  assuming millimeters; the data is meters, so the divisor is 4.0.
- `obs_scene`: a single bytes blob per trajectory containing JSON (with a
  trailing pickle field). The `task_description` JSON field is the language
  string; the pickled `frozen_config` tail is ignored here.

The data loader yields fixed-length action chunks of `chunk_size` (default
100) starting from a sampled timestep `t`. Pad mask `is_pad` is True for
chunk slots that fall past episode end.  Observations (proximity, qpos)
are taken at `t` (the start of the chunk), matching the standard ACT
training recipe.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import decord  # fast frame-accurate MP4 reader
    decord.bridge.set_bridge("native")
except ImportError:  # pragma: no cover
    decord = None


SENSOR_NAMES: tuple[str, ...] = tuple(
    [f"link2_sensor_{i}" for i in range(7)]
    + [f"link3_sensor_{i}" for i in range(8)]
    + [f"link5_sensor_{i}" for i in range(6)]
    + [f"link6_sensor_{i}" for i in range(8)]
)
assert len(SENSOR_NAMES) == 29

# molmospaces task_info blobs use bare `Infinity` / `NaN` (not valid JSON);
# json.loads tolerates these via parse_constant.
def _json_loads_lenient(raw: bytes | str) -> dict:
    if isinstance(raw, bytes):
        raw = raw.rstrip(b"\x00").decode("utf-8")
    return json.loads(raw, parse_constant=lambda c: float(c))


def _decode_jsonrow(arr: np.ndarray) -> dict:
    """Decode a uint8 row that holds a JSON string padded with nulls."""
    return _json_loads_lenient(bytes(arr))


# `obs_scene` is a single bytes blob with JSON followed by a pickle field
# (frozen_config). We need just the JSON part. The simplest robust extractor
# matches "task_description" via regex (the JSON is small and well-formed
# up to the frozen_config field which we discard).
_TASK_DESC_RE = re.compile(rb'"task_description"\s*:\s*"([^"]*)"')


def _extract_task_description(obs_scene_blob: bytes) -> str:
    m = _TASK_DESC_RE.search(obs_scene_blob)
    if m is None:
        return ""
    return m.group(1).decode("utf-8", errors="replace")


@dataclass
class TrajIndex:
    """Index entry: one trajectory in one HDF5 file."""
    file_path: str
    traj_key: str
    n_timesteps: int


@dataclass
class FrankaSkinDatasetConfig:
    root_dirs: Sequence[Path]
    """One or more directories containing house_*/trajectories_batch_*.h5 files."""

    chunk_size: int = 100
    """Action prediction horizon k (ACT chunk_size)."""

    use_proximity: bool = True
    """If False, the proximity tensor is zeroed out (baseline VLM-only mode)."""

    qpos_dim: int = 7
    """Number of arm joints to read from `obs/agent/qpos`."""

    action_dim: int = 8
    """Action vector size. 7 = arm only; 8 = arm + gripper (default). Gripper is
    the binary command {0.0, 255.0} from actions/joint_pos.gripper[0], rescaled
    to [0, 1] for L1-loss compatibility with the arm joint magnitudes (~rad)."""

    gripper_action_scale: float = 255.0
    """Divide raw `actions/joint_pos.gripper[0]` by this to normalize {0, 255} → {0, 1}."""

    depth_max_m: float = 4.0
    """Normalize raw depth (meters) by this constant so values land in ~[0,1]."""

    return_language: bool = True
    """If True, emit `task_description` string per item (per-traj cached)."""

    return_image: bool = True
    """If True, read RGB frames at timestep `t` from per-episode MP4s."""

    image_camera_names: Sequence[str] = ("exo_camera_1", "wrist_camera")
    """Camera basenames whose MP4s should be read into the `image` field."""

    image_resolution: tuple[int, int] | None = None
    """If set, resize each RGB frame to (H, W) via cv2.INTER_AREA. None = native."""

    sample_full_episodes: bool = False
    """If True, t=0 always; if False, uniform-random t in [0, T)."""

    only_successful: bool = True
    """Filter trajectories whose final-step `success` flag is False."""


class FrankaSkinHDF5Dataset(Dataset):
    """Streaming HDF5 trajectory dataset for the franka_skin pipeline.

    Items are sampled per `__len__ = sum(T_i)`, one per timestep across all
    trajectories. Each `__getitem__` opens the source HDF5 lazily (file
    handles are cached per-worker via `_get_handle`).
    """

    def __init__(self, cfg: FrankaSkinDatasetConfig) -> None:
        self.cfg = cfg
        self._index: list[TrajIndex] = []
        self._cum_lengths: list[int] = []  # exclusive prefix sum of T
        self._handles: dict[str, h5py.File] = {}
        self._lang_cache: dict[tuple[str, str], str] = {}
        self._video_cache: dict[tuple[str, str], object] = {}
        self._build_index()
        if cfg.return_image and decord is None:
            raise ImportError("return_image=True requires `decord` (pip install decord).")

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------
    def _iter_h5_files(self) -> Iterable[Path]:
        for root in self.cfg.root_dirs:
            root = Path(root)
            yield from sorted(root.glob("house_*/trajectories_batch_*.h5"))

    def _build_index(self) -> None:
        running = 0
        for h5_path in self._iter_h5_files():
            try:
                with h5py.File(h5_path, "r") as f:
                    for tk in f.keys():
                        if not tk.startswith("traj_"):
                            continue
                        t = f[tk]
                        n_t = int(t["success"].shape[0])
                        if self.cfg.only_successful and not bool(t["success"][-1]):
                            continue
                        self._index.append(
                            TrajIndex(file_path=str(h5_path), traj_key=tk, n_timesteps=n_t)
                        )
                        running += n_t
                        self._cum_lengths.append(running)
            except (OSError, KeyError):
                # tolerate corrupt or in-progress files
                continue

    def __len__(self) -> int:
        # One sample per timestep — the canonical ACT training schedule.
        return self._cum_lengths[-1] if self._cum_lengths else 0

    # ------------------------------------------------------------------
    # h5 file handle cache (per-worker)
    # ------------------------------------------------------------------
    def _get_handle(self, file_path: str) -> h5py.File:
        h = self._handles.get(file_path)
        if h is None:
            h = h5py.File(file_path, "r", swmr=True)
            self._handles[file_path] = h
        return h

    def __getstate__(self) -> dict:
        # Drop file handles before pickling (DataLoader worker fork).
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_video_cache"] = {}
        return state

    # ------------------------------------------------------------------
    # MP4 frame reader (per-worker decord cache)
    # ------------------------------------------------------------------
    def _episode_index(self, traj_key: str) -> int:
        # `traj_<i>` ↔ `episode_<i:08d>` in the sibling MP4s.
        assert traj_key.startswith("traj_"), traj_key
        return int(traj_key.split("_", 1)[1])

    def _video_path(self, traj: TrajIndex, cam_name: str) -> Path:
        h5_dir = Path(traj.file_path).parent
        ep = self._episode_index(traj.traj_key)
        return h5_dir / f"episode_{ep:08d}_{cam_name}_batch_1_of_1.mp4"

    def _get_video(self, traj: TrajIndex, cam_name: str):
        key = (traj.file_path, traj.traj_key + ":" + cam_name)
        cached = self._video_cache.get(key)
        if cached is not None:
            return cached
        vr = decord.VideoReader(str(self._video_path(traj, cam_name)))
        # Bound cache so we don't OOM on large datasets.
        if len(self._video_cache) > 64:
            self._video_cache.pop(next(iter(self._video_cache)))
        self._video_cache[key] = vr
        return vr

    def _read_image(self, traj: TrajIndex, t: int) -> np.ndarray:
        """Return (num_cam, 3, H, W) float32 in [0, 1]."""
        import cv2
        frames = []
        for cam in self.cfg.image_camera_names:
            vr = self._get_video(traj, cam)
            t_clamped = min(t, len(vr) - 1)
            arr = vr[t_clamped].asnumpy()  # (H, W, 3) uint8
            if self.cfg.image_resolution is not None:
                h_out, w_out = self.cfg.image_resolution
                arr = cv2.resize(arr, (w_out, h_out), interpolation=cv2.INTER_AREA)
            frames.append(arr.astype(np.float32) / 255.0)
        # Resize to common shape if cams differ; assume they don't unless image_resolution is set.
        stacked = np.stack(frames, axis=0)         # (num_cam, H, W, 3)
        return np.transpose(stacked, (0, 3, 1, 2)).copy()  # (num_cam, 3, H, W)

    # ------------------------------------------------------------------
    # core sampling
    # ------------------------------------------------------------------
    def _locate(self, idx: int) -> tuple[TrajIndex, int]:
        """Map a flat global timestep index to (TrajIndex, local_t)."""
        traj_idx = int(np.searchsorted(self._cum_lengths, idx, side="right"))
        traj = self._index[traj_idx]
        prior = self._cum_lengths[traj_idx - 1] if traj_idx > 0 else 0
        local_t = idx - prior
        return traj, local_t

    def __getitem__(self, idx: int) -> dict:
        traj, t = self._locate(idx)
        if self.cfg.sample_full_episodes:
            t = 0
        f = self._get_handle(traj.file_path)
        traj_grp = f[traj.traj_key]
        n_t = traj.n_timesteps

        # ---------- proximity at t ----------
        prox = np.zeros((29, 8, 8), dtype=np.float32)
        if self.cfg.use_proximity:
            for s_idx, s_name in enumerate(SENSOR_NAMES):
                # shape (n_substeps, 8, 8); mean-pool substep
                substep_frame = traj_grp[f"obs/proximity/{s_name}"][t]
                prox[s_idx] = substep_frame.mean(axis=0)
            prox /= float(self.cfg.depth_max_m)
            np.clip(prox, 0.0, 1.0, out=prox)

        # ---------- qpos at t (arm only) ----------
        qpos_blob = traj_grp["obs/agent/qpos"][t]
        qpos = np.zeros(self.cfg.qpos_dim, dtype=np.float32)
        try:
            qpos_d = _decode_jsonrow(qpos_blob)
            arm = qpos_d.get("arm", [])
            qpos[: min(len(arm), self.cfg.qpos_dim)] = arm[: self.cfg.qpos_dim]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # ---------- action chunk [t : t+k] ----------
        k = self.cfg.chunk_size
        action_chunk = np.zeros((k, self.cfg.action_dim), dtype=np.float32)
        is_pad = np.ones(k, dtype=bool)
        n_real = min(k, n_t - t)
        n_arm = min(7, self.cfg.action_dim)
        include_gripper = self.cfg.action_dim >= 8
        gscale = float(self.cfg.gripper_action_scale)
        for j in range(n_real):
            blob = traj_grp["actions/joint_pos"][t + j]
            try:
                d = _decode_jsonrow(blob)
                a = d.get("arm", [])
                if not a:
                    # terminal-step `{}` blob — pad and continue
                    is_pad[j] = True
                    continue
                action_chunk[j, :n_arm] = a[:n_arm]
                if include_gripper:
                    g_raw = d.get("gripper", [0.0])
                    g = float(g_raw[0]) if g_raw else 0.0
                    action_chunk[j, 7] = g / gscale
                is_pad[j] = False
            except (json.JSONDecodeError, UnicodeDecodeError):
                # leave row as zeros; mark pad so loss ignores it
                is_pad[j] = True

        out = {
            "proximity": torch.from_numpy(prox),  # (29, 8, 8) in [0,1]
            "qpos": torch.from_numpy(qpos),        # (qpos_dim,)
            "action": torch.from_numpy(action_chunk),  # (k, action_dim)
            "is_pad": torch.from_numpy(is_pad),    # (k,)
        }

        if self.cfg.return_image:
            img = self._read_image(traj, t)
            out["image"] = torch.from_numpy(img)   # (num_cam, 3, H, W)

        if self.cfg.return_language:
            out["language"] = self._language_for(traj)

        return out

    # ------------------------------------------------------------------
    # language extraction (cached per trajectory)
    # ------------------------------------------------------------------
    def _language_for(self, traj: TrajIndex) -> str:
        key = (traj.file_path, traj.traj_key)
        cached = self._lang_cache.get(key)
        if cached is not None:
            return cached
        f = self._get_handle(traj.file_path)
        scene = f[traj.traj_key]["obs_scene"][()]
        if isinstance(scene, np.ndarray):
            scene = scene.tobytes()
        elif isinstance(scene, str):
            scene = scene.encode()
        text = _extract_task_description(scene)
        self._lang_cache[key] = text
        return text


# ---------------------------------------------------------------------
# minimal smoke test (run as `python -m pla.dataset <root>`)
# ---------------------------------------------------------------------
def _smoke_test(root: str) -> None:
    cfg = FrankaSkinDatasetConfig(root_dirs=[Path(root)])
    ds = FrankaSkinHDF5Dataset(cfg)
    print(f"trajectories indexed: {len(ds._index)}")
    print(f"total timesteps: {len(ds)}")
    if len(ds) == 0:
        return
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} min={v.min().item():.4f} max={v.max().item():.4f}")
        else:
            print(f"  {k}: {repr(v)[:100]}")


if __name__ == "__main__":
    import sys
    _smoke_test(sys.argv[1])
