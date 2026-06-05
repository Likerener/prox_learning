"""Audit proximity sensor activation in existing trajectory HDF5 files.

This is a read-only analysis script. It scans trajectory batches, reduces each
sensor frame to a minimum depth, and writes a CSV plus diagnostic plots showing
which sensors activate, in which phases, and whether any look self-sensing.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only on missing env deps.
    missing = exc.name or str(exc)
    print(
        f"Missing required dependency: {missing}. Install the repo dependencies first, "
        "for example `pip install -e .` or `pip install h5py numpy pandas matplotlib`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

try:
    import seaborn as sns
except ImportError:  # seaborn is optional.
    sns = None


THRESHOLDS = (0.50, 0.20, 0.10, 0.05)
CANONICAL_PHASES = ("approach", "pregrasp", "grasp_lift", "transit", "place")
NUMERIC_PHASE_MAP = {
    0: "approach",
    1: "pregrasp",
    2: "grasp_lift",
    3: "transit",
    4: "place",
}
POLICY_PHASE_CANONICAL_MAP = {
    "gripper_open": "approach",
    "open": "approach",
    "pregrasp": "pregrasp",
    "grasp": "grasp_lift",
    "gripper_close": "grasp_lift",
    "close": "grasp_lift",
    "lift": "grasp_lift",
    "preplace": "transit",
    "place": "place",
    "retreat": "post_place",
    "go_home": "post_place",
    "home": "post_place",
    "unknown": "unknown",
}
LIFT_M = 0.05
PREGRASP_M = 0.10
PLACE_M = 0.10
GRIPPER_CLOSED_THRESH = 0.10
DEFAULT_GLOBS = (
    "assets/datagen/**/trajectories_batch_*.h5",
    "analysis_output/**/datagen_raw/**/trajectories_batch_*.h5",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        batch_n = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.mean += delta * batch_n / total_n
            self.m2 += batch_m2 + delta * delta * self.n * batch_n / total_n
            self.n = total_n
        self.min_value = min(self.min_value, float(values.min()))

    @property
    def std(self) -> float:
        return float(math.sqrt(self.m2 / self.n)) if self.n else float("nan")


@dataclass
class SensorAgg:
    n_frames: int = 0
    n_valid_frames: int = 0
    raw_threshold_counts: dict[float, int] = field(
        default_factory=lambda: {thr: 0 for thr in THRESHOLDS}
    )
    stats: RunningStats = field(default_factory=RunningStats)
    threshold_counts: dict[float, int] = field(
        default_factory=lambda: {thr: 0 for thr in THRESHOLDS}
    )
    phase_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    phase_valid_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    phase_activation_020: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    success_counts: dict[bool, int] = field(default_factory=lambda: defaultdict(int))
    success_activation_020: dict[bool, int] = field(default_factory=lambda: defaultdict(int))
    within_traj_std: list[float] = field(default_factory=list)
    frame_abs_diff: list[float] = field(default_factory=list)
    samples: list[np.ndarray] = field(default_factory=list)
    sample_n: int = 0


@dataclass(frozen=True)
class DepthFilter:
    min_depth_m: float
    min_inclusive: bool
    max_depth_m: float = 4.0

    @property
    def label(self) -> str:
        left = "[" if self.min_inclusive else "("
        return f"{left}{self.min_depth_m:.2f}, {self.max_depth_m:.2f}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_glob",
        action="append",
        default=None,
        help=(
            "Glob matching trajectory H5 files. Can be passed more than once. "
            "If omitted, default datagen globs are searched."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis_output/sensor_activation_audit"),
        help="Directory for CSV, plots, and summary.json.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap for quick validation runs.",
    )
    parser.add_argument(
        "--max_quantile_samples_per_sensor",
        type=int,
        default=1_000_000,
        help="Bound retained samples per sensor for percentile estimates.",
    )
    parser.add_argument(
        "--phase_source",
        choices=("auto", "policy", "heuristic"),
        default="auto",
        help=(
            "Phase labels to use. auto decodes obs/extra/policy_phase first, then "
            "falls back to kinematic/grasp-state inference if the saved phase is "
            "constant/uninformative."
        ),
    )
    parser.add_argument(
        "--strict_near_zero_filter",
        action="store_true",
        help="Use the stricter valid-depth rule 0.05 <= depth <= 4.0m instead of 0 < depth <= 4.0m.",
    )
    parser.add_argument(
        "--valid_depth_max_m",
        type=float,
        default=4.0,
        help="Maximum depth, in meters, retained as a valid proximity pixel.",
    )
    parser.add_argument(
        "--old_audit_dir",
        type=Path,
        default=None,
        help="Optional previous audit directory to compare against in report.md and summary.json.",
    )
    return parser.parse_args()


def make_depth_filter(args: argparse.Namespace) -> DepthFilter:
    if args.strict_near_zero_filter:
        return DepthFilter(min_depth_m=0.05, min_inclusive=True, max_depth_m=args.valid_depth_max_m)
    return DepthFilter(min_depth_m=0.0, min_inclusive=False, max_depth_m=args.valid_depth_max_m)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def expand_globs(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidates = [pattern]
        p_pattern = Path(pattern)
        if not p_pattern.is_absolute():
            repo_pattern = str(REPO_ROOT / pattern)
            if repo_pattern not in candidates:
                candidates.append(repo_pattern)
        for candidate in candidates:
            for match in glob.glob(candidate, recursive=True):
                p = Path(match)
                if p.is_file():
                    paths.add(p.resolve())
    return sorted(paths, key=lambda p: natural_key(str(p)))


def list_traj_keys(root: h5py.File) -> list[str]:
    keys = [
        key
        for key in root.keys()
        if key.startswith("traj_") and isinstance(root[key], h5py.Group)
    ]
    return sorted(keys, key=natural_key)


def sensor_link(sensor_name: str) -> str:
    match = re.search(r"(link\d+)", sensor_name)
    return match.group(1) if match else "unknown"


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_scalar(value.item())
        if value.dtype.kind in "uifb" and value.size > 1:
            return int(np.nanargmax(value))
        if value.dtype.kind in "SU":
            raw = value.tobytes()
            return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        return value.tolist()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_json_rows(arr: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in arr:
        try:
            if isinstance(row, (bytes, np.bytes_)):
                raw = bytes(row)
            elif np.asarray(row).dtype.kind in "SU":
                raw = np.asarray(row).tobytes()
            else:
                raw = bytes(np.asarray(row).astype(np.uint8))
            text = raw.decode("utf-8", errors="ignore").split("\x00", 1)[0].strip()
            parsed = json.loads(text) if text else {}
            rows.append(parsed if isinstance(parsed, dict) else {})
        except Exception:
            rows.append({})
    return rows


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def tcp_world(tcp_robot: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    return base_pose[:3] + quat_to_rotation(base_pose[3:7]) @ tcp_robot[:3]


def infer_heuristic_phases(traj: h5py.Group, n_frames: int, warned: set[str]) -> np.ndarray | None:
    required = (
        "obs/extra/tcp_pose",
        "obs/extra/robot_base_pose",
        "obs/extra/obj_start",
        "obs/extra/grasp_state_pickup_obj",
        "obs/agent/qpos",
    )
    missing = [key for key in required if key not in traj]
    if missing:
        warned.add(f"heuristic_phase_missing_keys:{','.join(missing)}")
        return None

    try:
        tcp_pose = traj["obs/extra/tcp_pose"][:n_frames]
        base_pose = traj["obs/extra/robot_base_pose"][:n_frames]
        obj_xyz = traj["obs/extra/obj_start"][0, :3].astype(np.float64)
        tcp_xyz = np.empty((n_frames, 3), dtype=np.float64)
        for t in range(n_frames):
            tcp_xyz[t] = tcp_world(tcp_pose[t], base_pose[t])

        d_xy = np.linalg.norm(tcp_xyz[:, :2] - obj_xyz[None, :2], axis=1)
        grasp_rows = decode_json_rows(traj["obs/extra/grasp_state_pickup_obj"][:n_frames])
        held = np.array(
            [bool(row.get("gripper", {}).get("held", False)) for row in grasp_rows],
            dtype=bool,
        )
        qpos_rows = decode_json_rows(traj["obs/agent/qpos"][:n_frames])
        grip = np.array(
            [
                float(np.mean(row.get("gripper", [0.0, 0.0])[:2])) if row else 0.0
                for row in qpos_rows
            ],
            dtype=np.float64,
        )
        grip_closed = grip > GRIPPER_CLOSED_THRESH
    except Exception as exc:
        warned.add(f"heuristic_phase_failed:{exc}")
        return None

    first_held = int(np.argmax(held)) if held.any() else -1
    if first_held >= 0:
        lift = tcp_xyz[:, 2] - tcp_xyz[first_held, 2]
    else:
        lift = np.zeros(n_frames, dtype=np.float64)

    phases = np.empty(n_frames, dtype=object)
    for t in range(n_frames):
        if held[t] and lift[t] > LIFT_M:
            phases[t] = "transit"
        elif held[t] or grip_closed[t]:
            phases[t] = "grasp_lift"
        elif d_xy[t] < PREGRASP_M:
            phases[t] = "pregrasp"
        else:
            phases[t] = "approach"

    if "obs/extra/obj_end" in traj:
        try:
            obj_end = traj["obs/extra/obj_end"][:]
            target_xy = obj_end[0, :2] if np.linalg.norm(obj_end[0, :3]) > 1e-6 else None
            if target_xy is not None:
                for t in range(n_frames):
                    if phases[t] == "transit" and held[t]:
                        if float(np.linalg.norm(tcp_xyz[t, :2] - target_xy)) < PLACE_M:
                            phases[t] = "place"
        except Exception as exc:
            warned.add(f"heuristic_place_phase_failed:{exc}")
    return phases


def normalize_phase_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text.strip().strip("\x00")).strip("_").lower()


def canonicalize_policy_phase_name(text: str) -> str:
    normalized = normalize_phase_text(text)
    return POLICY_PHASE_CANONICAL_MAP.get(normalized, normalized or "unknown")


def load_policy_phase_lookup(traj: h5py.Group, warned: set[str]) -> dict[int, str]:
    if "obs_scene" not in traj:
        return {}
    try:
        raw = decode_scalar(traj["obs_scene"][()])
        if not isinstance(raw, str) or not raw.strip():
            return {}
        parsed = json.loads(raw)
        phase_map = parsed.get("policy_phases", {}) if isinstance(parsed, dict) else {}
        if not isinstance(phase_map, dict):
            return {}
        lookup: dict[int, str] = {}
        for name, idx in phase_map.items():
            try:
                lookup[int(idx)] = canonicalize_policy_phase_name(str(name))
            except Exception:
                continue
        return lookup
    except Exception as exc:
        warned.add(f"unreadable_obs_scene_policy_phases:{exc}")
        return {}


def normalize_phase(value: Any, phase_lookup: dict[int, str] | None = None) -> str:
    value = decode_scalar(value)
    if isinstance(value, str):
        text = value.strip().strip("\x00")
        if not text:
            return "unknown"
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("phase", "policy_phase", "name", "label"):
                    if key in parsed:
                        return normalize_phase(parsed[key], phase_lookup)
                if parsed:
                    return normalize_phase(next(iter(parsed.values())), phase_lookup)
            return normalize_phase(parsed, phase_lookup)
        except Exception:
            pass
        return canonicalize_policy_phase_name(text)
    if isinstance(value, (int, np.integer)):
        idx = int(value)
        if phase_lookup and idx in phase_lookup:
            return phase_lookup[idx]
        return NUMERIC_PHASE_MAP.get(idx, f"phase_{idx}")
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            idx = int(value)
            if phase_lookup and idx in phase_lookup:
                return phase_lookup[idx]
            return NUMERIC_PHASE_MAP.get(idx, f"phase_{idx}")
        return f"phase_{str(value).replace('.', '_')}"
    return "unknown"


def load_policy_phases(traj: h5py.Group, n_frames: int, warned: set[str]) -> np.ndarray:
    path = "obs/extra/policy_phase"
    if path not in traj:
        warned.add("missing_policy_phase")
        return np.full(n_frames, "unknown", dtype=object)

    ds = traj[path]
    try:
        raw = ds[:]
    except Exception as exc:
        warned.add(f"unreadable_policy_phase:{exc}")
        return np.full(n_frames, "unknown", dtype=object)

    phase_lookup = load_policy_phase_lookup(traj, warned)
    if raw.ndim >= 2 and raw.shape[0] == n_frames and raw.dtype.kind in "uifb":
        labels = [normalize_phase(row, phase_lookup) for row in raw]
    elif raw.shape == ():
        labels = [normalize_phase(raw, phase_lookup)] * n_frames
    else:
        flat = np.asarray(raw)
        if flat.shape[0] != n_frames:
            warned.add(f"policy_phase_length_mismatch:{flat.shape[0]}!={n_frames}")
        usable = min(int(flat.shape[0]), n_frames)
        labels = [normalize_phase(flat[i], phase_lookup) for i in range(usable)]
        labels.extend(["unknown"] * (n_frames - usable))
    return np.asarray(labels, dtype=object)


def phase_labels_are_informative(phases: np.ndarray) -> bool:
    unique = {str(p) for p in np.unique(phases) if str(p) != "unknown"}
    if len(unique) > 1:
        return True
    if len(unique) == 1 and next(iter(unique)) != "approach":
        return True
    return False


def select_phases(
    traj: h5py.Group,
    n_frames: int,
    phase_source: str,
    warned: set[str],
) -> tuple[np.ndarray, str]:
    if phase_source == "heuristic":
        inferred = infer_heuristic_phases(traj, n_frames, warned)
        if inferred is not None:
            return inferred, "heuristic"
        return np.full(n_frames, "unknown", dtype=object), "unknown"

    policy = load_policy_phases(traj, n_frames, warned)
    if phase_source == "policy":
        return policy, "policy"

    if phase_labels_are_informative(policy):
        return policy, "policy"

    inferred = infer_heuristic_phases(traj, n_frames, warned)
    if inferred is not None and phase_labels_are_informative(inferred):
        warned.add("policy_phase_uninformative_used_heuristic")
        return inferred, "heuristic"
    return policy, "policy"


def trajectory_success(traj: h5py.Group, warned: set[str]) -> bool | None:
    if "success" not in traj:
        warned.add("missing_success")
        return None
    try:
        values = traj["success"][:]
        if values.size == 0:
            warned.add("empty_success")
            return None
        return bool(values[-1])
    except Exception as exc:
        warned.add(f"unreadable_success:{exc}")
        return None


def valid_depth_mask(arr: np.ndarray, depth_filter: DepthFilter) -> np.ndarray:
    finite = np.isfinite(arr)
    if depth_filter.min_inclusive:
        lower_ok = arr >= depth_filter.min_depth_m
    else:
        lower_ok = arr > depth_filter.min_depth_m
    return finite & lower_ok & (arr <= depth_filter.max_depth_m)


def reduce_sensor_min_depth(
    ds: h5py.Dataset,
    depth_filter: DepthFilter,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(ds[:], dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim == 1:
        raw_min = np.where(np.isfinite(arr), arr, np.nan)
        valid = valid_depth_mask(arr, depth_filter)
        filtered_min = np.where(valid, arr, np.nan)
        return np.asarray(filtered_min, dtype=np.float64), np.asarray(raw_min, dtype=np.float64)

    axes = tuple(range(1, arr.ndim))
    finite = np.isfinite(arr)
    raw_has_finite = np.any(finite, axis=axes)
    raw_min = np.min(np.where(finite, arr, np.inf), axis=axes)
    raw_min = np.where(raw_has_finite, raw_min, np.nan)

    valid = valid_depth_mask(arr, depth_filter)
    has_valid = np.any(valid, axis=axes)
    filtered_min = np.min(np.where(valid, arr, np.inf), axis=axes)
    filtered_min = np.where(has_valid, filtered_min, np.nan)
    return np.asarray(filtered_min, dtype=np.float64), np.asarray(raw_min, dtype=np.float64)


def add_samples(agg: SensorAgg, values: np.ndarray, cap: int) -> None:
    if cap <= 0:
        return
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if agg.sample_n < cap:
        remaining = cap - agg.sample_n
        chunk = values[:remaining].copy()
        agg.samples.append(chunk)
        agg.sample_n += int(chunk.size)


def quantiles(agg: SensorAgg) -> tuple[float, float, float]:
    if not agg.samples:
        return (float("nan"), float("nan"), float("nan"))
    values = np.concatenate(agg.samples)
    return tuple(float(x) for x in np.percentile(values, [5, 50, 95]))


def update_sensor_agg(
    agg: SensorAgg,
    min_depth: np.ndarray,
    raw_min_depth: np.ndarray,
    phases: np.ndarray,
    success: bool | None,
    sample_cap: int,
) -> None:
    valid = np.isfinite(min_depth)
    values = min_depth[valid]
    agg.n_frames += int(min_depth.size)
    agg.n_valid_frames += int(values.size)
    agg.stats.update(values)
    add_samples(agg, values, sample_cap)
    raw_values = np.asarray(raw_min_depth, dtype=np.float64)
    for thr in THRESHOLDS:
        agg.threshold_counts[thr] += int(np.count_nonzero(values < thr))
        agg.raw_threshold_counts[thr] += int(np.count_nonzero(raw_values < thr))

    usable = min(len(phases), len(min_depth))
    if usable:
        valid_usable = np.isfinite(min_depth[:usable])
        frame_active = valid_usable & (min_depth[:usable] < 0.20)
        for phase in np.unique(phases[:usable]):
            phase_text = str(phase)
            mask = phases[:usable] == phase
            agg.phase_counts[phase_text] += int(np.count_nonzero(mask))
            agg.phase_valid_counts[phase_text] += int(np.count_nonzero(mask & valid_usable))
            agg.phase_activation_020[phase_text] += int(np.count_nonzero(frame_active & mask))

    if success is not None:
        agg.success_counts[success] += int(values.size)
        agg.success_activation_020[success] += int(np.count_nonzero(values < 0.20))

    if values.size:
        agg.within_traj_std.append(float(np.std(values)))
    if min_depth.size > 1:
        valid_pair = valid[:-1] & valid[1:]
        diffs = np.abs(min_depth[1:][valid_pair] - min_depth[:-1][valid_pair])
        if diffs.size:
            agg.frame_abs_diff.append(float(np.median(diffs)))


def analyze_files(
    paths: list[Path],
    sample_cap: int,
    phase_source: str,
    depth_filter: DepthFilter,
) -> tuple[dict[str, SensorAgg], dict[str, Any], dict[str, dict[str, SensorAgg]], dict[str, SensorAgg]]:
    sensors: dict[str, SensorAgg] = defaultdict(SensorAgg)
    sensors_by_house: dict[str, dict[str, SensorAgg]] = defaultdict(lambda: defaultdict(SensorAgg))
    sensors_by_link: dict[str, SensorAgg] = defaultdict(SensorAgg)
    warnings_seen: set[str] = set()
    phase_names: set[str] = set()
    phase_frame_counts: dict[str, int] = defaultdict(int)
    phase_source_counts: dict[str, int] = defaultdict(int)
    n_traj = 0
    n_frames_total = 0
    success_traj = 0
    failure_traj = 0
    unknown_success_traj = 0

    for h5_path in paths:
        print(f"[audit] reading {h5_path}")
        try:
            with h5py.File(h5_path, "r") as root:
                for traj_key in list_traj_keys(root):
                    traj = root[traj_key]
                    if "obs/proximity" not in traj:
                        warnings_seen.add(f"missing_proximity:{h5_path}:{traj_key}")
                        continue
                    prox = traj["obs/proximity"]
                    sensor_names = sorted(prox.keys(), key=natural_key)
                    if not sensor_names:
                        warnings_seen.add(f"empty_proximity:{h5_path}:{traj_key}")
                        continue

                    first_shape = prox[sensor_names[0]].shape
                    if not first_shape:
                        warnings_seen.add(f"scalar_proximity:{h5_path}:{traj_key}")
                        continue
                    n_frames = int(first_shape[0])
                    phases, selected_phase_source = select_phases(
                        traj,
                        n_frames,
                        phase_source,
                        warnings_seen,
                    )
                    phase_source_counts[selected_phase_source] += 1
                    phase_names.update(str(p) for p in np.unique(phases))
                    for phase, count in zip(*np.unique(phases, return_counts=True)):
                        phase_frame_counts[str(phase)] += int(count)
                    success = trajectory_success(traj, warnings_seen)
                    if success is True:
                        success_traj += 1
                    elif success is False:
                        failure_traj += 1
                    else:
                        unknown_success_traj += 1

                    n_traj += 1
                    n_frames_total += n_frames
                    house_id = h5_path.parent.name if h5_path.parent.name.startswith("house_") else "unknown_house"
                    for sensor_name in sensor_names:
                        try:
                            min_depth, raw_min_depth = reduce_sensor_min_depth(prox[sensor_name], depth_filter)
                        except Exception as exc:
                            warnings_seen.add(f"unreadable_sensor:{h5_path}:{traj_key}:{sensor_name}:{exc}")
                            continue
                        if len(min_depth) != n_frames:
                            warnings_seen.add(
                                f"sensor_length_mismatch:{h5_path}:{traj_key}:{sensor_name}:"
                                f"{len(min_depth)}!={n_frames}"
                            )
                        link = sensor_link(sensor_name)
                        update_sensor_agg(sensors[sensor_name], min_depth, raw_min_depth, phases, success, sample_cap)
                        update_sensor_agg(
                            sensors_by_house[house_id][sensor_name],
                            min_depth,
                            raw_min_depth,
                            phases,
                            success,
                            sample_cap,
                        )
                        update_sensor_agg(sensors_by_link[link], min_depth, raw_min_depth, phases, success, sample_cap)
        except OSError as exc:
            warnings_seen.add(f"unreadable_h5:{h5_path}:{exc}")

    meta = {
        "n_trajectories": n_traj,
        "n_frames": n_frames_total,
        "phase_names": sorted(phase_names, key=natural_key),
        "phase_frame_counts": dict(sorted(phase_frame_counts.items(), key=lambda item: natural_key(item[0]))),
        "phase_source_counts": dict(phase_source_counts),
        "warnings": sorted(warnings_seen),
        "success_trajectories": success_traj,
        "failure_trajectories": failure_traj,
        "unknown_success_trajectories": unknown_success_traj,
    }
    return (
        dict(sensors),
        meta,
        {hid: dict(saggs) for hid, saggs in sensors_by_house.items()},
        dict(sensors_by_link),
    )


def rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else float("nan")


def finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def suspicious_reason_list(row: dict[str, Any], phases: list[str]) -> list[str]:
    reasons: list[str] = []
    n_valid = int(row.get("n_valid_frames", 0) or 0)
    valid_ratio = float(row.get("valid_frame_ratio", float("nan")))
    activation_020 = float(row.get("activation_lt_0_20m", float("nan")))
    activation_005 = float(row.get("activation_lt_0_05m", float("nan")))
    med_within = float(row.get("median_within_traj_std_m", float("nan")))
    med_diff = float(row.get("median_frame_to_frame_abs_diff_m", float("nan")))
    p50 = float(row.get("p50_min_depth_m", float("nan")))
    p95 = float(row.get("p95_min_depth_m", float("nan")))

    if n_valid == 0:
        reasons.append("no_valid_frames")
    elif finite_number(valid_ratio) and valid_ratio < 0.50:
        reasons.append("very_low_valid_frame_ratio")

    if n_valid >= 20 and finite_number(activation_020) and activation_020 >= 0.90:
        reasons.append("activation_lt_0_20m_close_to_1")

    if (
        n_valid >= 20
        and finite_number(med_within)
        and finite_number(med_diff)
        and med_within <= 0.01
        and med_diff <= 1e-4
    ):
        reasons.append("extremely_low_frame_to_frame_variation")

    if (
        n_valid >= 20
        and (
            (finite_number(p95) and p95 <= 0.055)
            or (finite_number(p50) and p50 <= 0.050)
            or (finite_number(activation_005) and activation_005 >= 0.50)
        )
    ):
        reasons.append("min_depth_consistently_near_lower_bound")

    active_phase_count = 0
    for phase in phases:
        phase_rate = row.get(f"phase_{phase}_activation_lt_0_20m", float("nan"))
        phase_valid = int(row.get(f"phase_{phase}_n_valid_frames", 0) or 0)
        if phase_valid >= 10 and finite_number(phase_rate) and float(phase_rate) >= 0.20:
            active_phase_count += 1
    row["phase_spread_active_phase_count"] = active_phase_count
    if n_valid >= 50 and active_phase_count >= 4 and finite_number(activation_020) and activation_020 >= 0.20:
        reasons.append("activation_spread_across_many_phases")

    return reasons


def build_summary_frame(sensors: dict[str, SensorAgg], phase_names: list[str], entity_col: str = "sensor_name") -> pd.DataFrame:
    all_phases = list(CANONICAL_PHASES)
    for phase in phase_names:
        if phase not in all_phases:
            all_phases.append(phase)

    rows: list[dict[str, Any]] = []
    for name in sorted(sensors.keys(), key=natural_key):
        agg = sensors[name]
        p05, p50, p95 = quantiles(agg)
        n_frames = agg.n_frames
        n_valid = agg.n_valid_frames
        success_rate = rate(agg.success_activation_020[True], agg.success_counts[True])
        failure_rate = rate(agg.success_activation_020[False], agg.success_counts[False])
        med_within = float(np.median(agg.within_traj_std)) if agg.within_traj_std else float("nan")
        med_diff = float(np.median(agg.frame_abs_diff)) if agg.frame_abs_diff else float("nan")
        row: dict[str, Any] = {
            entity_col: name,
            "link": sensor_link(name),
            "n_frames": n_frames,
            "n_valid_frames": n_valid,
            "valid_frame_ratio": rate(n_valid, n_frames),
            "activation_lt_0_50m": rate(agg.threshold_counts[0.50], n_valid),
            "activation_lt_0_20m": rate(agg.threshold_counts[0.20], n_valid),
            "activation_lt_0_10m": rate(agg.threshold_counts[0.10], n_valid),
            "activation_lt_0_05m": rate(agg.threshold_counts[0.05], n_valid),
            "near_saturation_lt_0_05m": rate(agg.threshold_counts[0.05], n_valid),
            "raw_frame_activation_lt_0_50m": rate(agg.raw_threshold_counts[0.50], n_frames),
            "raw_frame_activation_lt_0_20m": rate(agg.raw_threshold_counts[0.20], n_frames),
            "raw_frame_activation_lt_0_10m": rate(agg.raw_threshold_counts[0.10], n_frames),
            "raw_frame_activation_lt_0_05m": rate(agg.raw_threshold_counts[0.05], n_frames),
            "mean_min_depth_m": agg.stats.mean if n_valid else float("nan"),
            "std_min_depth_m": agg.stats.std,
            "p05_min_depth_m": p05,
            "p50_min_depth_m": p50,
            "p95_min_depth_m": p95,
            "min_depth_m": agg.stats.min_value if n_valid else float("nan"),
            "median_within_traj_std_m": med_within,
            "median_frame_to_frame_abs_diff_m": med_diff,
            "success_activation_lt_0_20m": success_rate,
            "failure_activation_lt_0_20m": failure_rate,
            "success_minus_failure_activation_lt_0_20m": (
                success_rate - failure_rate
                if np.isfinite(success_rate) and np.isfinite(failure_rate)
                else float("nan")
            ),
        }
        for phase in all_phases:
            row[f"phase_{phase}_n_frames"] = agg.phase_counts[phase]
            row[f"phase_{phase}_n_valid_frames"] = agg.phase_valid_counts[phase]
            row[f"phase_{phase}_valid_frame_ratio"] = rate(agg.phase_valid_counts[phase], agg.phase_counts[phase])
            row[f"phase_{phase}_activation_lt_0_20m"] = rate(
                agg.phase_activation_020[phase],
                agg.phase_valid_counts[phase],
            )
        reasons = suspicious_reason_list(row, all_phases)
        row["suspicious_self_sensing"] = bool(reasons)
        row["suspicious_reason"] = ";".join(reasons)
        rows.append(row)

    return pd.DataFrame(rows)


def save_activation_threshold_plot(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "activation_lt_0_50m",
        "activation_lt_0_20m",
        "activation_lt_0_10m",
        "activation_lt_0_05m",
    ]
    plot_df = df[["sensor_name", *cols]].melt(
        id_vars="sensor_name", var_name="threshold", value_name="activation_rate"
    )
    labels = {
        "activation_lt_0_50m": "<0.50m",
        "activation_lt_0_20m": "<0.20m",
        "activation_lt_0_10m": "<0.10m",
        "activation_lt_0_05m": "<0.05m",
    }
    plot_df["threshold"] = plot_df["threshold"].map(labels)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 5.5))
    for threshold, group in plot_df.groupby("threshold", sort=False):
        ax.plot(group["sensor_name"], group["activation_rate"], marker="o", label=threshold)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Activation rate")
    ax.set_xlabel("Sensor")
    ax.set_title("Per-sensor activation rates by threshold")
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Min depth")
    fig.tight_layout()
    fig.savefig(out / "per_sensor_activation_thresholds.png", dpi=160)
    plt.close(fig)


def save_phase_heatmap(df: pd.DataFrame, phase_names: list[str], out: Path) -> None:
    phase_cols = [
        f"phase_{phase}_activation_lt_0_20m"
        for phase in list(CANONICAL_PHASES) + [p for p in phase_names if p not in CANONICAL_PHASES]
        if f"phase_{phase}_activation_lt_0_20m" in df.columns
    ]
    if not phase_cols:
        print("[audit] no phase columns available; skipping phase heatmap")
        return
    matrix = df.set_index("sensor_name")[phase_cols]
    matrix = matrix.rename(columns={col: col.removeprefix("phase_").removesuffix("_activation_lt_0_20m") for col in phase_cols})
    fig, ax = plt.subplots(figsize=(max(6, len(phase_cols) * 1.2), max(7, len(df) * 0.28)))
    if sns is not None:
        sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Activation <0.20m"})
    else:
        im = ax.imshow(matrix.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
        fig.colorbar(im, ax=ax, label="Activation <0.20m")
    ax.set_title("Per-phase sensor activation")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Sensor")
    fig.tight_layout()
    fig.savefig(out / "per_phase_activation_heatmap.png", dpi=160)
    plt.close(fig)


def save_self_sensing_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(df["suspicious_self_sensing"], "#d62728", "#1f77b4")
    ax.scatter(df["activation_lt_0_50m"], df["median_within_traj_std_m"], c=colors, s=55, alpha=0.85)
    for _, row in df.iterrows():
        if bool(row["suspicious_self_sensing"]):
            ax.annotate(row["sensor_name"], (row["activation_lt_0_50m"], row["median_within_traj_std_m"]), fontsize=8)
    ax.axvline(0.8, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Activation rate <0.50m")
    ax.set_ylabel("Median within-trajectory std (m)")
    ax.set_title("Suspicious/self-sensing heuristic")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "self_sensing_scatter.png", dpi=160)
    plt.close(fig)


def save_success_failure_plot(df: pd.DataFrame, out: Path) -> bool:
    col = "success_minus_failure_activation_lt_0_20m"
    values = df[col]
    if not np.isfinite(values).any():
        print("[audit] no failed and successful trajectory comparison available; skipping success_vs_failure_activation.png")
        return False
    plot_df = df.sort_values(col)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 5.5))
    colors = np.where(plot_df[col] >= 0, "#2ca02c", "#d62728")
    ax.bar(plot_df["sensor_name"], plot_df[col], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Success - failure activation rate <0.20m")
    ax.set_xlabel("Sensor")
    ax.set_title("Success vs failure proximity activation")
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "success_vs_failure_activation.png", dpi=160)
    plt.close(fig)
    return True


def ordered_phases(phase_names: list[str]) -> list[str]:
    phases = list(CANONICAL_PHASES)
    for phase in phase_names:
        if phase not in phases:
            phases.append(phase)
    return phases


def build_phase_activation_table(
    df: pd.DataFrame,
    phase_names: list[str],
    entity_col: str = "sensor_name",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for phase in ordered_phases(phase_names):
            rows.append(
                {
                    entity_col: row[entity_col],
                    "link": row.get("link", sensor_link(str(row[entity_col]))),
                    "phase": phase,
                    "n_frames": int(row.get(f"phase_{phase}_n_frames", 0) or 0),
                    "n_valid_frames": int(row.get(f"phase_{phase}_n_valid_frames", 0) or 0),
                    "valid_frame_ratio": row.get(f"phase_{phase}_valid_frame_ratio", float("nan")),
                    "activation_lt_0_20m": row.get(f"phase_{phase}_activation_lt_0_20m", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def weighted_rate(rows: pd.DataFrame, rate_col: str = "activation_lt_0_20m") -> float:
    valid = rows[np.isfinite(rows[rate_col]) & (rows["n_valid_frames"] > 0)]
    if valid.empty:
        return float("nan")
    numer = float((valid[rate_col] * valid["n_valid_frames"]).sum())
    denom = int(valid["n_valid_frames"].sum())
    return numer / denom if denom else float("nan")


def compare_old_audit(old_audit_dir: Path | None, cleaned_df: pd.DataFrame) -> dict[str, Any]:
    if old_audit_dir is None:
        return {"available": False, "reason": "no_old_audit_dir_supplied"}
    old_csv = old_audit_dir / "sensor_activation_summary.csv"
    if not old_csv.exists():
        return {"available": False, "reason": f"missing {old_csv}"}

    old_df = pd.read_csv(old_csv)
    needed = {"sensor_name", "activation_lt_0_20m", "activation_lt_0_05m", "min_depth_m"}
    if not needed.issubset(old_df.columns):
        return {"available": False, "reason": "old summary missing expected columns"}

    merged = old_df[["sensor_name", "activation_lt_0_20m", "activation_lt_0_05m", "min_depth_m"]].merge(
        cleaned_df[
            [
                "sensor_name",
                "activation_lt_0_20m",
                "activation_lt_0_05m",
                "raw_frame_activation_lt_0_20m",
                "raw_frame_activation_lt_0_05m",
                "min_depth_m",
                "n_valid_frames",
                "valid_frame_ratio",
            ]
        ],
        on="sensor_name",
        how="inner",
        suffixes=("_old", "_cleaned"),
    )
    if merged.empty:
        return {"available": False, "reason": "no overlapping sensors"}

    merged["delta_activation_lt_0_20m"] = merged["activation_lt_0_20m_cleaned"] - merged["activation_lt_0_20m_old"]
    merged["delta_activation_lt_0_05m"] = merged["activation_lt_0_05m_cleaned"] - merged["activation_lt_0_05m_old"]
    top_deltas = (
        merged.assign(abs_delta=lambda x: x["delta_activation_lt_0_20m"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(10)[
            [
                "sensor_name",
                "activation_lt_0_20m_old",
                "activation_lt_0_20m_cleaned",
                "delta_activation_lt_0_20m",
                "activation_lt_0_05m_old",
                "activation_lt_0_05m_cleaned",
                "delta_activation_lt_0_05m",
                "valid_frame_ratio",
            ]
        ]
        .to_dict(orient="records")
    )
    old_zero_min = int(np.count_nonzero(np.isfinite(merged["min_depth_m_old"]) & (merged["min_depth_m_old"] <= 0.0)))
    cleaned_zero_min = int(
        np.count_nonzero(np.isfinite(merged["min_depth_m_cleaned"]) & (merged["min_depth_m_cleaned"] <= 0.0))
    )
    max_abs_delta_020 = float(merged["delta_activation_lt_0_20m"].abs().max())
    max_abs_delta_005 = float(merged["delta_activation_lt_0_05m"].abs().max())
    materially_changed = bool(max_abs_delta_020 >= 0.05 or max_abs_delta_005 >= 0.05 or cleaned_zero_min < old_zero_min)
    return {
        "available": True,
        "old_audit_dir": str(old_audit_dir),
        "rows_compared": int(len(merged)),
        "old_zero_or_negative_min_depth_rows": old_zero_min,
        "cleaned_zero_or_negative_min_depth_rows": cleaned_zero_min,
        "max_abs_delta_activation_lt_0_20m": max_abs_delta_020,
        "max_abs_delta_activation_lt_0_05m": max_abs_delta_005,
        "mean_delta_activation_lt_0_20m": float(merged["delta_activation_lt_0_20m"].mean()),
        "materially_changed": materially_changed,
        "top_activation_delta_rows": top_deltas,
    }


def build_audit_findings(
    df: pd.DataFrame,
    per_house_df: pd.DataFrame | None,
    link_df: pd.DataFrame,
    phase_table: pd.DataFrame,
    old_audit_dir: Path | None,
) -> dict[str, Any]:
    focus_links = {"link5", "link6"}
    strongest_sensors = (
        df.sort_values(["activation_lt_0_20m", "n_valid_frames"], ascending=[False, False])
        .head(10)[
            [
                "sensor_name",
                "link",
                "n_valid_frames",
                "valid_frame_ratio",
                "activation_lt_0_20m",
                "phase_pregrasp_activation_lt_0_20m",
                "phase_grasp_lift_activation_lt_0_20m",
                "suspicious_self_sensing",
            ]
        ]
        .to_dict(orient="records")
    )
    strongest_links = (
        link_df.sort_values(["activation_lt_0_20m", "n_valid_frames"], ascending=[False, False])
        .head(10)[
            [
                "link",
                "n_sensors",
                "n_valid_frames",
                "valid_frame_ratio",
                "activation_lt_0_20m",
                "phase_pregrasp_activation_lt_0_20m",
                "phase_grasp_lift_activation_lt_0_20m",
            ]
        ]
        .to_dict(orient="records")
    )

    useful_houses: list[dict[str, Any]] = []
    if per_house_df is not None and not per_house_df.empty:
        house_rows = per_house_df[per_house_df["link"].isin(focus_links)].copy()
        house_rows["pregrasp_or_grasp_lift_signal"] = house_rows[
            ["phase_pregrasp_activation_lt_0_20m", "phase_grasp_lift_activation_lt_0_20m"]
        ].max(axis=1, skipna=True)
        house_rows["pregrasp_or_grasp_lift_valid_frames"] = house_rows[
            ["phase_pregrasp_n_valid_frames", "phase_grasp_lift_n_valid_frames"]
        ].max(axis=1, skipna=True)
        house_rows = house_rows[
            (house_rows["valid_frame_ratio"] >= 0.80)
            & (house_rows["pregrasp_or_grasp_lift_valid_frames"] >= 5)
            & (house_rows["pregrasp_or_grasp_lift_signal"] >= 0.10)
            & (~house_rows["suspicious_self_sensing"])
        ]
        if not house_rows.empty:
            idx = house_rows.groupby("house_id")["pregrasp_or_grasp_lift_signal"].idxmax()
            useful_houses = (
                house_rows.loc[idx]
                .sort_values("pregrasp_or_grasp_lift_signal", ascending=False)
                .head(20)[
                    [
                        "house_id",
                        "sensor_name",
                        "link",
                        "n_valid_frames",
                        "valid_frame_ratio",
                        "activation_lt_0_20m",
                        "phase_pregrasp_activation_lt_0_20m",
                        "phase_grasp_lift_activation_lt_0_20m",
                        "pregrasp_or_grasp_lift_signal",
                    ]
                ]
                .to_dict(orient="records")
            )

    focus_phase_rates: dict[str, float] = {}
    focus_phase_counts: dict[str, int] = {}
    focus_phase_table = phase_table[phase_table["link"].isin(focus_links)]
    for phase in CANONICAL_PHASES:
        phase_rows = focus_phase_table[focus_phase_table["phase"] == phase]
        focus_phase_rates[phase] = weighted_rate(phase_rows)
        focus_phase_counts[phase] = int(phase_rows["n_valid_frames"].sum()) if not phase_rows.empty else 0

    target_phase_rate = weighted_rate(focus_phase_table[focus_phase_table["phase"].isin(["pregrasp", "grasp_lift"])])
    other_phase_rate = weighted_rate(
        focus_phase_table[
            focus_phase_table["phase"].isin(["approach", "transit", "place"])
        ]
    )
    activation_concentrates_in_pregrasp_grasp_lift = bool(
        np.isfinite(target_phase_rate)
        and np.isfinite(other_phase_rate)
        and target_phase_rate > other_phase_rate
    )
    keep_environment = bool(
        (
            focus_phase_counts.get("pregrasp", 0) >= 20
            and finite_number(focus_phase_rates.get("pregrasp"))
            and focus_phase_rates["pregrasp"] >= 0.10
        )
        or (
            focus_phase_counts.get("grasp_lift", 0) >= 20
            and finite_number(focus_phase_rates.get("grasp_lift"))
            and focus_phase_rates["grasp_lift"] >= 0.10
        )
    )

    suspicious_rows = []
    if per_house_df is not None and not per_house_df.empty:
        suspicious_rows = (
            per_house_df[per_house_df["suspicious_self_sensing"]]
            .sort_values(["activation_lt_0_20m", "valid_frame_ratio"], ascending=[False, True])
            .head(20)[
                [
                    "house_id",
                    "sensor_name",
                    "link",
                    "n_frames",
                    "n_valid_frames",
                    "valid_frame_ratio",
                    "activation_lt_0_20m",
                    "activation_lt_0_05m",
                    "median_frame_to_frame_abs_diff_m",
                    "suspicious_reason",
                ]
            ]
            .to_dict(orient="records")
        )

    return {
        "useful_valid_activation_houses": useful_houses,
        "strongest_valid_signal_sensors": strongest_sensors,
        "strongest_valid_signal_links": strongest_links,
        "focus_link_phase_activation_lt_0_20m": focus_phase_rates,
        "focus_link_phase_valid_frame_counts": focus_phase_counts,
        "target_phase_activation_lt_0_20m": target_phase_rate,
        "other_canonical_phase_activation_lt_0_20m": other_phase_rate,
        "activation_concentrates_in_pregrasp_grasp_lift": activation_concentrates_in_pregrasp_grasp_lift,
        "keep_environment_by_decision_criteria": keep_environment,
        "suspicious_house_sensor_rows_preview": suspicious_rows,
        "old_audit_comparison": compare_old_audit(old_audit_dir, df),
    }


def fmt_pct(value: Any) -> str:
    if not finite_number(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def write_markdown_report(
    findings: dict[str, Any],
    out: Path,
    depth_filter: DepthFilter,
    strict_mode: bool,
    suspicious_count: int,
) -> None:
    useful = findings["useful_valid_activation_houses"]
    links = findings["strongest_valid_signal_links"][:5]
    sensors = findings["strongest_valid_signal_sensors"][:8]
    focus_rates = findings["focus_link_phase_activation_lt_0_20m"]
    focus_counts = findings["focus_link_phase_valid_frame_counts"]
    old = findings["old_audit_comparison"]

    lines: list[str] = []
    mode = "strict near-zero" if strict_mode else "cleaned"
    lines.append(f"# Proximity Activation Audit ({mode})")
    lines.append("")
    lines.append(f"Valid depth rule: `{depth_filter.label} m`; activation rates use valid frames only.")
    lines.append("")
    lines.append("## Which houses have useful valid proximity activation?")
    if useful:
        house_text = ", ".join(
            f"{row['house_id']} ({row['sensor_name']}, signal {fmt_pct(row['pregrasp_or_grasp_lift_signal'])})"
            for row in useful[:12]
        )
        lines.append(house_text)
    else:
        lines.append("No house met the useful-signal heuristic for link5/link6 pregrasp or grasp_lift activation.")

    lines.append("")
    lines.append("## Which links/sensors carry the strongest valid signal?")
    if links:
        lines.append(
            "Top links: "
            + ", ".join(
                f"{row['link']} ({fmt_pct(row['activation_lt_0_20m'])}, {int(row['n_valid_frames'])} valid sensor-frames)"
                for row in links
            )
        )
    if sensors:
        lines.append(
            "Top sensors: "
            + ", ".join(
                f"{row['sensor_name']} ({fmt_pct(row['activation_lt_0_20m'])})"
                for row in sensors
            )
        )

    lines.append("")
    lines.append("## Does activation concentrate in pregrasp and grasp_lift?")
    phase_bits = [
        f"{phase}: {fmt_pct(focus_rates.get(phase))} over {focus_counts.get(phase, 0)} valid frames"
        for phase in CANONICAL_PHASES
    ]
    lines.append("For link5/link6, valid-frame activation <0.20m by phase is: " + "; ".join(phase_bits) + ".")
    if findings["activation_concentrates_in_pregrasp_grasp_lift"]:
        lines.append("By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.")
    else:
        lines.append("Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.")

    lines.append("")
    lines.append("## Which rows should be excluded?")
    if suspicious_count:
        preview = findings["suspicious_house_sensor_rows_preview"]
        lines.append(
            f"{suspicious_count} house/sensor rows were flagged. The CSV gives exact reasons; top examples: "
            + ", ".join(
                f"{row['house_id']} {row['sensor_name']} ({row['suspicious_reason']})"
                for row in preview[:8]
            )
        )
    else:
        lines.append("No house/sensor rows were flagged by the suspicious behavior heuristics.")

    lines.append("")
    lines.append("## Did filtering materially change the old audit?")
    if old.get("available"):
        lines.append(
            "Compared with the old audit, "
            f"{old['old_zero_or_negative_min_depth_rows']} rows had zero/negative old min depths versus "
            f"{old['cleaned_zero_or_negative_min_depth_rows']} after filtering. "
            f"Max absolute delta in <0.20m activation was {fmt_pct(old['max_abs_delta_activation_lt_0_20m'])}; "
            f"max delta in <0.05m near-saturation was {fmt_pct(old['max_abs_delta_activation_lt_0_05m'])}."
        )
        lines.append(
            "Material-change flag: "
            + ("yes" if old.get("materially_changed") else "no")
            + ". Treat cleaned numbers as authoritative."
        )
    else:
        lines.append(f"Old audit comparison unavailable: {old.get('reason', 'unknown reason')}.")

    lines.append("")
    lines.append("## Decision")
    if findings["keep_environment_by_decision_criteria"]:
        lines.append("Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.")
    else:
        lines.append("Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.")

    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_summary_json(
    df: pd.DataFrame,
    link_df: pd.DataFrame,
    findings: dict[str, Any],
    paths: list[Path],
    meta: dict[str, Any],
    out: Path,
    depth_filter: DepthFilter,
    strict_mode: bool,
    success_failure_plot_written: bool,
) -> None:
    pregrasp_col = "phase_pregrasp_activation_lt_0_20m"
    summary = {
        "number_of_h5_files": len(paths),
        "number_of_trajectories": int(meta["n_trajectories"]),
        "number_of_frames": int(meta["n_frames"]),
        "number_of_sensors": int(len(df)),
        "depth_filter": {
            "valid_depth_range_m": depth_filter.label,
            "min_depth_m": depth_filter.min_depth_m,
            "min_inclusive": depth_filter.min_inclusive,
            "max_depth_m": depth_filter.max_depth_m,
            "strict_near_zero_filter": bool(strict_mode),
        },
        "thresholds_used_m": list(THRESHOLDS),
        "suspicious_sensors_list": df.loc[df["suspicious_self_sensing"], "sensor_name"].tolist(),
        "top_sensors_by_activation_lt_0_20m": df.sort_values("activation_lt_0_20m", ascending=False)
        .head(10)[["sensor_name", "n_valid_frames", "valid_frame_ratio", "activation_lt_0_20m"]]
        .to_dict(orient="records"),
        "top_links_by_activation_lt_0_20m": link_df.sort_values("activation_lt_0_20m", ascending=False)
        .head(10)[["link", "n_sensors", "n_valid_frames", "valid_frame_ratio", "activation_lt_0_20m"]]
        .to_dict(orient="records"),
        "top_sensors_by_phase_pregrasp_activation_lt_0_20m": (
            df.sort_values(pregrasp_col, ascending=False)
            .head(10)[["sensor_name", f"phase_pregrasp_n_valid_frames", pregrasp_col]]
            .to_dict(orient="records")
            if pregrasp_col in df.columns and np.isfinite(df[pregrasp_col]).any()
            else []
        ),
        "findings": findings,
        "success_failure_comparison_available": bool(
            meta["success_trajectories"] > 0 and meta["failure_trajectories"] > 0
        ),
        "success_vs_failure_plot_written": success_failure_plot_written,
        "success_trajectories": int(meta["success_trajectories"]),
        "failure_trajectories": int(meta["failure_trajectories"]),
        "unknown_success_trajectories": int(meta["unknown_success_trajectories"]),
        "phases_seen": meta["phase_names"],
        "phase_frame_counts": meta["phase_frame_counts"],
        "phase_source_counts": meta["phase_source_counts"],
        "warnings": meta["warnings"],
        "input_h5_files": [str(p) for p in paths],
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    depth_filter = make_depth_filter(args)
    patterns = args.data_glob or list(DEFAULT_GLOBS)
    if not args.out_dir.is_absolute():
        args.out_dir = REPO_ROOT / args.out_dir
    if args.old_audit_dir is not None and not args.old_audit_dir.is_absolute():
        args.old_audit_dir = REPO_ROOT / args.old_audit_dir
    paths = expand_globs(patterns)
    if args.max_files is not None:
        paths = paths[: args.max_files]

    print("[audit] matched H5 files:")
    for p in paths:
        print(f"  {p}")
    if not paths:
        print("[audit] no H5 files matched; nothing to do", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[audit] valid depth range: {depth_filter.label} m")
    sensors, meta, sensors_by_house, sensors_by_link = analyze_files(
        paths,
        args.max_quantile_samples_per_sensor,
        args.phase_source,
        depth_filter,
    )
    if not sensors:
        print("[audit] no proximity sensors found", file=sys.stderr)
        raise SystemExit(1)

    df = build_summary_frame(sensors, meta["phase_names"])
    csv_path = args.out_dir / "sensor_activation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[audit] wrote {csv_path}")

    link_df = build_summary_frame(sensors_by_link, meta["phase_names"], entity_col="link")
    sensor_counts_by_link = df.groupby("link")["sensor_name"].nunique()
    link_df.insert(1, "n_sensors", link_df["link"].map(sensor_counts_by_link).fillna(0).astype(int))
    per_link_csv_path = args.out_dir / "per_link_summary.csv"
    link_df.to_csv(per_link_csv_path, index=False)
    print(f"[audit] wrote {per_link_csv_path}")

    phase_table = build_phase_activation_table(df, meta["phase_names"])
    phase_table_csv_path = args.out_dir / "phase_activation_table.csv"
    phase_table.to_csv(phase_table_csv_path, index=False)
    print(f"[audit] wrote {phase_table_csv_path}")

    link_phase_table = build_phase_activation_table(link_df, meta["phase_names"], entity_col="link")
    link_phase_table_csv_path = args.out_dir / "per_link_phase_activation.csv"
    link_phase_table.to_csv(link_phase_table_csv_path, index=False)
    print(f"[audit] wrote {link_phase_table_csv_path}")

    # Build per-house summary frame
    per_house_rows = []
    for house_id, house_sensors in sorted(sensors_by_house.items(), key=lambda x: natural_key(x[0])):
        house_df = build_summary_frame(house_sensors, meta["phase_names"])
        house_df.insert(0, "house_id", house_id)
        per_house_rows.append(house_df)

    per_house_df = None
    if per_house_rows:
        per_house_df = pd.concat(per_house_rows, ignore_index=True)
        per_house_csv_path = args.out_dir / "per_house_sensor_activation.csv"
        per_house_df.to_csv(per_house_csv_path, index=False)
        print(f"[audit] wrote {per_house_csv_path}")

    suspicious_df = (
        per_house_df[per_house_df["suspicious_self_sensing"]].copy()
        if per_house_df is not None
        else pd.DataFrame()
    )
    suspicious_csv_path = args.out_dir / "suspicious_house_sensor_rows.csv"
    suspicious_df.to_csv(suspicious_csv_path, index=False)
    print(f"[audit] wrote {suspicious_csv_path}")

    save_activation_threshold_plot(df, args.out_dir)
    print(f"[audit] wrote {args.out_dir / 'per_sensor_activation_thresholds.png'}")
    save_phase_heatmap(df, meta["phase_names"], args.out_dir)
    print(f"[audit] wrote {args.out_dir / 'per_phase_activation_heatmap.png'}")
    save_self_sensing_scatter(df, args.out_dir)
    print(f"[audit] wrote {args.out_dir / 'self_sensing_scatter.png'}")
    success_failure_plot_written = save_success_failure_plot(df, args.out_dir)
    if success_failure_plot_written:
        print(f"[audit] wrote {args.out_dir / 'success_vs_failure_activation.png'}")

    findings = build_audit_findings(df, per_house_df, link_df, phase_table, args.old_audit_dir)
    write_markdown_report(
        findings,
        args.out_dir,
        depth_filter,
        args.strict_near_zero_filter,
        suspicious_count=int(len(suspicious_df)),
    )
    print(f"[audit] wrote {args.out_dir / 'report.md'}")

    write_summary_json(
        df,
        link_df,
        findings,
        paths,
        meta,
        args.out_dir,
        depth_filter,
        args.strict_near_zero_filter,
        success_failure_plot_written,
    )
    print(f"[audit] wrote {args.out_dir / 'summary.json'}")
    print(
        f"[audit] complete: {len(paths)} files, {meta['n_trajectories']} trajectories, "
        f"{meta['n_frames']} frames, {len(df)} sensors"
    )
    print(f"[audit] phase source counts: {meta['phase_source_counts']}")
    print(f"[audit] phase frame counts: {meta['phase_frame_counts']}")
    if meta["failure_trajectories"] == 0:
        print("[audit] no failed trajectories found; success/failure comparison is unavailable")
    for warning in meta["warnings"][:20]:
        print(f"[audit] warning: {warning}")
    if len(meta["warnings"]) > 20:
        print(f"[audit] ... {len(meta['warnings']) - 20} more warnings in summary.json")


if __name__ == "__main__":
    main()
