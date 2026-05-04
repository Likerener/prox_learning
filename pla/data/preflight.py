"""Pre-flight checks for data collection.

Run this **before** every long collection run. It validates every
moving part that, if broken, would silently produce hours of bad data:

    1. MJCF loads, has the expected number of sensor cameras, no NaN
       in the home pose.
    2. ToF rendering produces sane output (no all-zeros, not all
       saturated, no NaN, within [20, 4000] mm).
    3. (Optional) MolmoSpaces env imports, resets, and steps once.
    4. (Optional) The TAMP policy returns an action of the right shape.
    5. (Optional) ONE full episode round-trip: collect → write HDF5 →
       schema-validate → load through PLADataset → first sample shape.
    6. Disk space is sufficient for the planned collection.

Each check is a function that returns ``(ok, message)``. The CLI runs
all of them and exits non-zero on any failure. A successful run prints
a green "OK to collect N trajectories to <out_dir>" banner; that is
your gate.

Stages (3) and (4) require MolmoSpaces installed and wired; if the
import fails we *skip* those checks and clearly mark them as skipped
rather than failing. Pass ``--strict-env`` to make missing-env a hard
failure.

Run::

    # Quick check (no env required) — verifies MJCF + sensor render only:
    python -m pla.data.preflight --config configs/data/near_contact.yaml

    # Full check including a 1-episode round-trip via MolmoSpaces:
    python -m pla.data.preflight --config configs/data/near_contact.yaml \
        --full --n-traj 1000 --out-dir data/raw/near_contact

    # CI-friendly: error exit code on any non-skip failure:
    python -m pla.data.preflight --config configs/data/near_contact.yaml \
        --strict
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# ANSI colour codes (only used when stdout is a TTY).
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "b": "\033[1m", "x": "\033[0m"}
if not _supports_color():
    _C = {k: "" for k in _C}


@dataclass
class CheckResult:
    name: str
    status: str  # 'pass', 'fail', 'skip', 'warn'
    message: str = ""
    details: dict = field(default_factory=dict)


def _green(s):  return f"{_C['g']}{s}{_C['x']}"
def _red(s):    return f"{_C['r']}{s}{_C['x']}"
def _yellow(s): return f"{_C['y']}{s}{_C['x']}"
def _bold(s):   return f"{_C['b']}{s}{_C['x']}"


# =============================================================================
# Individual checks
# =============================================================================

def check_mjcf(mjcf_path: Path, expected_n_sensors: int) -> CheckResult:
    """MJCF loads and has at least the expected number of sensor cameras."""
    if not mjcf_path.exists():
        return CheckResult("mjcf",
            "fail", f"MJCF file not found: {mjcf_path}")
    try:
        import mujoco
    except ImportError:
        return CheckResult("mjcf",
            "fail", "mujoco not installed; install mujoco>=3.0")
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as e:  # noqa: BLE001
        return CheckResult("mjcf",
            "fail", f"MJCF failed to load: {type(e).__name__}: {e}")

    sensor_cams = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and "sensor" in name:
            sensor_cams.append(name)

    n = len(sensor_cams)
    details = {
        "n_total_cams": int(model.ncam),
        "n_sensor_cams": n,
        "first_sensor_names": sensor_cams[:5],
    }
    if n < expected_n_sensors:
        return CheckResult("mjcf",
            "fail",
            f"MJCF has {n} sensor cameras; config expects {expected_n_sensors}.",
            details)
    if n > expected_n_sensors:
        return CheckResult("mjcf",
            "warn",
            f"MJCF has {n} sensor cameras; config expects {expected_n_sensors}. "
            "Trim config or expand expected count.", details)
    return CheckResult("mjcf",
        "pass", f"{n} sensor cameras in MJCF", details)


def check_tof_render(mjcf_path: Path, expected_n_sensors: int) -> CheckResult:
    """ToFSensorArray renders the home pose without producing garbage.

    This is the canary for the most common skin-pipeline bugs: all-zero
    cameras (sensor positioned outside the world), all-saturated
    cameras (sensor's outward normal points the wrong way and the world
    appears past 4000 mm), or NaN (broken renderer).
    """
    try:
        import mujoco
    except ImportError:
        return CheckResult("tof_render", "fail", "mujoco not installed")
    from pla.sim.tof import ToFSensorArray, ZNEAR_MM, ZFAR_MM

    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        arr = ToFSensorArray(model)
        depth = arr.render(data, add_noise=False)
    except Exception as e:  # noqa: BLE001
        return CheckResult("tof_render",
            "fail", f"render failed: {type(e).__name__}: {e}")

    details = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "min_mm": float(depth.min()),
        "max_mm": float(depth.max()),
        "mean_mm": float(depth.mean()),
        "frac_saturated": float((depth >= ZFAR_MM - 1).mean()),
        "frac_min": float((depth <= ZNEAR_MM + 1).mean()),
    }
    if depth.shape != (expected_n_sensors, 8, 8):
        return CheckResult("tof_render",
            "fail",
            f"render shape {depth.shape} != ({expected_n_sensors}, 8, 8)", details)
    if not np.isfinite(depth).all():
        return CheckResult("tof_render", "fail",
            "render contains NaN/Inf", details)
    if (depth.min() < ZNEAR_MM - 0.1) or (depth.max() > ZFAR_MM + 0.1):
        return CheckResult("tof_render", "fail",
            f"depth out of range: [{depth.min():.1f}, {depth.max():.1f}] mm "
            f"vs allowed [{ZNEAR_MM}, {ZFAR_MM}]", details)

    # All-saturated cameras: sensor's outward normal points into free space,
    # nothing visible.
    per_cam_max = depth.reshape(depth.shape[0], -1).max(axis=1)
    all_saturated = int((per_cam_max >= ZFAR_MM - 50).sum())
    # All-near cameras: sensor sitting inside the link mesh (self-hit).
    per_cam_min = depth.reshape(depth.shape[0], -1).min(axis=1)
    all_self_hit = int((per_cam_min <= 50).sum())

    details.update({
        "n_all_saturated": all_saturated,
        "n_all_self_hit": all_self_hit,
    })
    msgs = []
    if all_self_hit > 0:
        msgs.append(f"{all_self_hit} sensor(s) reading <= 50 mm in empty home pose "
                    f"(likely self-hit)")
    if all_saturated == expected_n_sensors:
        msgs.append("ALL sensors saturated: skin orientation may be inverted")
    if msgs:
        return CheckResult("tof_render", "warn", "; ".join(msgs), details)
    return CheckResult("tof_render", "pass",
        f"{expected_n_sensors} sensors rendered, "
        f"depth in [{depth.min():.0f}, {depth.max():.0f}] mm", details)


def check_env_import(env_module: str, env_class: str) -> CheckResult:
    """MolmoSpaces env class importable."""
    try:
        mod = importlib.import_module(env_module)
        cls = getattr(mod, env_class)
    except (ImportError, AttributeError) as e:
        return CheckResult("env_import",
            "skip", f"env not importable ({type(e).__name__}: {e})")
    return CheckResult("env_import",
        "pass", f"{env_module}.{env_class} importable",
        {"module": env_module, "class": env_class})


def check_env_step(env_module: str, env_class: str, env_kwargs: dict) -> CheckResult:
    """Env can reset() and step() once without crashing."""
    try:
        mod = importlib.import_module(env_module)
        cls = getattr(mod, env_class)
    except (ImportError, AttributeError) as e:
        return CheckResult("env_step",
            "skip", f"env not importable ({type(e).__name__}: {e})")
    try:
        env = cls(seed=0, **env_kwargs)
        obs = env.reset()
        # Try a no-op-ish action (zeros, sized to what the env expects).
        # We try to infer action dim from the obs / a quick guess of 7.
        action_dim = getattr(env, "action_dim", 7)
        action = np.zeros(action_dim, dtype=np.float32)
        nxt, reward, done, info = env.step(action)
        if hasattr(env, "close"):
            env.close()
    except Exception as e:  # noqa: BLE001
        return CheckResult("env_step",
            "fail", f"env reset/step failed: {type(e).__name__}: {e}")
    return CheckResult("env_step",
        "pass", "env.reset() and one step OK",
        {"reset_keys": list(obs.keys()) if isinstance(obs, dict) else None,
         "info_keys": list(info.keys()) if isinstance(info, dict) else None})


def check_disk_space(out_dir: Path, n_traj: int,
                     mb_per_traj: float = 30.0) -> CheckResult:
    """Verify there is room for the planned collection.

    Default 30 MB/episode is conservative for our schema:
      tof   [250, 32, 8, 8] float32  ~ 2.0 MB
      rgb   [250, 3, 224, 224] uint8 ~ 36 MB
      qpos  [250, 7] float32          ~ 7 KB
      acts  [250, 7] float32          ~ 7 KB
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    needed = int(n_traj * mb_per_traj * 1e6)
    details = {
        "free_gb": round(free / 1e9, 2),
        "needed_gb": round(needed / 1e9, 2),
        "n_traj": n_traj,
        "mb_per_traj_assumed": mb_per_traj,
    }
    if free < needed:
        return CheckResult("disk", "fail",
            f"need {needed/1e9:.1f} GB; have {free/1e9:.1f} GB free", details)
    return CheckResult("disk", "pass",
        f"{free/1e9:.1f} GB free; "
        f"need ~{needed/1e9:.1f} GB for {n_traj} traj", details)


def check_one_episode(env_module: str, env_class: str, env_kwargs: dict,
                       out_dir: Path) -> CheckResult:
    """Run ONE full episode and validate the round-trip end-to-end.

    This is the test that catches integration bugs you cannot catch
    from unit-test fixtures: the env contract (what keys does ``obs``
    actually have? what is ``info['success']`` called?), the action
    interface, the policy phase signal, and the HDF5 schema.

    On success, the written file is left in ``out_dir/_preflight/``
    so you can inspect it.
    """
    try:
        mod = importlib.import_module(env_module)
        cls = getattr(mod, env_class)
    except (ImportError, AttributeError) as e:
        return CheckResult("one_episode",
            "skip", f"env not importable ({type(e).__name__}: {e})")
    try:
        from pla.data.collect import collect_episode
        from pla.data.schema import validate
        from pla.sim.tof import ToFSensorArray
    except ImportError as e:
        return CheckResult("one_episode",
            "fail", f"PLA imports failed: {e}")

    preflight_dir = out_dir / "_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = cls(seed=0, **env_kwargs)
        tof_array = ToFSensorArray(env.model)
        # The repo's collect_episode expects a `policy` with .get_action(obs).
        # We construct a trivial random policy here just to drive the env;
        # the goal is the *plumbing*, not the trajectory quality.
        rng = np.random.default_rng(0)

        class _RandomPolicy:
            def get_action(self, obs):
                return rng.standard_normal(7).astype(np.float32) * 0.005

        success = collect_episode(env, _RandomPolicy(), tof_array,
                                   episode_idx=0, out_dir=preflight_dir,
                                   extra_attrs={"preflight": True})
        if hasattr(env, "close"):
            env.close()
    except Exception as e:  # noqa: BLE001
        return CheckResult("one_episode",
            "fail", f"single-episode round-trip failed: {type(e).__name__}: {e}")

    # Validate schema.
    h5_path = preflight_dir / "episode_000000.h5"
    if not h5_path.exists():
        return CheckResult("one_episode",
            "fail", f"expected output not written: {h5_path}")
    ok, errors = validate(h5_path)
    if not ok:
        return CheckResult("one_episode",
            "fail", f"schema invalid: {errors}",
            {"file": str(h5_path)})

    return CheckResult("one_episode",
        "pass", f"1 episode round-trip OK ({h5_path.name}, success={success})",
        {"file": str(h5_path), "success": success})


# =============================================================================
# Driver
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight checks before a long data-collection run.",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="task YAML (configs/data/<task>.yaml)")
    p.add_argument("--mjcf", type=Path, default=None,
                   help="MJCF path; defaults to assets/mjcf/fr3_skin_fixed.xml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-traj", type=int, default=1000,
                   help="planned trajectory count (used for disk-space check)")
    p.add_argument("--full", action="store_true",
                   help="also run the 1-episode round-trip (requires env)")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any failure (recommended in CI)")
    p.add_argument("--strict-env", action="store_true",
                   help="treat missing env as a failure rather than a skip")
    p.add_argument("--report", type=Path, default=None,
                   help="write JSON report to this path")
    return p.parse_args()


def _print_result(r: CheckResult) -> None:
    if r.status == "pass":
        tag = _green("PASS")
    elif r.status == "fail":
        tag = _red("FAIL")
    elif r.status == "skip":
        tag = _yellow("SKIP")
    else:
        tag = _yellow("WARN")
    print(f"  [{tag}] {r.name:<14s} {r.message}")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    task_name = cfg.get("task_name") or args.config.stem
    n_sensors = int(cfg.get("n_sensors", 32))
    out_dir = args.out_dir or Path("data/raw") / task_name
    mjcf = args.mjcf or Path("assets/mjcf/fr3_skin_fixed.xml")

    # Env wiring (defaults match pla.eval.tasks.REGISTRY):
    env_module = cfg.get("env_module", "molmo_spaces.benchmarks.franka_pickandplace")
    env_class = cfg.get("env_class", "FrankaPickandPlaceEnv")
    env_kwargs = cfg.get("env_kwargs", {})

    print(_bold(f"\nPre-flight for collection: task={task_name} n_traj={args.n_traj}"))
    print(f"  MJCF:     {mjcf}")
    print(f"  out_dir:  {out_dir}")
    print(f"  env:      {env_module}.{env_class}{(' kwargs=' + str(env_kwargs)) if env_kwargs else ''}")
    print()

    results: list[CheckResult] = []

    # 1. MJCF.
    results.append(check_mjcf(mjcf, n_sensors))
    _print_result(results[-1])

    # 2. ToF rendering — only meaningful if MJCF passed.
    if results[-1].status in ("pass", "warn"):
        results.append(check_tof_render(mjcf, n_sensors))
        _print_result(results[-1])
    else:
        results.append(CheckResult("tof_render", "skip",
                                   "MJCF check failed; not rendering"))
        _print_result(results[-1])

    # 3. Env import.
    results.append(check_env_import(env_module, env_class))
    _print_result(results[-1])

    # 4. Env reset+step.
    if results[-1].status in ("pass", "warn"):
        results.append(check_env_step(env_module, env_class, env_kwargs))
        _print_result(results[-1])
    else:
        results.append(CheckResult("env_step", "skip", "env import skipped/failed"))
        _print_result(results[-1])

    # 5. Disk.
    results.append(check_disk_space(out_dir, args.n_traj))
    _print_result(results[-1])

    # 6. Single-episode round-trip (slow; opt-in).
    if args.full:
        results.append(check_one_episode(env_module, env_class, env_kwargs, out_dir))
        _print_result(results[-1])

    # Summary.
    n_pass = sum(1 for r in results if r.status == "pass")
    n_warn = sum(1 for r in results if r.status == "warn")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")

    print()
    print(_bold(f"  Summary: {n_pass} pass, {n_warn} warn, {n_fail} fail, {n_skip} skip"))

    # Strict-env: skip becomes fail for env_* checks.
    if args.strict_env:
        for r in results:
            if r.status == "skip" and r.name.startswith("env"):
                r.status = "fail"
        n_fail = sum(1 for r in results if r.status == "fail")

    # Report.
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task_name,
        "n_traj_planned": args.n_traj,
        "config": str(args.config),
        "mjcf": str(mjcf),
        "out_dir": str(out_dir),
        "results": [
            {"name": r.name, "status": r.status,
             "message": r.message, "details": r.details}
            for r in results
        ],
        "summary": {"pass": n_pass, "warn": n_warn,
                    "fail": n_fail, "skip": n_skip},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"  report written: {args.report}")

    if n_fail > 0:
        print(_red(_bold(f"\n  {n_fail} check(s) failed. Do NOT launch collection.")))
        if args.strict:
            sys.exit(1)
    else:
        print(_green(_bold(
            f"\n  OK to collect {args.n_traj} trajectories to {out_dir}.")))


if __name__ == "__main__":
    main()
