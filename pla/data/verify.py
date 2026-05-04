"""Dataset sanity check (post-collection).

Run this **after** collection finishes and **before** every training start.
Walks every HDF5 shard, validates schema, and computes:

    Required (gates training start):
      * 0 NaN across tof, qpos, actions
      * schema_ok == 100 %
      * proximity-informative trajectories >= 30 % (any reading < 200 mm)

    Reported (informational, but anomalies are flagged):
      * success rate
      * episode-length distribution (percentiles)
      * per-sensor coverage: how many sensors NEVER see anything close
      * frozen-frame count per file
      * action distribution (mean, std, |max|)
      * disk footprint per shard

Targets that fail with ``--strict`` make the script exit non-zero so a CI
pipeline / shell wrapper can short-circuit before training compute is
committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from pla.data.schema import validate

PROX_THRESHOLD_MM = 200.0
ACT_ABS_MAX = 1.0  # joint-delta sane absolute upper bound (rad)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--threshold-mm", type=float, default=PROX_THRESHOLD_MM)
    p.add_argument("--min-frac-prox-informative", type=float, default=0.30)
    p.add_argument("--min-success-rate", type=float, default=0.30,
                   help="warn (not fail) if success rate is below this")
    p.add_argument("--report", type=Path, default=None,
                   help="write a full JSON report to this path")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any required-target failure.")
    return p.parse_args()


def _audit_file(f: Path, threshold_mm: float) -> dict:
    """Per-file audit. Cheap, runs once per shard during full sweep."""
    rec: dict = {
        "file": str(f),
        "size_bytes": int(f.stat().st_size),
        "schema_ok": False,
        "schema_errors": [],
        "episodes": [],
    }
    schema_ok, schema_errors = validate(f)
    rec["schema_ok"] = bool(schema_ok)
    rec["schema_errors"] = schema_errors[:5]
    if not schema_ok:
        return rec
    try:
        with h5py.File(f, "r") as h:
            for key in h.keys():
                grp = h[key]
                if "observations" not in grp or "tof" not in grp["observations"]:
                    continue
                tof = grp["observations"]["tof"][:]
                qpos = grp["observations"]["qpos"][:]
                acts = grp["actions"][:]
                rgb = grp["observations"]["rgb"][:] if "rgb" in grp["observations"] else None

                # Frozen-frame count: consecutive RGB frames byte-identical.
                if rgb is not None and rgb.shape[0] >= 2:
                    diffs = np.any(rgb[1:] != rgb[:-1], axis=tuple(range(1, rgb.ndim)))
                    n_frozen = int((~diffs).sum())
                else:
                    n_frozen = 0

                # Per-sensor std across the episode — dead sensors have ~0 std.
                per_sensor_std = tof.std(axis=(0, 2, 3))   # [N]
                # Per-sensor never-close — sensor never read below threshold.
                per_sensor_min = tof.min(axis=(0, 2, 3))   # [N]

                ep = {
                    "key": key,
                    "T": int(tof.shape[0]),
                    "n_sensors": int(tof.shape[1]),
                    "success": bool(grp.attrs.get("success", False)),
                    "tof_min_mm": float(tof.min()),
                    "tof_max_mm": float(tof.max()),
                    "tof_mean_mm": float(tof.mean()),
                    "act_mean": float(acts.mean()),
                    "act_abs_max": float(np.abs(acts).max()),
                    "act_std": float(acts.std()),
                    "any_nan": bool(
                        np.isnan(tof).any() or np.isnan(qpos).any() or np.isnan(acts).any()
                    ),
                    "any_inf": bool(
                        np.isinf(tof).any() or np.isinf(qpos).any() or np.isinf(acts).any()
                    ),
                    "prox_informative": bool(np.any(tof < threshold_mm)),
                    "n_steps_close": int(np.any(
                        tof.reshape(tof.shape[0], -1) < threshold_mm, axis=1
                    ).sum()),
                    "n_frozen_rgb": n_frozen,
                    "per_sensor_std_min": float(per_sensor_std.min()),
                    "per_sensor_min_min_mm": float(per_sensor_min.min()),
                    "per_sensor_min_max_mm": float(per_sensor_min.max()),
                }
                rec["episodes"].append(ep)
    except Exception as e:  # noqa: BLE001
        rec["schema_errors"].append(f"read failed: {type(e).__name__}: {e}")
    return rec


def verify_dataset(data_dir: Path, threshold_mm: float = PROX_THRESHOLD_MM) -> dict:
    """Walk the data dir; produce a full audit dict."""
    files = sorted(p for p in Path(data_dir).rglob("*.h5"))
    audits = [_audit_file(p, threshold_mm) for p in files]

    eps = [ep for a in audits for ep in a["episodes"]]
    n = len(eps)
    if n == 0:
        return {
            "data_dir": str(data_dir),
            "total": 0, "files": [str(p) for p in files],
            "audits": audits, "errors": [a for a in audits if a["schema_errors"]],
        }

    Ts = np.asarray([e["T"] for e in eps])
    n_sensors_set = {e["n_sensors"] for e in eps}
    success_count = sum(1 for e in eps if e["success"])
    prox_count = sum(1 for e in eps if e["prox_informative"])
    nan_count = sum(1 for e in eps if e["any_nan"])
    inf_count = sum(1 for e in eps if e["any_inf"])
    schema_ok_count = sum(1 for a in audits if a["schema_ok"])
    n_steps_total = int(Ts.sum())
    n_steps_close = sum(e["n_steps_close"] for e in eps)
    act_abs_max_global = max(e["act_abs_max"] for e in eps) if eps else 0.0
    n_with_dead_sensors = sum(1 for e in eps if e["per_sensor_std_min"] < 0.5)
    n_with_frozen_rgb = sum(
        1 for e in eps if e["n_frozen_rgb"] > max(e["T"] // 5, 1)
    )

    # Per-sensor coverage across the whole dataset:
    # for each sensor index, the minimum value across ALL files.
    if eps:
        coverage = np.full(max(e["n_sensors"] for e in eps), np.inf)
        for a in audits:
            for ep in a["episodes"]:
                # We didn't keep the per-sensor min vector; recompute on the fly
                # from per_sensor_min_min_mm doesn't help here. So we re-open the
                # files lazily for the dataset-level coverage check.
                pass
        # Cheap proxy: a sensor is suspicious if its overall closest reading
        # is > 1500 mm; we sample by file rather than re-loading every shard.
        # For accurate per-sensor coverage we re-open each file once below.
        coverage = _per_sensor_global_min(files)
    else:
        coverage = np.array([])

    return {
        "data_dir": str(data_dir),
        "total": n,
        "n_files": len(files),
        "n_sensors_seen": sorted(n_sensors_set),
        "success_count": success_count,
        "success_rate": success_count / n,
        "schema_ok_count": schema_ok_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "prox_informative_count": prox_count,
        "frac_traj_prox_informative": prox_count / n,
        "n_total_steps": n_steps_total,
        "n_close_steps": n_steps_close,
        "frac_steps_close": n_steps_close / max(n_steps_total, 1),
        "T_min": int(Ts.min()),
        "T_max": int(Ts.max()),
        "T_mean": float(Ts.mean()),
        "T_p10": float(np.percentile(Ts, 10)),
        "T_p50": float(np.percentile(Ts, 50)),
        "T_p90": float(np.percentile(Ts, 90)),
        "act_abs_max_global": act_abs_max_global,
        "n_with_dead_sensors": n_with_dead_sensors,
        "n_with_frozen_rgb": n_with_frozen_rgb,
        "per_sensor_global_min_mm": coverage.tolist() if coverage.size else [],
        "n_dead_sensors_dataset": int((coverage > 1500).sum()) if coverage.size else 0,
        "audits": audits,
        "errors": [a for a in audits if a["schema_errors"] or any(
            ep["any_nan"] or ep["any_inf"] for ep in a["episodes"])],
    }


def _per_sensor_global_min(files: list[Path]) -> np.ndarray:
    """Per-sensor global minimum over the full dataset.

    Used for the dead-sensor check (sensor whose closest reading across
    every file is still > 1500 mm probably never sees the scene).
    """
    if not files:
        return np.array([])
    out: np.ndarray | None = None
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                for k in h.keys():
                    if "observations" not in h[k] or "tof" not in h[k]["observations"]:
                        continue
                    tof = h[f"{k}/observations/tof"][:]
                    per = tof.min(axis=(0, 2, 3))
                    if out is None:
                        out = per.astype(np.float64)
                    else:
                        out = np.minimum(out, per)
                    break
        except Exception:  # noqa: BLE001
            continue
    return out if out is not None else np.array([])


def print_report(stats: dict, threshold_mm: float,
                 min_frac: float, min_success: float) -> bool:
    n = stats["total"]
    print("=" * 64)
    print(f"data_dir:                 {stats.get('data_dir')}")
    print(f"Files:                    {stats.get('n_files', 0)}")
    print(f"Episodes processed:       {n}")
    if n == 0:
        return False
    print(f"Schema OK:                {stats['schema_ok_count']}/{stats.get('n_files',0)}")
    print(f"NaN episodes:             {stats['nan_count']}/{n}")
    print(f"Inf episodes:             {stats['inf_count']}/{n}")
    print(f"Successful:               {stats['success_count']}/{n} "
          f"(rate {100*stats['success_rate']:.1f}%)")
    print(f"Proximity-informative:    {stats['prox_informative_count']}/{n} "
          f"({100*stats['frac_traj_prox_informative']:.1f}%)")
    print(f"Frac steps with <{threshold_mm:.0f} mm reading: "
          f"{100*stats['frac_steps_close']:.1f}%")
    print(f"Episode length (T):       "
          f"min={stats['T_min']} p10={stats['T_p10']:.0f} "
          f"p50={stats['T_p50']:.0f} p90={stats['T_p90']:.0f} "
          f"max={stats['T_max']} mean={stats['T_mean']:.0f}")
    print(f"Action |max| global:      {stats['act_abs_max_global']:.3f} "
          f"(sane upper bound {ACT_ABS_MAX})")
    print(f"n_sensors seen in data:   {stats['n_sensors_seen']}")
    print(f"Dead sensors (never <1500mm): "
          f"{stats['n_dead_sensors_dataset']}/{len(stats['per_sensor_global_min_mm'])}")
    print(f"Episodes with stuck sensor (std < 0.5mm): "
          f"{stats['n_with_dead_sensors']}/{n}")
    print(f"Episodes with frozen RGB (>1/5 frames identical): "
          f"{stats['n_with_frozen_rgb']}/{n}")
    if stats["errors"]:
        print(f"\nERRORS ({len(stats['errors'])}):")
        for e in stats["errors"][:20]:
            print(f"  {e['file']}: {e.get('schema_errors', [])}")
    print("=" * 64)

    must_pass = (
        stats["frac_traj_prox_informative"] >= min_frac
        and stats["nan_count"] == 0
        and stats["inf_count"] == 0
        and stats["schema_ok_count"] == stats.get("n_files", 0)
        and stats["act_abs_max_global"] <= ACT_ABS_MAX
    )
    soft_warn = (
        stats["success_rate"] < min_success
        or stats["n_dead_sensors_dataset"] > 0
        or stats["n_with_frozen_rgb"] > 0
        or stats["n_with_dead_sensors"] > n // 20  # >5% of episodes
    )
    print()
    print(f"Required: prox_informative >= {100*min_frac:.0f}%, "
          f"NaN/Inf == 0, schema_ok == 100%, |action| <= {ACT_ABS_MAX}: "
          f"{'PASS' if must_pass else 'FAIL'}")
    if soft_warn:
        print("Warnings:")
        if stats["success_rate"] < min_success:
            print(f"  * success_rate {100*stats['success_rate']:.1f}% < "
                  f"{100*min_success:.0f}% -- task may be too hard or env broken")
        if stats["n_dead_sensors_dataset"] > 0:
            print(f"  * {stats['n_dead_sensors_dataset']} sensor(s) never read "
                  f"below 1500 mm in the entire dataset")
        if stats["n_with_frozen_rgb"] > 0:
            print(f"  * {stats['n_with_frozen_rgb']} episode(s) have frozen-RGB "
                  f"signature -- camera may have stalled")
        if stats["n_with_dead_sensors"] > n // 20:
            print(f"  * {stats['n_with_dead_sensors']}/{n} episodes had a stuck "
                  f"sensor (std < 0.5 mm across the episode)")
    return must_pass


def main() -> None:
    args = parse_args()
    stats = verify_dataset(args.data_dir, threshold_mm=args.threshold_mm)
    ok = print_report(stats, args.threshold_mm,
                      args.min_frac_prox_informative,
                      args.min_success_rate)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, indent=2))
        print(f"report: {args.report}")
    if not ok and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
