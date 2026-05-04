"""Streaming validator for in-progress data collection.

Sits next to the collector (separate process or background tmux pane)
and watches the output directory. For every new HDF5 shard:

    * structurally validate (schema)
    * range-check ToF (no NaN; in [20, 4000] mm)
    * proximity-informative? (any reading < 200 mm)
    * action range sane (max abs < ACT_ABS_MAX)
    * length within expected band

Maintains running statistics and writes a JSON heartbeat every N
episodes. If the bad-streak exceeds a threshold, writes an
``ABORT`` marker file in the out_dir; the collector polls for this
file at the top of every episode and exits cleanly.

Run::

    # Default: watch the out_dir, abort after 10 consecutive bad shards.
    python -m pla.data.sentinel --data-dir data/raw/near_contact

    # CI / unattended mode: never abort, just log.
    python -m pla.data.sentinel --data-dir data/raw/near_contact --no-abort

    # Stop after the planned target reached.
    python -m pla.data.sentinel --data-dir data/raw/near_contact --target-n 1000

The collector cooperates by checking for ``out_dir/SENTINEL_ABORT``
between episodes — see the ``collector_should_stop()`` helper.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np

from pla.data.schema import validate

# Hard-coded contract values — keep in sync with pla.sim.tof.
TOF_MIN_MM = 20.0
TOF_MAX_MM = 4000.0
ACT_ABS_MAX = 1.0           # joint-delta max sane abs value (rad)
PROX_THRESHOLD_MM = 200.0
LEN_MIN = 30
LEN_MAX = 2000

ABORT_MARKER = "SENTINEL_ABORT"


@dataclass
class EpisodeAudit:
    file: str
    ok: bool
    n_steps: int = 0
    success: bool | None = None
    prox_informative: bool = False
    tof_min_mm: float = float("nan")
    tof_max_mm: float = float("nan")
    act_abs_max: float = float("nan")
    issues: list[str] = field(default_factory=list)


def audit_episode(h5_path: Path, retries: int = 3,
                  retry_sleep_s: float = 0.5) -> EpisodeAudit:
    """Validate one shard. Cheap; safe to call inline.

    Retries on h5py BlockingIOError (file still being written by the
    collector) — the sentinel can race the writer in fast-collection
    scenarios.
    """
    audit = EpisodeAudit(file=str(h5_path), ok=False)
    last_exc: Exception | None = None
    for _ in range(max(retries, 1)):
        try:
            ok, errors = validate(h5_path)
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(retry_sleep_s)
    else:
        audit.issues.append(f"validate raised: {type(last_exc).__name__}: {last_exc}")
        return audit
    if not ok:
        audit.issues.append(f"schema: {errors[:3]}")
        return audit

    try:
        with h5py.File(h5_path, "r") as f:
            for ep in f.keys():
                obs = f[f"{ep}/observations"]
                tof = obs["tof"][:]
                acts = f[f"{ep}/actions"][:]
                audit.n_steps = int(tof.shape[0])
                audit.success = bool(f[ep].attrs.get("success", False))
                audit.tof_min_mm = float(tof.min())
                audit.tof_max_mm = float(tof.max())
                audit.act_abs_max = float(np.abs(acts).max())
                audit.prox_informative = bool(np.any(tof < PROX_THRESHOLD_MM))
                if not np.isfinite(tof).all():
                    audit.issues.append("tof: NaN/Inf")
                if not np.isfinite(acts).all():
                    audit.issues.append("actions: NaN/Inf")
                if tof.min() < TOF_MIN_MM - 0.5 or tof.max() > TOF_MAX_MM + 0.5:
                    audit.issues.append(
                        f"tof out of range [{tof.min():.1f}, {tof.max():.1f}] mm")
                if audit.act_abs_max > ACT_ABS_MAX:
                    audit.issues.append(
                        f"action |max| {audit.act_abs_max:.3f} > {ACT_ABS_MAX}")
                if not (LEN_MIN <= audit.n_steps <= LEN_MAX):
                    audit.issues.append(
                        f"episode length {audit.n_steps} outside [{LEN_MIN}, {LEN_MAX}]")
                # Frozen-frame: are consecutive RGB frames byte-identical?
                rgb = obs["rgb"][:]
                if rgb.shape[0] >= 2:
                    diffs = np.any(rgb[1:] != rgb[:-1], axis=(1, 2, 3))
                    n_frozen = int((~diffs).sum())
                    if n_frozen > rgb.shape[0] // 5:
                        audit.issues.append(
                            f"frozen RGB: {n_frozen}/{rgb.shape[0]-1} consecutive identical")
                # Per-sensor: any sensor stuck at one value across all steps?
                # tof shape [T, N, 8, 8].
                per_sensor_std = tof.std(axis=(0, 2, 3))
                n_dead = int((per_sensor_std < 0.5).sum())
                if n_dead > 0:
                    audit.issues.append(
                        f"{n_dead} sensor(s) std < 0.5 mm across episode (stuck)")
                break  # only one episode per file in our schema
        audit.ok = len(audit.issues) == 0
    except Exception as e:  # noqa: BLE001
        audit.issues.append(f"read failed: {type(e).__name__}: {e}")
    return audit


def write_abort(out_dir: Path, reason: str) -> None:
    (out_dir / ABORT_MARKER).write_text(
        json.dumps({"reason": reason, "ts": time.time()}, indent=2)
    )


def collector_should_stop(out_dir: Path) -> bool:
    """The collector calls this at the top of each episode.

    If ``True``, stop collecting cleanly and return.
    """
    return (out_dir / ABORT_MARKER).exists()


def _summary_stats(audits: list[EpisodeAudit]) -> dict:
    if not audits:
        return {"n": 0}
    n = len(audits)
    successes = [a.success for a in audits if a.success is not None]
    return {
        "n_total": n,
        "n_ok": sum(1 for a in audits if a.ok),
        "n_success": sum(1 for s in successes if s),
        "frac_prox_informative": sum(1 for a in audits if a.prox_informative) / n,
        "mean_n_steps": float(np.mean([a.n_steps for a in audits])),
        "min_tof_mm": float(min(a.tof_min_mm for a in audits)),
        "max_tof_mm": float(max(a.tof_max_mm for a in audits)),
        "mean_act_abs_max": float(np.mean([a.act_abs_max for a in audits])),
        "issue_counts": _count_issue_types(audits),
    }


def _count_issue_types(audits: list[EpisodeAudit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in audits:
        for issue in a.issues:
            key = issue.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming validator for collection.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--target-n", type=int, default=None,
                   help="stop watching once this many shards exist")
    p.add_argument("--bad-streak", type=int, default=10,
                   help="abort if this many consecutive bad shards land")
    p.add_argument("--min-prox-informative", type=float, default=0.30,
                   help="abort if rolling prox-informative fraction drops below this "
                        "(after at least 50 episodes)")
    p.add_argument("--heartbeat-every", type=int, default=10)
    p.add_argument("--poll-secs", type=float, default=1.0)
    p.add_argument("--no-abort", action="store_true",
                   help="never write the ABORT marker; only log")
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    audits: list[EpisodeAudit] = []
    bad_streak: deque[bool] = deque(maxlen=args.bad_streak)

    print(f"[sentinel] watching {args.data_dir} (poll {args.poll_secs}s, "
          f"abort after {args.bad_streak} consecutive bad)")
    last_heartbeat = 0
    while True:
        # Stop if abort marker already exists.
        if (args.data_dir / ABORT_MARKER).exists():
            print(f"[sentinel] {ABORT_MARKER} present; exiting")
            break

        new_files = sorted(p for p in args.data_dir.glob("*.h5") if p not in seen)
        for p in new_files:
            audit = audit_episode(p)
            audits.append(audit)
            seen.add(p)
            bad_streak.append(not audit.ok)
            tag = "OK " if audit.ok else "BAD"
            iss = ("  | " + "; ".join(audit.issues)) if audit.issues else ""
            print(f"[sentinel] {tag} {p.name} "
                  f"T={audit.n_steps} success={audit.success} "
                  f"prox={int(audit.prox_informative)}{iss}")
            # Streak check.
            if (not args.no_abort and len(bad_streak) == args.bad_streak
                    and all(bad_streak)):
                msg = f"{args.bad_streak} consecutive bad shards; aborting"
                print(f"[sentinel] {msg}")
                write_abort(args.data_dir, msg)
                break
            # Late-stage prox-informative check.
            if (len(audits) >= 50
                    and not args.no_abort
                    and (sum(1 for a in audits if a.prox_informative) / len(audits))
                          < args.min_prox_informative):
                msg = (f"prox-informative fraction "
                       f"{sum(1 for a in audits if a.prox_informative)/len(audits):.2f} "
                       f"< {args.min_prox_informative} after {len(audits)} eps; aborting")
                print(f"[sentinel] {msg}")
                write_abort(args.data_dir, msg)
                break

        # Heartbeat.
        if (len(audits) // max(args.heartbeat_every, 1)) > last_heartbeat:
            last_heartbeat = len(audits) // args.heartbeat_every
            stats = _summary_stats(audits)
            print(f"[sentinel] heartbeat {len(audits)} eps: "
                  f"ok={stats['n_ok']} success={stats['n_success']} "
                  f"prox-informative={100*stats['frac_prox_informative']:.0f}%")

        # Termination.
        if args.target_n and len(audits) >= args.target_n:
            print(f"[sentinel] target {args.target_n} reached; exiting")
            break
        if (args.data_dir / ABORT_MARKER).exists():
            break
        time.sleep(args.poll_secs)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "summary": _summary_stats(audits),
            "audits": [asdict(a) for a in audits],
        }, indent=2))
        print(f"[sentinel] report: {args.report}")


if __name__ == "__main__":
    main()
