"""Build prox_mapping.json for an ACT-style dataset.

The ACT dataset under e.g. `act_style_data/mug_house1_random_everything/` is
the output of a conversion script that collapsed each
`assets/datagen/<dataset>/<config>/<timestamp>/house_*/trajectories_batch_*.h5`
into one `episode_<n>.hdf5`. Proximity readings were not carried across the
conversion, so we need a deterministic way to find the source h5 for each
ACT episode at training time. Filename order is fragile (the conversion
script's scheduling can interleave), so we match by `qpos[0]` instead.

Output schema: `<act_dataset_dir>/prox_mapping.json`

    {
      "act_dataset_dir": "...",
      "source_glob":     "...",
      "sensor_names":    ["link2_sensor_0", ..., "link6_sensor_7"],
      "n_sensors":       29,
      "qpos_atol":       1e-6,
      "episodes": {
        "0": {"source_h5": "<absolute path>", "traj_key": "traj_0"},
        "1": {"source_h5": "...",             "traj_key": "traj_0"},
        ...
      }
    }

Usage:

    /opt/conda/envs/mlspaces/bin/python -m pact.act_prox.build_mapping \
        --act_dataset_dir act_style_data/mug_house1_random_everything
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


SENSOR_LINKS = ("link2", "link3", "link5", "link6")


# Timesteps used to build the qpos signature. t=0 is identical across all
# trajectories (deterministic init); t>=5 has diverged enough that any two
# random_everything trajectories disagree in the 6th+ decimal. Five timesteps
# = 45 floats per signature — robust beyond any plausible numerical coincidence.
QPOS_SIGNATURE_TIMESTEPS = (5, 10, 15, 20, 25)


def _decode_qpos(src: h5py.Group, traj_key: str, t: int) -> np.ndarray:
    """Decode `obs/agent/qpos[t]` from a single source-h5 traj as concat(arm, gripper).

    Source qpos is JSON bytes per-timestep with shape {"arm":[7], "base":[],
    "gripper":[2]}. ACT-side qpos is `concat(arm[7], gripper[2]) -> (9,)`,
    which we confirmed by checking ACT episode 0's qpos[0] against the source
    h5 it was converted from.
    """
    raw = bytes(src[f"{traj_key}/obs/agent/qpos"][t].tolist()).rstrip(b"\x00")
    payload = json.loads(raw.decode("utf-8", errors="strict"))
    arm = np.asarray(payload["arm"], dtype=np.float32)
    gripper = np.asarray(payload["gripper"], dtype=np.float32)
    return np.concatenate([arm, gripper], axis=0)


def _qpos_signature(src: h5py.Group, traj_key: str, timesteps=QPOS_SIGNATURE_TIMESTEPS) -> np.ndarray:
    """Stack qpos at a fixed set of timesteps → unique-per-traj signature."""
    return np.stack([_decode_qpos(src, traj_key, t) for t in timesteps], axis=0)


def _qpos_signature_from_act(act_qpos: np.ndarray, timesteps=QPOS_SIGNATURE_TIMESTEPS) -> np.ndarray:
    return np.stack([act_qpos[t] for t in timesteps], axis=0)


def _list_canonical_sensors(src: h5py.Group, traj_key: str) -> List[str]:
    """Return the sorted list of `link{2,3,5,6}_sensor_*` keys for a traj."""
    keys = sorted(src[f"{traj_key}/obs/proximity"].keys())
    return [k for k in keys if any(k.startswith(L + "_sensor_") for L in SENSOR_LINKS)]


def _index_source_trajs(source_files: List[str]) -> List[Tuple[str, str, np.ndarray]]:
    """Walk each source h5; return [(abs_path, traj_key, qpos_signature), ...]."""
    out: List[Tuple[str, str, np.ndarray]] = []
    for path in source_files:
        try:
            with h5py.File(path, "r") as f:
                traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                for tk in traj_keys:
                    try:
                        sig = _qpos_signature(f, tk)
                        out.append((str(Path(path).resolve()), tk, sig))
                    except (KeyError, ValueError, json.JSONDecodeError) as e:
                        print(f"[index] skip {path}:{tk}: {e}", file=sys.stderr)
        except OSError as e:
            print(f"[index] skip {path}: {e}", file=sys.stderr)
    return out


def _match_episode_to_source(
    act_sig: np.ndarray,
    source_index: List[Tuple[str, str, np.ndarray]],
    atol: float,
) -> Tuple[str, str]:
    """Find the *unique* source traj whose qpos signature matches within atol."""
    hits: List[Tuple[str, str]] = []
    for path, tk, sig in source_index:
        if sig.shape != act_sig.shape:
            continue
        if np.allclose(act_sig, sig, atol=atol):
            hits.append((path, tk))
    if len(hits) == 0:
        raise RuntimeError(f"no source traj matched ACT qpos signature (shape {act_sig.shape})")
    if len(hits) > 1:
        raise RuntimeError(
            f"ACT qpos signature matched multiple sources:\n  "
            + "\n  ".join(f"{p}:{t}" for p, t in hits)
        )
    return hits[0]


def _self_test(mapping: Dict, act_dataset_dir: Path, n_samples: int = 5) -> None:
    """Sanity-check `n_samples` random entries against the actual h5s."""
    rng = np.random.default_rng(0)
    ep_keys = sorted(mapping["episodes"].keys(), key=int)
    sample_keys = rng.choice(ep_keys, size=min(n_samples, len(ep_keys)), replace=False)
    sensor_names = mapping["sensor_names"]
    for ek in sample_keys:
        entry = mapping["episodes"][ek]
        act_path = act_dataset_dir / f"episode_{ek}.hdf5"
        with h5py.File(act_path, "r") as af, h5py.File(entry["source_h5"], "r") as sf:
            T_act = af["observations/qpos"].shape[0]
            src_grp = sf[entry["traj_key"]]
            for sn in sensor_names:
                ds = src_grp[f"obs/proximity/{sn}"]
                T_src = ds.shape[0]
                if T_src < T_act:
                    raise AssertionError(
                        f"episode {ek}: src T={T_src} < act T={T_act} for sensor {sn}"
                    )
                # Spot-read first 2 timesteps to confirm finiteness.
                arr = ds[:2]
                if not np.isfinite(arr).all():
                    raise AssertionError(f"episode {ek}: non-finite values in {sn}")
                if arr.min() < 0.0:
                    raise AssertionError(f"episode {ek}: negative depth in {sn}")
        print(f"[self-test] episode {ek}: {entry['source_h5'].split('/')[-3]}/{entry['traj_key']}  OK")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--act_dataset_dir", required=True,
                   help="Path to ACT-style dataset directory containing episode_*.hdf5.")
    p.add_argument(
        "--source_glob",
        default=str(
            Path(__file__).resolve().parents[2]
            / "assets/datagen/mug_house_1_random_everything"
              "/FrankaSkinPickAndPlacePilotMediumConfig"
              "/*/house_*/trajectories_batch_*.h5"
        ),
        help="Glob for source h5 trajectories (must cover all ACT episodes).",
    )
    p.add_argument("--out", default=None,
                   help="Output mapping JSON path (default: <act_dataset_dir>/prox_mapping.json).")
    p.add_argument("--qpos_atol", type=float, default=1e-6,
                   help="Absolute tolerance for the qpos[0] equality match.")
    p.add_argument("--n_self_test", type=int, default=5,
                   help="Number of random entries to self-test against the actual h5s.")
    args = p.parse_args()

    act_dir = Path(args.act_dataset_dir).resolve()
    if not act_dir.is_dir():
        print(f"[map] {act_dir} is not a directory", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else act_dir / "prox_mapping.json"

    act_files = sorted(act_dir.glob("episode_*.hdf5"))
    if not act_files:
        print(f"[map] no episode_*.hdf5 under {act_dir}", file=sys.stderr)
        return 1
    print(f"[map] {len(act_files)} ACT episodes under {act_dir}")

    source_files = sorted(glob.glob(args.source_glob))
    if not source_files:
        print(f"[map] no source h5s matched {args.source_glob!r}", file=sys.stderr)
        return 1
    print(f"[map] {len(source_files)} source h5s discovered via {args.source_glob}")

    source_index = _index_source_trajs(source_files)
    print(f"[map] {len(source_index)} source trajectories indexed")

    # Determine canonical sensor list from the first source traj; assert consistency.
    canonical_path, canonical_tk, _ = source_index[0]
    with h5py.File(canonical_path, "r") as sf:
        sensor_names = _list_canonical_sensors(sf, canonical_tk)
        for path, tk, _ in source_index[1:]:
            with h5py.File(path, "r") as sf2:
                names = _list_canonical_sensors(sf2, tk)
                if names != sensor_names:
                    print(
                        f"[map] sensor list mismatch: {path}:{tk} has {names!r}, "
                        f"expected {sensor_names!r}",
                        file=sys.stderr,
                    )
                    return 1
    print(f"[map] canonical sensor list: {len(sensor_names)} sensors -> {sensor_names[:3]} ... {sensor_names[-3:]}")

    episodes: Dict[str, Dict[str, str]] = {}
    used_sources: set[Tuple[str, str]] = set()
    for ep_path in act_files:
        ep_idx = ep_path.stem.split("_", 1)[1]
        with h5py.File(ep_path, "r") as af:
            act_qpos = af["observations/qpos"][:].astype(np.float32)
        if act_qpos.shape[0] <= max(QPOS_SIGNATURE_TIMESTEPS):
            print(
                f"[map] episode_{ep_idx}.hdf5: too short (T={act_qpos.shape[0]}) "
                f"for the qpos signature (needs t>{max(QPOS_SIGNATURE_TIMESTEPS)})",
                file=sys.stderr,
            )
            return 1
        act_sig = _qpos_signature_from_act(act_qpos)
        try:
            src_h5, traj_key = _match_episode_to_source(act_sig, source_index, args.qpos_atol)
        except RuntimeError as e:
            print(f"[map] episode_{ep_idx}.hdf5: {e}", file=sys.stderr)
            return 1
        key = (src_h5, traj_key)
        if key in used_sources:
            print(
                f"[map] episode_{ep_idx}.hdf5 matched a source traj already used "
                f"by another episode: {src_h5}:{traj_key}",
                file=sys.stderr,
            )
            return 1
        used_sources.add(key)
        episodes[ep_idx] = {"source_h5": src_h5, "traj_key": traj_key}

    mapping = {
        "act_dataset_dir": str(act_dir),
        "source_glob":     args.source_glob,
        "sensor_names":    sensor_names,
        "n_sensors":       len(sensor_names),
        "qpos_atol":       args.qpos_atol,
        "episodes":        episodes,
    }

    _self_test(mapping, act_dir, n_samples=args.n_self_test)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2))
    print(f"[map] wrote {out_path}  ({len(episodes)} episodes)")
    print("MAPPING OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
