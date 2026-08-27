from pathlib import Path
import glob
import json
import h5py

REPO = Path("/home/qinzhengfangli/molmo_test/prox_learning").resolve()

ACT_DIR = (
    REPO / "act_style_data/openfrontcluttered_52_20260623"
)

SOURCE_GLOB = str(
    REPO
    / "assets/datagen/roger_open_front_cluttered_40traj_20260622"
    / "FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig"
    / "20260622_*"
    / "house_*"
    / "trajectories_batch_*.h5"
)

OUT = ACT_DIR / "prox_mapping.json"

source_files = sorted(Path(p).resolve() for p in glob.glob(SOURCE_GLOB))

if not source_files:
    raise RuntimeError(f"No source H5 files matched:\n{SOURCE_GLOB}")

episode_files = sorted(
    ACT_DIR.glob("episode_*.hdf5"),
    key=lambda p: int(p.stem.split("_")[1]),
)

if len(episode_files) != 52:
    raise RuntimeError(f"Expected 52 ACT episodes, found {len(episode_files)}")

source_entries = []
sensor_names = None

for source_h5 in source_files:
    with h5py.File(source_h5, "r") as h:
        traj_keys = sorted(
            (k for k in h.keys() if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[1]),
        )

        for traj_key in traj_keys:
            source_entries.append(
                {
                    "source_h5": str(source_h5),
                    "traj_key": traj_key,
                }
            )

            if sensor_names is None:
                traj = h[traj_key]

                candidates = []

                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        leaf = name.split("/")[-1]
                        if (
                            leaf.startswith("link")
                            and "_sensor_" in leaf
                        ):
                            candidates.append(leaf)

                traj.visititems(visit)
                sensor_names = sorted(set(candidates))

if len(source_entries) != 52:
    raise RuntimeError(
        f"Expected 52 source trajectories, found {len(source_entries)}"
    )

if not sensor_names or len(sensor_names) != 29:
    raise RuntimeError(
        f"Expected 29 proximity sensors, found "
        f"{0 if sensor_names is None else len(sensor_names)}: {sensor_names}"
    )

episodes = {
    str(i): entry
    for i, entry in enumerate(source_entries)
}

mapping = {
    "act_dataset_dir": str(ACT_DIR),
    "source_glob": SOURCE_GLOB,
    "sensor_names": sensor_names,
    "n_sensors": len(sensor_names),
    "qpos_atol": 1e-5,
    "episodes": episodes,
}

OUT.write_text(json.dumps(mapping, indent=2))

print(f"[exact-map] source H5 files: {len(source_files)}")
print(f"[exact-map] source trajectories: {len(source_entries)}")
print(f"[exact-map] ACT episodes: {len(episode_files)}")
print(f"[exact-map] sensors: {len(sensor_names)}")
print(f"[exact-map] wrote: {OUT}")

for i in range(min(5, len(source_entries))):
    print(f"  episode_{i}.hdf5 -> {source_entries[i]}")
