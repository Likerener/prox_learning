from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.convert_pla_to_act import convert

SRC_ROOT = REPO / (
    "assets/datagen/roger_open_front_cluttered_40traj_20260622/"
    "FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig"
)
DST = REPO / "act_style_data/openfrontcluttered_52_20260623"

def main():
    files = sorted(
        p for p in SRC_ROOT.rglob("trajectories_batch_*.h5")
        if "debug" not in p.parts
    )

    if not files:
        raise RuntimeError(f"No source H5 files found under {SRC_ROOT}")

    if DST.exists() and any(DST.glob("episode_*.hdf5")):
        raise RuntimeError(f"{DST} already contains episodes")

    DST.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    for src_h5 in files:
        run_id = src_h5.parents[1].name
        house = src_h5.parent.name
        stage = DST / f".stage_{run_id}_{house}"

        shutil.rmtree(stage, ignore_errors=True)

        print(f"[convert] {run_id}/{house}")
        convert(
            src_h5=src_h5,
            dst_dir=stage,
            image_h=240,
            image_w=320,
            max_episodes=None,
        )

        produced = sorted(
            stage.glob("episode_*.hdf5"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        for episode in produced:
            dst_episode = DST / f"episode_{global_idx}.hdf5"
            shutil.move(str(episode), str(dst_episode))
            print(f"  {episode.name} -> {dst_episode.name}")
            global_idx += 1

        shutil.rmtree(stage, ignore_errors=True)

    print(f"[convert] DONE: {global_idx} episodes written to {DST}")

if __name__ == "__main__":
    main()
