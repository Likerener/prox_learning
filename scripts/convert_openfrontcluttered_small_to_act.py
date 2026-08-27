from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.convert_pla_to_act import convert

SRC_ROOT = REPO / (
    "assets/datagen/roger_open_front_cluttered_sweep_12_80/"
    "FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig"
)
DST = REPO / "act_style_data/openfrontcluttered_small_20260622"

def main():
    files = sorted(SRC_ROOT.rglob("trajectories_batch_1_of_1.h5"))
    if not files:
        raise RuntimeError(f"No source H5 files found under {SRC_ROOT}")

    if DST.exists() and any(DST.glob("episode_*.hdf5")):
        raise RuntimeError(
            f"{DST} already contains episodes. Refusing to overwrite."
        )

    DST.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    for src_h5 in files:
        stage = DST / f".stage_{src_h5.parent.name}"

        if stage.exists():
            shutil.rmtree(stage)

        print(f"[convert] source: {src_h5}")
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

        if not produced:
            print(f"[convert] WARNING: no episode produced from {src_h5}")
            shutil.rmtree(stage, ignore_errors=True)
            continue

        for episode in produced:
            dst_episode = DST / f"episode_{global_idx}.hdf5"
            if dst_episode.exists():
                raise RuntimeError(f"Refusing to overwrite {dst_episode}")

            shutil.move(str(episode), str(dst_episode))
            print(
                f"[convert] {src_h5.parent.name}/{episode.name}"
                f" -> {dst_episode.name}"
            )
            global_idx += 1

        shutil.rmtree(stage, ignore_errors=True)

    print(f"[convert] DONE: {global_idx} episodes written to {DST}")

if __name__ == "__main__":
    main()
