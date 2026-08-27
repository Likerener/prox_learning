import csv
import subprocess
from pathlib import Path

IN = Path("tight_space_pick_omni_summary.csv")
OUTDIR = Path("tight_space_contact_sheets_all")
OUTDIR.mkdir(exist_ok=True)

rows = list(csv.DictReader(IN.open()))
rows = sorted(rows, key=lambda r: (int(r["house"]), r["timestamp"], r["traj"]))

print("num rows:", len(rows))

for i, r in enumerate(rows, 1):
    video = Path(r["exo_video"])
    if not video.exists():
        print(f"[{i:02d}] missing:", video)
        continue

    out = OUTDIR / f"{i:02d}_house{r['house']}_{r['timestamp']}_{r['traj']}_exo.jpg"

    if out.exists() and out.stat().st_size > 10000:
        print(f"[{i:02d}] exists {out}")
        continue

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", "fps=1,scale=320:-1,tile=5x4",
        "-frames:v", "1",
        str(out),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        print(f"[{i:02d}] wrote {out}")
    except subprocess.TimeoutExpired:
        print(f"[{i:02d}] timeout: {video}")
    except subprocess.CalledProcessError as e:
        print(f"[{i:02d}] ffmpeg failed: {video} code={e.returncode}")

print()
print("done")
print("num jpg:", len(list(OUTDIR.glob("*.jpg"))))
