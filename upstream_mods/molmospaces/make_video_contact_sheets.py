import csv
import subprocess
from pathlib import Path

IN = Path("tight_space_pick_omni_summary.csv")
OUTDIR = Path("tight_space_contact_sheets")
OUTDIR.mkdir(exist_ok=True)

rows = list(csv.DictReader(IN.open()))
rows = sorted(rows, key=lambda r: (int(r["house"]), r["timestamp"], r["traj"]))

# 先只做前 20 个，确认 ffmpeg 能跑，别一次全跑卡住
rows = rows[:20]

for i, r in enumerate(rows, 1):
    video = Path(r["exo_video"])
    if not video.exists():
        print("missing:", video)
        continue

    out = OUTDIR / f"{i:02d}_house{r['house']}_{r['timestamp']}_{r['traj']}_exo.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", "fps=1,scale=320:-1,tile=5x4",
        "-frames:v", "1",
        str(out),
    ]

    print(f"[{i:02d}] {video} -> {out}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print()
print("wrote contact sheets to:", OUTDIR)
print("num jpg:", len(list(OUTDIR.glob('*.jpg'))))
