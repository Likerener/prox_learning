import csv
from pathlib import Path

IN = Path("tight_space_pick_omni_summary.csv")
OUT = Path("tight_space_video_review_list.txt")

rows = list(csv.DictReader(IN.open()))

# 先按 house / timestamp / traj 排序，人工看起来最清楚
rows = sorted(rows, key=lambda r: (int(r["house"]), r["timestamp"], r["traj"]))

with OUT.open("w") as f:
    f.write("Tight-space candidate video review list\n")
    f.write("=" * 80 + "\n\n")
    f.write("Goal: manually inspect videos for tasks where proximity sensors may help.\n")
    f.write("Look for: cabinet/drawer/shelf/corner/counter-edge/clutter/narrow gaps.\n\n")

    for i, r in enumerate(rows, 1):
        f.write(f"[{i:02d}] house={r['house']} timestamp={r['timestamp']} traj={r['traj']} frames={r['frames']} success={r['success']}\n")
        f.write(f"     wrist: {r['wrist_video']}\n")
        f.write(f"     exo:   {r['exo_video']}\n")
        f.write(f"     dir:   {r['all_videos_dir']}\n")
        f.write("\n")

print("wrote:", OUT)
print("num rows:", len(rows))
print()
print("Preview:")
print(OUT.read_text().splitlines()[0:25])
