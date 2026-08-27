import csv
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

IN = Path("tight_space_pick_omni_summary.csv")
OUTDIR = Path("tight_space_contact_sheets")
OUTDIR.mkdir(exist_ok=True)

rows = list(csv.DictReader(IN.open()))
rows = sorted(rows, key=lambda r: (int(r["house"]), r["timestamp"], r["traj"]))

# 先做前 20 个
rows = rows[:20]

def get_duration(video):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def extract_frame(video, t, out_img):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(t, 0)),
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_img),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for i, r in enumerate(rows, 1):
    video = Path(r["exo_video"])
    if not video.exists():
        print(f"[{i:02d}] missing: {video}")
        continue

    out = OUTDIR / f"{i:02d}_house{r['house']}_{r['timestamp']}_{r['traj']}_exo.jpg"

    try:
        duration = get_duration(video)
    except Exception as e:
        print(f"[{i:02d}] ffprobe failed: {video} | {e}")
        continue

    # 抽 12 张，避开最开头/最结尾
    n = 12
    times = [(j + 0.5) * duration / n for j in range(n)]

    frames = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for j, t in enumerate(times):
            img_path = td / f"frame_{j:02d}.jpg"
            extract_frame(video, t, img_path)
            if img_path.exists() and img_path.stat().st_size > 0:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((320, 180))
                    frames.append((j, img.copy()))
                except Exception:
                    pass

    if not frames:
        print(f"[{i:02d}] no frames extracted: {video}")
        continue

    cell_w, cell_h = 320, 210
    cols = 4
    rows_grid = 3
    sheet = Image.new("RGB", (cols * cell_w, rows_grid * cell_h + 40), "white")
    draw = ImageDraw.Draw(sheet)

    title = f"{i:02d} house={r['house']} {r['timestamp']} {r['traj']} frames={r['frames']} success={r['success']}"
    draw.text((10, 10), title, fill="black")

    for k, (j, img) in enumerate(frames[:12]):
        x = (k % cols) * cell_w
        y = 40 + (k // cols) * cell_h
        sheet.paste(img, (x, y))
        draw.text((x + 5, y + 185), f"t={times[j]:.1f}s", fill="black")

    sheet.save(out, quality=95)
    print(f"[{i:02d}] wrote {out}")

print()
print("done")
print("num jpg:", len(list(OUTDIR.glob('*.jpg'))))
