import h5py
import json
import csv
import re
from pathlib import Path
import numpy as np

ROOT = Path("assets/experiment_output/datagen/pick_omni_v1/FrankaPickOmniCamConfig")
OUT = Path("tight_space_pick_omni_summary.csv")

def decode_task_info(row):
    try:
        b = bytes(row.tolist()).rstrip(b"\x00")
        if not b:
            return {}
        return json.loads(b.decode("utf-8", errors="ignore"))
    except Exception:
        return {}

def parse_house(fp):
    m = re.search(r"house_(\d+)", str(fp))
    return int(m.group(1)) if m else None

def dist(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a[:3] - b[:3]))

rows = []

files = sorted(ROOT.glob("*/house_*/trajectories_batch_*.h5"))
print("num h5:", len(files))

for fp in files:
    timestamp = fp.parts[-3]
    house = parse_house(fp)

    with h5py.File(fp, "r") as f:
        for traj_name in sorted(f.keys()):
            g = f[traj_name]
            extra = g["obs/extra"]

            n = len(extra["tcp_pose"])
            phase = extra["policy_phase"][:]
            tcp = extra["tcp_pose"][:]
            obj_start = extra["obj_start"][:]
            obj_end = extra["obj_end"][:]
            retries = extra["policy_num_retries"][:] if "policy_num_retries" in extra else np.zeros(n)

            final_info = decode_task_info(extra["task_info"][-1])
            success = final_info.get("success", None)
            final_pos_error = final_info.get("position_error", None)
            final_rot_error = final_info.get("rotation_error", None)

            # distance metrics
            tcp_obj_dists = np.linalg.norm(tcp[:, :3] - obj_start[:, :3], axis=1)
            min_tcp_obj = float(np.min(tcp_obj_dists))
            final_tcp_obj = float(tcp_obj_dists[-1])

            # pregrasp phase normally seems to be phase == 2
            pre_mask = (phase == 2)
            if pre_mask.any():
                pre_min_tcp_obj = float(np.min(tcp_obj_dists[pre_mask]))
                pre_frames = int(pre_mask.sum())
            else:
                pre_min_tcp_obj = ""
                pre_frames = 0

            # object moved distance
            object_move_dist = dist(obj_start[0], obj_end[-1])

            # corresponding videos
            house_dir = fp.parent
            traj_idx = int(traj_name.replace("traj_", ""))
            video_prefix = f"episode_{traj_idx:08d}"
            videos = sorted(house_dir.glob(video_prefix + "*.mp4"))

            rows.append({
                "timestamp": timestamp,
                "house": house,
                "traj": traj_name,
                "h5_path": str(fp),
                "frames": n,
                "success": success,
                "final_position_error": final_pos_error,
                "final_rotation_error": final_rot_error,
                "policy_phases": ",".join(map(str, sorted(set(phase.tolist())))),
                "pregrasp_frames": pre_frames,
                "min_tcp_to_obj_m": round(min_tcp_obj, 4),
                "pregrasp_min_tcp_to_obj_m": round(pre_min_tcp_obj, 4) if pre_min_tcp_obj != "" else "",
                "final_tcp_to_obj_m": round(final_tcp_obj, 4),
                "object_move_dist_m": round(object_move_dist, 4),
                "max_policy_retries": int(np.max(retries)),
                "num_videos": len(videos),
                "wrist_video": str(next((v for v in videos if "wrist_camera_zed_mini_batch" in v.name and "depth" not in v.name), "")),
                "exo_video": str(next((v for v in videos if "randomized_zed2_analogue_1_batch" in v.name and "depth" not in v.name), "")),
                "all_videos_dir": str(house_dir),
            })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("wrote:", OUT)
print("num trajectories:", len(rows))

print("\n=== first 10 rows ===")
for r in rows[:10]:
    print(
        "ts=", r["timestamp"],
        "house=", r["house"],
        "traj=", r["traj"],
        "frames=", r["frames"],
        "success=", r["success"],
        "pre_min=", r["pregrasp_min_tcp_to_obj_m"],
        "wrist=", r["wrist_video"]
    )
