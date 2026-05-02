"""Reconstruct a 3D point cloud from the 29 proximity sensors over a whole
trajectory and render it as a multi-view PNG + MP4.

Pipeline per timestep t and per patch p:
  1. depth_p  = obs/extra/proximity[t, p, :, :]              # (8, 8) float32
  2. pinhole unproject with 180° flip:
        x_body =  u · α · depth
        y_body = -v · α · depth
        z_body =       depth
  3. pybullet FK on fr3_full_skin.urdf with panda[t, 0:7] → patch body pose
     in robot base frame (pos_base, R_base)
  4. robot base → world via obs/extra/robot_base_pose[t]
  5. filter to valid taxels (znear + ε < depth < zfar - ε)

Outputs (default — override via env vars PLA_CVAE_RUN, PLA_PROX_H5):
  <RUN>/plots_data/sensor_pointcloud.png
  <RUN>/plots_data/sensor_pointcloud_traj{TRAJ}.mp4
"""
from __future__ import annotations
import os, re
from pathlib import Path
import numpy as np
import h5py
import pybullet as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

REPO_ROOT = Path(__file__).resolve().parents[2]
H5 = Path(os.environ.get(
    'PLA_PROX_H5',
    str(REPO_ROOT / 'data/skin_pick_fixed_v1/house_0/trajectories_batch_1_of_1.h5'),
))
RUN = Path(os.environ.get('PLA_CVAE_RUN', str(REPO_ROOT / 'runs/cvae_v3')))
OUT_DIR = RUN / 'plots_data'
URDF_SRC = REPO_ROOT / 'assets/urdf/fr3_full_skin.urdf'
URDF_CLEAN = Path('/tmp/fr3_skinonly.urdf')
TRAJ_IDX = 0

FOVY_DEG, ZNEAR, ZFAR, RES = 45.0, 0.02, 4.0, 8

MJCF_ORDER = (
    [f'link6_sensor_{i}' for i in range(8)] +
    [f'link5_sensor_{i}' for i in range(6)] +
    [f'link3_sensor_{i}' for i in range(8)] +
    [f'link2_sensor_{i}' for i in range(7)]
)


def load_fk_skeleton():
    text = open(URDF_SRC).read()
    text = re.sub(r'<visual[^>]*>.*?</visual>', '', text, flags=re.S)
    text = re.sub(r'<collision[^>]*>.*?</collision>', '', text, flags=re.S)
    URDF_CLEAN.write_text(text)
    if not p.isConnected():
        p.connect(p.DIRECT)
    rid = p.loadURDF(str(URDF_CLEAN), useFixedBase=True)
    arm_joints = {}
    link_to_idx = {}
    for i in range(p.getNumJoints(rid)):
        ji = p.getJointInfo(rid, i)
        jname, jtype, link_name = ji[1].decode(), ji[2], ji[12].decode()
        if jtype != p.JOINT_FIXED:
            arm_joints[jname] = i
        link_to_idx[link_name] = i
    sensor_links = [(n, link_to_idx[n]) for n in MJCF_ORDER if n in link_to_idx]
    return rid, arm_joints, sensor_links


def build_world_pts(panda, prox, rbp):
    rid, arm_joints, sensor_links = load_fk_skeleton()
    assert len(sensor_links) == 29, f'got {len(sensor_links)} sensor links'
    base_pos = rbp[0, 0:3].astype(np.float32)
    base_R = np.array(p.getMatrixFromQuaternion(rbp[0, 3:7])).reshape(3, 3).astype(np.float32)

    half = np.tan(np.deg2rad(FOVY_DEG) / 2)
    u = (np.arange(RES) + 0.5) / RES * 2 - 1
    uu, vv = np.meshgrid(u, u, indexing='xy')
    T = panda.shape[0]

    all_pts, all_t, all_p, all_d = [], [], [], []
    arm_names = [f'fr3_joint{i}' for i in range(1, 8)]
    for t in range(T):
        for nm, q in zip(arm_names, panda[t, :7]):
            p.resetJointState(rid, arm_joints[nm], float(q))
        for pidx, (_, lidx) in enumerate(sensor_links):
            depth = prox[t, pidx]
            m = (depth > ZNEAR + 1e-3) & (depth < ZFAR - 1e-3)
            if not m.any(): continue
            ls = p.getLinkState(rid, lidx, computeForwardKinematics=True)
            pos = np.array(ls[4]); R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
            d = depth[m]
            uu_m, vv_m = uu[m], vv[m]
            pts_body = np.stack([uu_m * half * d, -vv_m * half * d, d], axis=-1)
            pts_base = pos[None, :] + pts_body @ R.T
            pts_world = pts_base @ base_R.T + base_pos[None, :]
            all_pts.append(pts_world); all_d.append(d)
            all_t.append(np.full(len(d), t)); all_p.append(np.full(len(d), pidx))
    return (np.concatenate(all_pts), np.concatenate(all_t),
            np.concatenate(all_p), np.concatenate(all_d),
            base_pos, base_R)


def multiview_png(pts, t_arr, d_arr, base_pos, out_path, obj0=None, objE=None, tcp_world=None):
    fig = plt.figure(figsize=(16, 12), dpi=110)
    views = [('Top (X-Y)', 0, 1), ('Front (X-Z)', 0, 2), ('Side (Y-Z)', 1, 2)]
    for k, (title, i, j) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, k)
        sc = ax.scatter(pts[:, i], pts[:, j], c=t_arr, cmap='viridis', s=1.5, alpha=0.6)
        if tcp_world is not None:
            ax.plot(tcp_world[:, i], tcp_world[:, j], 'r-', lw=1.0, alpha=0.7, label='TCP path')
        if obj0 is not None:
            ax.scatter([obj0[i]], [obj0[j]], c='lime', s=120, marker='*', edgecolor='k', label='object start')
        if objE is not None:
            ax.scatter([objE[i]], [objE[j]], c='orange', s=120, marker='*', edgecolor='k', label='object end')
        ax.set_title(title); ax.set_xlabel('xyz'[i]); ax.set_ylabel('xyz'[j])
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label='timestep', shrink=0.7)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=t_arr, cmap='viridis', s=1.5, alpha=0.6)
    if tcp_world is not None:
        ax.plot(tcp_world[:, 0], tcp_world[:, 1], tcp_world[:, 2], 'r-', lw=1.0, alpha=0.7)
    if obj0 is not None:
        ax.scatter(*obj0, c='lime', s=120, marker='*', edgecolor='k')
    if objE is not None:
        ax.scatter(*objE, c='orange', s=120, marker='*', edgecolor='k')
    ax.set_title('3D')
    fig.suptitle(
        f'Sensor-derived point cloud, traj_{TRAJ_IDX}  ({len(pts)} points, {int(t_arr.max() + 1)} timesteps)',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'wrote {out_path}')


def animation_mp4(pts, t_arr, d_arr, base_pos, out_path, n_frames):
    fig = plt.figure(figsize=(8, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    def update(fi):
        ax.clear()
        m = t_arr <= fi
        if m.any():
            ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], c=d_arr[m], cmap='viridis_r', s=1.5, alpha=0.5, vmin=0, vmax=ZFAR)
        ax.set_xlim(base_pos[0] - 1.5, base_pos[0] + 1.5)
        ax.set_ylim(base_pos[1] - 1.5, base_pos[1] + 1.5)
        ax.set_zlim(max(0, base_pos[2] - 0.5), base_pos[2] + 1.8)
        ax.view_init(elev=22, azim=40 + 0.8 * fi)
        ax.set_title(f'accumulated sensor points up to t={fi}')
        return []
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(out_path), writer=animation.FFMpegWriter(fps=6), dpi=100)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    with h5py.File(H5, 'r') as f:
        panda = f[f'traj_{TRAJ_IDX}/env_states/articulations/panda'][:]
        prox = f[f'traj_{TRAJ_IDX}/obs/extra/proximity'][:]
        tcp = f[f'traj_{TRAJ_IDX}/obs/extra/tcp_pose'][:]
        rbp = f[f'traj_{TRAJ_IDX}/obs/extra/robot_base_pose'][:]
        obj0 = f[f'traj_{TRAJ_IDX}/obs/extra/obj_start'][0, :3]
        objE = f[f'traj_{TRAJ_IDX}/obs/extra/obj_end'][-1, :3]

    T = panda.shape[0]
    print(f'traj_{TRAJ_IDX}: T={T}  base={rbp[0, :3]}  object=({obj0[0]:.2f},{obj0[1]:.2f},{obj0[2]:.2f}) → ({objE[0]:.2f},{objE[1]:.2f},{objE[2]:.2f})')

    pts, t_arr, p_arr, d_arr, base_pos, base_R = build_world_pts(panda, prox, rbp)
    print(f'total hit points: {len(pts)} (from {T} steps × 29 patches × 64 taxels = {T*29*64})  valid rate={len(pts)/(T*29*64):.2%}')

    base_pos = rbp[0, :3]; base_R = np.array(p.getMatrixFromQuaternion(rbp[0, 3:7])).reshape(3, 3)
    tcp_world = tcp[:, :3] @ base_R.T + base_pos[None, :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    multiview_png(pts, t_arr, d_arr, base_pos,
                   OUT_DIR / 'sensor_pointcloud.png',
                   obj0=obj0, objE=objE, tcp_world=tcp_world)
    animation_mp4(pts, t_arr, d_arr, base_pos,
                   OUT_DIR / f'sensor_pointcloud_traj{TRAJ_IDX}.mp4', n_frames=T)


if __name__ == '__main__':
    main()
