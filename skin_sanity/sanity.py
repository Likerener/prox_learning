"""Minimal proximity-sensor sanity check.

Scene:
  - One camera body "sensor" at world pose (pos=(0,0,0.5), identity quat).
    Inside it: `<camera quat="0 1 0 0" fovy="45" resolution="8 8"/>` — same
    convention as the real Franka skin patches (180° about X so that
    MuJoCo's -Z camera ray aligns with body +Z).
  - A flat axis-aligned wall (plane-like thin box) perpendicular to the
    sensor's +Z axis at a known distance.

We render depth via `mujoco.Renderer` (depth mode), then reconstruct world-
frame 3D points with the exact formula used on the real data:

    half = tan(fovy/2)
    u = (arange(8)+0.5)/8 * 2 - 1           # in [-7/8..7/8]
    uu, vv = meshgrid(u, u, indexing='xy')
    dirs_body = normalize([uu*half, -vv*half, 1])     # body +Z forward
    pts_world = pos_world + depth * (R_world @ dirs_body)

If the convention is right, every reconstructed point should lie on the
wall plane (z = wall_z) within ~1 mm. If the ray formula is mirrored or
rotated wrong, the points will diverge from the wall predictably.

Also runs a second scene with the wall tilted 30° about Y so we can check
that *non-axial* rays land correctly (tests the uu/vv meshgrid sign/order).
"""
from __future__ import annotations
import h5py, numpy as np, mujoco, os, sys, json
import plotly.graph_objects as go

OUT_DIR = '/home/jaydv/code/skin_sanity'
FOVY_DEG = 45.0
RES = 8
ZNEAR, ZFAR = 0.02, 4.0

# --- Scene parameters ---
SENSOR_POS = np.array([0.0, 0.0, 0.5])    # world position of sensor body
WALL_DIST  = 0.30                           # distance along sensor +Z (= world +Z here)
WALL_Z     = SENSOR_POS[2] + WALL_DIST      # = 0.80
WALL_THICK = 0.005                          # 5 mm thin slab
WALL_HALF  = 0.5                            # 1x1m wall so all rays hit

# --- MJCF ---
def mjcf_axial() -> str:
    return f"""
<mujoco model="sanity_axial">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual><map znear="0.005" zfar="10"/></visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <!-- sensor body at world pos=(0,0,0.5), identity quat -->
    <body name="sensor" pos="{SENSOR_POS[0]} {SENSOR_POS[1]} {SENSOR_POS[2]}" quat="1 0 0 0">
      <site name="sensor_site" type="sphere" size="0.01" rgba="1 0 0 1"/>
      <camera name="sensor_cam" mode="fixed" pos="0 0 0" quat="0 1 0 0"
              fovy="{FOVY_DEG}" resolution="{RES} {RES}"/>
    </body>
    <!-- wall: thin box perpendicular to world +Z, centered at z=WALL_Z -->
    <body name="wall" pos="0 0 {WALL_Z}">
      <geom name="wall_geom" type="box"
            size="{WALL_HALF} {WALL_HALF} {WALL_THICK}"
            rgba="0.3 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def mjcf_tilted(tilt_rad: float = np.deg2rad(30)) -> str:
    # Rotate wall 30 deg about Y so rays hit a non-perpendicular surface.
    # Wall quat from (rpy = (0, tilt, 0)) — URDF-style: q = [w,x,y,z]
    cy, sy = np.cos(tilt_rad/2), np.sin(tilt_rad/2)
    qw, qx, qy, qz = cy, 0.0, sy, 0.0
    return f"""
<mujoco model="sanity_tilted">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual><map znear="0.005" zfar="10"/></visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="sensor" pos="{SENSOR_POS[0]} {SENSOR_POS[1]} {SENSOR_POS[2]}" quat="1 0 0 0">
      <site name="sensor_site" type="sphere" size="0.01" rgba="1 0 0 1"/>
      <camera name="sensor_cam" mode="fixed" pos="0 0 0" quat="0 1 0 0"
              fovy="{FOVY_DEG}" resolution="{RES} {RES}"/>
    </body>
    <body name="wall" pos="0 0 {WALL_Z}" quat="{qw} {qx} {qy} {qz}">
      <geom name="wall_geom" type="box"
            size="{WALL_HALF} {WALL_HALF} {WALL_THICK}"
            rgba="0.9 0.5 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# --- Render depth from sensor_cam ---
def render_depth(xml: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=RES, width=RES) as r:
        r.enable_depth_rendering()
        r.update_scene(data, camera='sensor_cam')
        depth = r.render().copy()
    depth[depth < ZNEAR] = np.nan
    depth[depth > ZFAR]  = np.nan
    return depth  # (8,8)

# --- Reconstruction -------------------------------------------------------
# MuJoCo's `enable_depth_rendering()` returns **linear axial depth** (the z-
# component of the hit point in the camera frame), NOT ray length. Using the
# wrong interpretation was the original bug. For a pinhole camera with an 8x8
# grid and fovy=FOVY_DEG, the correct unproject is:
#     x_cam = u * half * depth,   y_cam = v * half * depth,   z_cam = depth
# where (u, v) are normalized image coords in [-1, 1].
#
# Then we map camera frame -> body frame. The sensor's <camera> has
# quat="0 1 0 0" (180 deg about X), which means R_body_from_cam =
#   diag(1, -1, -1)  — camera +X = body +X, camera +Y = body -Y, camera +Z = body -Z.
# The camera looks along -Z_cam; after the flip, that's along +Z_body.
# So in body frame:
#     x_body = +u * half * depth
#     y_body = -v * half * depth   (sign flip from camera +Y -> body -Y)
#     z_body = +depth              (MuJoCo returns positive depth; the flip
#                                   puts it along body +Z)

def reconstruct(depth: np.ndarray, sensor_pos: np.ndarray, sensor_R: np.ndarray):
    half = np.tan(np.deg2rad(FOVY_DEG) / 2.0)
    u = (np.arange(RES) + 0.5) / RES * 2 - 1       # image x coord, [-7/8 .. 7/8]
    uu, vv = np.meshgrid(u, u, indexing='xy')       # (H,W) grid
    mask = np.isfinite(depth)
    d = depth[mask]
    uu_m, vv_m = uu[mask], vv[mask]
    x_body =  uu_m * half * d
    y_body = -vv_m * half * d
    z_body = d
    pts_body = np.stack([x_body, y_body, z_body], axis=-1)  # (K, 3)
    pts = sensor_pos[None, :] + pts_body @ sensor_R.T
    return pts, None, mask

# ================================================================
# RUN
# ================================================================
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

# --- axial wall ---
xml = mjcf_axial()
depth = render_depth(xml)
pts, dirs_body, mask = reconstruct(depth, SENSOR_POS, np.eye(3))

# Expected depth for each taxel: wall is perpendicular to ray along +Z of body
# depth_expected = WALL_DIST / dir_body_z  (since dir_body is unit, z-component
# = 1/||[uu*half, -vv*half, 1]||). So depth * dir_body_z = WALL_DIST.
# Easier: reconstructed z must equal WALL_Z for ALL points.
# In MuJoCo, <geom type="box" size="a b c"> is half-size. The slab extends
# z = WALL_Z ± WALL_THICK. The front face (closest to sensor) sits at
# WALL_FRONT = WALL_Z - WALL_THICK.
WALL_FRONT = WALL_Z - WALL_THICK
z_vals = pts[:, 2]
axial_err = np.abs(z_vals - WALL_FRONT)
print(f'\n=== axial wall (front face z={WALL_FRONT:.4f}) ===')
print(f'depth grid min={np.nanmin(depth):.4f}  max={np.nanmax(depth):.4f}  ({mask.sum()} valid)')
print(f'  (uniform = axial/orthographic depth, NOT ray length)')
print(f'reconstructed z: min={z_vals.min():.4f}  max={z_vals.max():.4f}  mean={z_vals.mean():.4f}')
print(f'|z - WALL_FRONT| max={axial_err.max() * 1000:.3f} mm  mean={axial_err.mean() * 1000:.3f} mm')

results['axial'] = dict(pts=pts, depth=depth,
                        sensor_pos=SENSOR_POS, wall_front=WALL_FRONT)

# --- tilted wall ---
xml = mjcf_tilted()
depth_t = render_depth(xml)
pts_t, _, mask_t = reconstruct(depth_t, SENSOR_POS, np.eye(3))

# Tilted wall plane: normal n = R_y(30) @ +z = [sin(30), 0, cos(30)].
# The front face (in wall-local z = -WALL_THICK) passes through
# p0 = center + R @ [0,0,-WALL_THICK].
tilt = np.deg2rad(30)
n = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
# wall center is at (0,0,WALL_Z); front face offset by -WALL_THICK along its local +z,
# which in world is (-WALL_THICK) * n
p0 = np.array([0, 0, WALL_Z]) - WALL_THICK * n
dist = (pts_t - p0[None, :]) @ n
print(f'\n=== tilted wall (30 deg about Y) ===')
print(f'depth grid min={np.nanmin(depth_t):.4f}  max={np.nanmax(depth_t):.4f}  ({mask_t.sum()} valid)')
print(f'signed dist to wall plane:  min={dist.min():.4f}  max={dist.max():.4f}')
print(f'|dist| max={np.abs(dist).max()*1000:.3f} mm  mean={np.abs(dist).mean()*1000:.3f} mm')
results['tilted'] = dict(pts=pts_t, depth=depth_t, wall_normal=n, wall_point=p0)

# --- Write synthetic HDF5 matching real-data shape for downstream test ---
h5path = os.path.join(OUT_DIR, 'synthetic.h5')
with h5py.File(h5path, 'w') as f:
    # reshape to (T=1, N=1, 8, 8)
    g = f.create_group('traj_0/obs/extra')
    g.create_dataset('proximity', data=depth[None, None].astype('float32'))
    # minimal env_states stub — sensor pose is fixed, no arm joints
    f.create_dataset('traj_0/env_states/articulations/panda',
                     data=np.zeros((1, 31), dtype='float32'))
    f.create_dataset('traj_0/sensor_pos', data=SENSOR_POS.astype('float32'))
    f.create_dataset('traj_0/sensor_quat', data=np.array([1,0,0,0], dtype='float32'))
    f.create_dataset('traj_0/wall_z', data=np.array([WALL_Z], dtype='float32'))
print(f'wrote synthetic HDF5 -> {h5path}')

# --- Plotly visualization ---
def plot(pts_axial, pts_tilted, wall_z, n, p0):
    fig = go.Figure()
    # sensor
    fig.add_trace(go.Scatter3d(
        x=[SENSOR_POS[0]], y=[SENSOR_POS[1]], z=[SENSOR_POS[2]],
        mode='markers+text', marker=dict(size=6,color='red',symbol='diamond'),
        text=['sensor'], name='sensor', textposition='top center'))
    # axial wall mesh (square at z=wall_z)
    xs = [-WALL_HALF, WALL_HALF, WALL_HALF, -WALL_HALF, -WALL_HALF]
    ys = [-WALL_HALF, -WALL_HALF, WALL_HALF, WALL_HALF, -WALL_HALF]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[wall_z]*5,
        mode='lines', line=dict(color='blue',width=3), name=f'axial wall (z={wall_z:.2f})'))
    # axial pts
    fig.add_trace(go.Scatter3d(
        x=pts_axial[:,0], y=pts_axial[:,1], z=pts_axial[:,2],
        mode='markers', marker=dict(size=4,color='green'),
        name=f'reconstructed (axial)  N={len(pts_axial)}'))
    # tilted pts
    fig.add_trace(go.Scatter3d(
        x=pts_tilted[:,0], y=pts_tilted[:,1], z=pts_tilted[:,2],
        mode='markers', marker=dict(size=4,color='orange'),
        name=f'reconstructed (tilted)  N={len(pts_tilted)}'))
    # tilted wall outline: 4 corners in its local frame (x in [-h, h], y in [-h,h], z=0)
    # rotate about Y by 30, translate by (0,0,wall_z)
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    Ry = np.array([[c, 0, s],[0,1,0],[-s,0,c]])
    corners = np.array([[-WALL_HALF,-WALL_HALF,0],[WALL_HALF,-WALL_HALF,0],
                        [WALL_HALF,WALL_HALF,0],[-WALL_HALF,WALL_HALF,0],
                        [-WALL_HALF,-WALL_HALF,0]]) @ Ry.T + p0[None,:]
    fig.add_trace(go.Scatter3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
        mode='lines', line=dict(color='orange',width=3),
        name='tilted wall outline (30° about Y)'))
    fig.update_layout(
        title='Minimal sanity-check: 1 sensor + 1 wall  (green/orange = reconstructed pts)',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'),
        margin=dict(l=0,r=0,t=40,b=0))
    return fig

fig = plot(pts, pts_t, WALL_Z, n, p0)
out_html = os.path.join(OUT_DIR, 'sanity.html')
fig.write_html(out_html, include_plotlyjs='cdn')
print(f'wrote {out_html}')

# --- Summary judgement ---
tol_mm = 5.0
axial_ok  = axial_err.max() * 1000 < tol_mm
tilted_ok = np.abs(dist).max() * 1000 < tol_mm
print()
print(f'AXIAL reconstruction  : {"PASS" if axial_ok  else "FAIL"}  (tolerance {tol_mm} mm)')
print(f'TILTED reconstruction : {"PASS" if tilted_ok else "FAIL"}  (tolerance {tol_mm} mm)')
if axial_ok and tilted_ok:
    print('\n=> Ray-direction convention is CORRECT. The anomalies in the real data')
    print('   (patches with static close readings) are physical, not a math bug.')
else:
    print('\n=> Ray-direction convention is WRONG. Need to adjust dirs_body formula.')
    print('   Candidate fixes: flip vv sign, flip uu sign, swap u/v, or use')
    print('   MuJoCo camera frame (-Z forward) directly instead of body-frame +Z.')
