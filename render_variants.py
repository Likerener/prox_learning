"""Contact sheet of every fume-hood size variant, one tile per scene.

Dimensions are read back off the compiled model rather than the filename, so the
labels describe what MuJoCo actually built.
"""
import glob
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

X_FRONT, Z_BENCH, T = 0.58, 0.72, 0.012
SCENES = "/home/qinzhengfangli/molmo_test/prox_learning/ms_main/molmo_spaces/data_generation/custom_scenes"
OUT = "/home/qinzhengfangli/molmo_test/prox_learning/figs"
os.makedirs(OUT, exist_ok=True)

paths = sorted(glob.glob(f"{SCENES}/fumehood_v*.xml"))
cols, rows = 6, -(-len(paths) // 6)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4))
axes = axes.ravel()

for ax, p in zip(axes, paths):
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    half_w = abs(m.geom("hood_side_l").pos[1]) - T
    depth = m.geom("hood_back").pos[0] - T - X_FRONT
    height = m.geom("hood_top").pos[2] - 0.015 - Z_BENCH

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [X_FRONT + depth / 2, 0.0, Z_BENCH + height / 2]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 35, -15

    r = mujoco.Renderer(m, 384, 512)
    r.update_scene(d, camera=cam)
    img = r.render()
    ax.imshow(img)
    if img.std() < 3.0:
        print('  WARNING: near-uniform render for', os.path.basename(p))
    r.close()
    ax.set_title(f"{os.path.basename(p)[9:-4]}   w={2*half_w:.2f} d={depth:.2f} h={height:.2f}",
                 fontsize=8)
    ax.axis("off")

for ax in axes[len(paths):]:
    ax.axis("off")

fig.suptitle("Fume-hood size variants — interior width x depth x height (m)", fontsize=13)
fig.tight_layout()
out = f"{OUT}/fumehood_size_variants.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
