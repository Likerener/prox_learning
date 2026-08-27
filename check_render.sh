#!/usr/bin/env bash
# Verify GPU-backed headless EGL rendering and report which EGL device to use.
source "$(dirname "$0")/openfront_env.sh"
unset MUJOCO_EGL_DEVICE_ID
echo "== EGL devices =="
"$PY" - <<'PYEOF'
import importlib
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
from OpenGL import GL
from mujoco.egl import egl_ext as EGL

devs = EGL.eglQueryDevicesEXT()
print(f"  n_devices = {len(devs)}")
good = []
for i in range(len(devs)):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
    try:
        import molmo_spaces.renderer.opengl_context as oc

        importlib.reload(oc)
        oc.EGLGLContext(64, 64, i).make_current()
        r = GL.glGetString(GL.GL_RENDERER).decode()
        gpu = "llvmpipe" not in r and "softpipe" not in r
        print(f"  device {i}: {r}  ->  {'GPU' if gpu else 'SOFTWARE'}")
        if gpu:
            good.append(i)
    except Exception as e:
        print(f"  device {i}: unusable ({type(e).__name__})")
print()
print(f"  USE: export MUJOCO_EGL_DEVICE_ID={good[0]}" if good
      else "  NO GPU EGL DEVICE — still software only.")
PYEOF
