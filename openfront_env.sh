export PL=~/molmo_test/prox_learning
export PY=~/molmo_test/molmospaces/.venv/bin/python
# The submodule copy (branch `envs`) is the code the openfront data was collected
# with, and it must shadow the standalone ~/molmo_test/molmospaces install.
export PYTHONPATH="$PL/submodules/molmospaces:$PL/submodules/act:$PL/submodules/act/detr:$PL"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
