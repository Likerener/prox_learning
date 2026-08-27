export PL=~/molmo_test/prox_learning
export PY=~/molmo_test/molmospaces/.venv/bin/python
# The fume-hood configs live on the fork's `main`; the submodule sits on `envs`
# (open-front configs) and the standalone checkout is a different lineage that
# predates them. $PL/ms_main is a worktree of `main` so both stay available.
export PYTHONPATH="$PL/ms_main:$PL/submodules/act:$PL/submodules/act/detr:$PL"
# Assets are cached per installation path; point at the submodule's existing
# cache so the scene and robot downloads are reused rather than repeated.
export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets/$(printf '%s' "$HOME/molmo_test/prox_learning/submodules/molmospaces" | base64 -w0 | tr '+/' '-_' | tr -d '=')"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
