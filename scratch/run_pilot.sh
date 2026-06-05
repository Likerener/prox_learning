#!/bin/bash
export MLSPACES_ASSETS_DIR='/mnt/d/Machines virtueles/prox_learning/assets'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/amine/molmospaces

echo "Starting pilot run..."
/home/amine/.venvs/prox_learning_molmospaces/bin/python '/mnt/d/Machines virtueles/prox_learning/scratch/run_pilot.py'
echo "Finished pilot run."
