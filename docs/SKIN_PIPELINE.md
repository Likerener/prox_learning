# FR3 Skin + Robotiq Proximity Sensing Pipeline

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Directory Structure](#directory-structure)
5. [Usage: Standalone Rendering](#usage-standalone-rendering)
6. [Usage: MolmoSpaces Data Collection](#usage-molmospaces-data-collection)
7. [Reading Proximity Sensor Data](#reading-proximity-sensor-data)
8. [Creating a Custom Datagen Config](#creating-a-custom-datagen-config)
9. [Sensor Specifications](#sensor-specifications)
10. [Known Issues and Workarounds](#known-issues-and-workarounds)
11. [Development Log](#development-log)
12. [File Reference](#file-reference)

---

## Overview

This pipeline integrates self-capacitive proximity skin sensors onto the Franka FR3 + Robotiq 2f-85 gripper robot used in the molmospaces data generation framework. The skin adds 29 SPAD-like depth cameras (simulating VL53L5CX multi-zone time-of-flight sensors) on arm links 2, 3, 5, and 6.

The robot model (`franka_droid_skin`) is a drop-in replacement for the standard `franka_droid` model. It uses the same `FrankaRobot` class, `FrankaDroidRobotView`, joint controllers, and gripper actuator. The skin bodies are passive (no joints, no actuators, no collision) so the existing control pipeline works unchanged.

### What the skin provides

- 4 skin patches (translucent blue overlays on links 2, 3, 5, 6)
- 29 depth cameras (8x8 pixels each, 45 deg FOV, 0.02-4.0m range)
- Per-step proximity measurements of nearby surfaces (arm self-body, objects, furniture)
- Visible in RGB renders (can be used for visual policy training with skin awareness)

### Model statistics

| Property | Value |
|----------|-------|
| Degrees of freedom (nq) | 13 (7 arm + 6 gripper linkage) |
| Actuators (nu) | 8 (7 arm position + 1 gripper tendon) |
| Cameras (ncam) | 31 (2 wrist + 29 proximity) |
| Bodies (nbody) | 55 |
| Geoms (ngeom) | 68 |

---

## Architecture

```
franka_droid_skin/model.xml
  |
  ├── FR3 arm (links 0-7, 7 revolute joints)
  │     ├── link2 → link2_skin → 7 sensor cameras
  │     ├── link3 → link3_skin → 8 sensor cameras
  │     ├── link5 → link5_skin → 6 sensor cameras
  │     └── link6 → link6_skin → 8 sensor cameras
  │
  ├── Robotiq 2f-85 gripper (via <attach>, prefix "gripper/")
  │     ├── left finger chain (driver → coupler → follower → pad)
  │     ├── right finger chain
  │     └── tendon-coupled actuator (fingers_actuator)
  │
  └── 2 wrist cameras (wrist_cam, gripper/wrist_camera)
```

Each sensor camera is a fixed MuJoCo camera inside a body that is rigidly attached to the skin mesh. The camera's -Z axis points outward (away from the arm surface). When rendered in depth mode at 8x8 resolution, each camera produces a proximity depth map equivalent to a VL53L5CX SPAD array.

---

## Installation

### Prerequisites

1. **Conda environment**: `mlspaces` (for physics/model loading)
2. **Pi0 venv**: `MolmoBot/MolmoBot-Pi0/.venv` (for rendering, since mlspaces has a broken mujoco-filament)
3. **molmospaces**: Installed in mlspaces conda env
4. **MLSPACES_ASSETS_DIR**: Must point to the `resources/` directory

### Step 1: Verify environment variables

These should already be in `~/.bashrc`:

```bash
export MLSPACES_ASSETS_DIR=/home/jaydv/code/molmo/resources
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
```

Reload if you just added them:

```bash
source ~/.bashrc
```

### Step 2: Verify molmospaces is installed

```bash
conda activate mlspaces
python -c "from molmo_spaces.configs.robot_configs import FrankaSkinRobotConfig; print('OK')"
```

If this fails with an import error for `FrankaSkinRobotConfig`, the config hasn't been registered yet. Check that `molmospaces/molmo_spaces/configs/robot_configs.py` contains the `FrankaSkinRobotConfig` class. If molmospaces is installed as a non-editable package, you need to reinstall:

```bash
cd /home/jaydv/code/molmo/molmospaces
pip install -e .
```

### Step 3: Verify franka_droid base model exists

The `franka_droid_skin` model depends on `franka_droid` (via symlinks). The base model is auto-downloaded by molmospaces' resource manager on first use. Force it:

```bash
conda activate mlspaces
python -c "
from molmo_spaces.molmo_spaces_constants import get_robot_path
p = get_robot_path('franka_droid') / 'model.xml'
print(f'{p} exists: {p.exists()}')
"
```

If it says `False`, trigger the resource manager:

```bash
cd /home/jaydv/code/molmo/molmospaces
PYTHONPATH=. python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"
```

### Step 4: Verify franka_droid_skin model exists

```bash
ls -la /home/jaydv/code/molmo/resources/robots/franka_droid_skin/
```

Expected output:
```
model.xml           -- Main MJCF
skin_meshes/        -- 4 STL files (actual, not symlinks)
assets/             -- symlink -> ../franka_droid/assets/
meshes/             -- symlink -> ../franka_droid/meshes/
robotiq_2f85_v4/    -- symlink -> ../franka_droid/robotiq_2f85_v4/
README.md
```

If the directory is missing, it was not set up. Recreate it:

```bash
mkdir -p /home/jaydv/code/molmo/resources/robots/franka_droid_skin/skin_meshes

# Copy skin mesh STLs
cp ~/code/proximity_learning/assets/.cache/gentact_ros_tools/meshes/skin/self-cap/link{2,3,5,6}_fancy.stl \
   /home/jaydv/code/molmo/resources/robots/franka_droid_skin/skin_meshes/

# Create symlinks to franka_droid assets
cd /home/jaydv/code/molmo/resources/robots/franka_droid_skin
ln -sfn ../franka_droid/assets assets
ln -sfn ../franka_droid/meshes meshes
ln -sfn ../franka_droid/robotiq_2f85_v4 robotiq_2f85_v4
```

Then regenerate model.xml (see [Development Log](#development-log) for the generation script).

### Step 5: Verify model loads

```bash
conda activate mlspaces
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml')
print(f'nq={m.nq} nu={m.nu} ncam={m.ncam} nbody={m.nbody}')
# Expected: nq=13 nu=8 ncam=31 nbody=55
"
```

### Step 6: Verify rendering works (Pi0 venv)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jaydv/code/molmo/MolmoBot/MolmoBot-Pi0/.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
print(f'Render OK: {img.shape} mean={img.mean():.1f}')
"
```

---

## Directory Structure

```
molmo/
├── molmospaces/                              # MolmoSpaces framework
│   └── molmo_spaces/
│       ├── configs/
│       │   ├── robot_configs.py              # FrankaSkinRobotConfig lives here
│       │   └── camera_configs.py             # Camera system definitions
│       ├── data_generation/
│       │   ├── main.py                       # CLI entry: python -m molmo_spaces.data_generation.main
│       │   ├── config_registry.py            # @register_config decorator
│       │   └── config/
│       │       └── object_manipulation_datagen_configs.py  # FrankaPickOmniCamConfig etc.
│       ├── env/
│       │   ├── env.py                        # CPUMujocoEnv
│       │   ├── sensors.py                    # get_core_sensors()
│       │   ├── sensors_cameras.py            # CameraSensor, DepthSensor
│       │   └── abstract_sensors.py           # SensorSuite
│       ├── robots/
│       │   └── franka.py                     # FrankaRobot.add_robot_to_scene()
│       ├── tasks/
│       │   └── task_sampler.py               # Scene + robot composition (line 596)
│       └── utils/
│           └── save_utils.py                 # HDF5 trajectory saving
│
├── resources/                                # MLSPACES_ASSETS_DIR
│   ├── robots/
│   │   ├── franka_droid/                     # Base model (auto-downloaded)
│   │   │   ├── model.xml                     # symlink from ~/.cache/molmo-spaces-resources/
│   │   │   ├── assets/                       # OBJ/STL mesh files
│   │   │   └── robotiq_2f85_v4/2f85.xml     # Gripper submodel
│   │   └── franka_droid_skin/                # Skin-augmented model (manual)
│   │       ├── model.xml                     # Modified MJCF with skin bodies
│   │       ├── skin_meshes/                  # link{2,3,5,6}_fancy.stl
│   │       ├── assets/                       # symlink -> ../franka_droid/assets/
│   │       ├── meshes/                       # symlink -> ../franka_droid/meshes/
│   │       └── robotiq_2f85_v4/              # symlink -> ../franka_droid/robotiq_2f85_v4/
│   └── scenes/
│       └── ithor/                            # Kitchen/room scenes (auto-downloaded)
│           ├── FloorPlan14_physics.xml
│           └── ...
│
├── fr3_skin_mujoco/                          # Scratch/development area
│   ├── WORK_LOG.md                           # This file
│   ├── robotiq.urdf                          # Source URDF with skin sensor poses
│   ├── fr3_skin_mujoco.py                    # URDF-to-MJCF converter (legacy)
│   ├── franka_skin_in_kitchen.png            # Rendered verification image
│   ├── franka_droid_skin_scene.png           # Standalone render
│   └── franka_droid_skin_depths.png          # Per-sensor depth grid
│
└── MolmoBot/
    └── MolmoBot-Pi0/
        └── .venv/                            # Rendering venv (vanilla mujoco 3.4.0)
```

---

## Usage: Standalone Rendering

### Render the robot in a kitchen scene

```python
#!/usr/bin/env python3
"""Render franka_droid_skin in an iTHOR kitchen scene."""
import mujoco
from mujoco import MjSpec
import numpy as np
from pathlib import Path
import sys

# Add molmospaces to path
sys.path.insert(0, '/home/jaydv/code/molmo/molmospaces')
from molmo_spaces.configs.robot_configs import FrankaSkinRobotConfig
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.franka import FrankaRobot

# Load scene and robot
scene_spec = MjSpec.from_file(str(Path(
    "/home/jaydv/code/molmo/resources/scenes/ithor/FloorPlan14_physics.xml"
)))
cfg = FrankaSkinRobotConfig()
robot_spec = MjSpec.from_file(str(get_robot_path(cfg.name) / cfg.robot_xml_path))

# Compose scene + robot (same as datagen pipeline)
FrankaRobot.add_robot_to_scene(
    robot_config=cfg, spec=scene_spec, robot_spec=robot_spec,
    prefix="robot_0/", pos=[0, -0.15, 0], quat=[1, 0, 0, 1],
    randomize_textures=False,
)
FrankaRobot.apply_control_overrides(scene_spec, cfg)
scene_spec.visual.global_.offwidth = 1280
scene_spec.visual.global_.offheight = 960

# Compile and set home pose
model = scene_spec.compile()
data = mujoco.MjData(model)
home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
for i, val in enumerate(home):
    name = f'robot_0/fr3_joint{i+1}'
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = val
mujoco.mj_forward(model, data)

# Render
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 2.0
cam.azimuth = -135
cam.elevation = -30
cam.lookat[:] = [0.0, -0.15, 0.6]

renderer = mujoco.Renderer(model, 960, 1280)
# IMPORTANT: the skin geoms live in geom_group=3. The default MjvOption
# renders groups 0..2 only, so without this the model renders without
# skin. Enable groups 0..3 (group 4 is iTHOR collision, keep it off).
scene_option = mujoco.MjvOption()
mujoco.mjv_defaultOption(scene_option)
scene_option.geomgroup[:] = [1, 1, 1, 1, 0, 0]
renderer.update_scene(data, camera=cam, scene_option=scene_option)
img = renderer.render()

import matplotlib.pyplot as plt
plt.imsave("franka_skin_kitchen.png", img)
print(f"Saved: {img.shape}")
```

Run with the Pi0 venv:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jaydv/code/molmo/MolmoBot/MolmoBot-Pi0/.venv/bin/python render_kitchen.py
```

### Render proximity sensor depth

```python
"""Render depth from all 29 proximity sensors in a composed scene."""
# ... (after composing scene+robot and calling mj_forward as above)

renderer_8x8 = mujoco.Renderer(model, 8, 8)
renderer_8x8.enable_depth_rendering()

for cam_id in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    if not name or 'sensor' not in name:
        continue
    renderer_8x8.update_scene(data, camera=name)
    depth = renderer_8x8.render().copy()  # (8, 8) float32, metres
    depth[(depth < 0.02) | (depth > 4.0)] = np.nan  # clip to SPAD range
    finite = np.isfinite(depth)
    if finite.any():
        print(f"{name}: {depth[finite].min():.3f}-{depth[finite].max():.3f}m "
              f"({finite.sum()}/64 valid pixels)")
```

---

## Usage: MolmoSpaces Data Collection

The standard molmospaces data generation pipeline can use `franka_droid_skin` with minimal config changes. There are two approaches:

### Approach 1: Use run_pipeline.py (quick, interactive)

The `run_pipeline.py` script supports dynamic robot selection. However, `franka_droid_skin` is not in its `--robot` choices by default. You can use `--config` with a registered config name instead.

First, register a config (see [Creating a Custom Datagen Config](#creating-a-custom-datagen-config) below), then:

```bash
cd /home/jaydv/code/molmo/molmospaces
MUJOCO_GL=egl PYTHONPATH=. python scripts/datagen/run_pipeline.py \
    --config FrankaSkinPickConfig \
    --seed 42
```

For interactive viewing (requires display):

```bash
PYTHONPATH=. python scripts/datagen/run_pipeline.py \
    --config FrankaSkinPickConfig \
    --viewer --seed 3
```

### Approach 2: Use main.py (production datagen)

```bash
cd /home/jaydv/code/molmo/molmospaces
MUJOCO_GL=egl PYTHONPATH=. python -m molmo_spaces.data_generation.main FrankaSkinPickConfig
```

This runs the full parallel pipeline with HDF5 output.

### Approach 3: Swap robot in existing config at runtime

If you don't want to register a new config, you can override the robot config on an existing config class:

```python
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import FrankaPickOmniCamConfig
from molmo_spaces.configs.robot_configs import FrankaSkinRobotConfig

config = FrankaPickOmniCamConfig()
# Override the robot config to use skin model
config = config.model_copy(update={"robot_config": FrankaSkinRobotConfig()})
```

---

## Reading Proximity Sensor Data

### Camera naming convention

After scene composition with `prefix="robot_0/"`, proximity cameras are named:

```
robot_0/link2_sensor_0  through  robot_0/link2_sensor_6   (7 cameras)
robot_0/link3_sensor_0  through  robot_0/link3_sensor_7   (8 cameras)
robot_0/link5_sensor_0  through  robot_0/link5_sensor_5   (6 cameras)
robot_0/link6_sensor_0  through  robot_0/link6_sensor_7   (8 cameras)
```

Total: 29 cameras, each producing an 8x8 float32 depth image in metres.

### Per-step data collection during rollouts

The proximity sensors are MuJoCo cameras. To read them during a rollout, you render depth from each sensor camera after every `mj_step` / `mj_forward`:

```python
import mujoco
import numpy as np

def collect_proximity_data(model, data, renderer_8x8):
    """Collect depth from all 29 proximity sensors. Returns dict of (8,8) arrays."""
    renderer_8x8.enable_depth_rendering()
    readings = {}
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if not name or 'sensor' not in name:
            continue
        renderer_8x8.update_scene(data, camera=name)
        depth = renderer_8x8.render().copy()
        # Clip to physical SPAD range
        depth[(depth < 0.02) | (depth > 4.0)] = 0.0  # 0 = no return
        readings[name] = depth
    return readings  # dict of 29 x (8,8) float32 arrays
```

### Flattening for model input

For a policy network, you might want a flat vector:

```python
def proximity_to_vector(readings):
    """Flatten all 29 sensor readings into a single vector."""
    # Sort by name for consistent ordering
    ordered = [readings[k] for k in sorted(readings.keys())]
    return np.concatenate([d.flatten() for d in ordered])  # shape: (29*64,) = (1856,)
```

### Saving to HDF5 alongside trajectory data

The molmospaces `save_utils.py` saves sensor data to HDF5 automatically for sensors registered in the SensorSuite. For proximity data not yet in the SensorSuite, you can save manually:

```python
import h5py

def save_proximity_to_hdf5(hdf5_path, episode_idx, proximity_trajectory):
    """
    proximity_trajectory: list of dicts, one per timestep.
    Each dict maps sensor_name -> (8,8) float32 array.
    """
    with h5py.File(hdf5_path, "a") as f:
        grp = f.require_group(f"traj_{episode_idx}/obs/proximity")
        # Stack into (T, 8, 8) per sensor
        sensor_names = sorted(proximity_trajectory[0].keys())
        for name in sensor_names:
            data = np.stack([step[name] for step in proximity_trajectory])  # (T, 8, 8)
            short_name = name.replace("robot_0/", "")
            grp.create_dataset(short_name, data=data, compression="gzip")
```

---

## Creating a Custom Datagen Config

To run data generation with the skin robot, register a config in molmospaces:

### Option A: Add to existing config file

Edit `molmospaces/molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py`:

```python
from molmo_spaces.configs.robot_configs import FrankaSkinRobotConfig

@register_config("FrankaSkinPickConfig")
class FrankaSkinPickConfig(PickBaseConfig):
    """Pick task with skin-augmented Franka."""
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "skin_pick_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_datagen"
```

### Option B: Create a new config file

Create `molmospaces/molmo_spaces/data_generation/config/skin_datagen_configs.py`:

```python
"""Data generation configs for Franka with proximity skin."""
from pathlib import Path

from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    FrankaOmniPurposeCameraSystem,
)
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaSkinRobotConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR


@register_config("FrankaSkinPickConfig")
class FrankaSkinPickConfig(PickBaseConfig):
    """Pick task with skin-augmented Franka + Robotiq."""
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "skin_pick_v1"

    @property
    def tag(self) -> str:
        return "franka_skin_pick_datagen"


@register_config("FrankaSkinPickAndPlaceConfig")
class FrankaSkinPickAndPlaceConfig(PickBaseConfig):
    """Pick-and-place task with skin-augmented Franka + Robotiq."""
    robot_config: BaseRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "skin_pnp_v1"
    task_type: str = "pick_and_place"

    @property
    def tag(self) -> str:
        return "franka_skin_pnp_datagen"
```

The file is auto-discovered by `main.py` (it imports everything in `molmo_spaces/data_generation/config/`).

### Running

```bash
cd /home/jaydv/code/molmo/molmospaces
MUJOCO_GL=egl PYTHONPATH=. python -m molmo_spaces.data_generation.main FrankaSkinPickConfig
```

---

## Sensor Specifications

### Proximity camera layout

| Link | Sensor count | Camera names | Skin mesh |
|------|-------------|-------------|-----------|
| fr3_link2 | 7 | link2_sensor_0 ... link2_sensor_6 | link2_fancy.stl |
| fr3_link3 | 8 | link3_sensor_0 ... link3_sensor_7 | link3_fancy.stl |
| fr3_link5 | 6 | link5_sensor_0 ... link5_sensor_5 | link5_fancy.stl |
| fr3_link6 | 8 | link6_sensor_0 ... link6_sensor_7 | link6_fancy.stl |

### Camera parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 8 x 8 pixels |
| Field of view | 45 degrees |
| Minimum range | 0.02 m (20 mm) |
| Maximum range | 4.0 m |
| Output type | float32 depth in metres |
| Orientation | Camera -Z = outward from arm surface |
| Mode | fixed (attached to sensor body frame) |

### Physical analogue

These cameras simulate a **VL53L5CX** multi-zone SPAD time-of-flight sensor:
- 8x8 zone array
- 2-400 cm range
- 45 degree FOV
- ~15 Hz update rate (in real hardware; in simulation, as fast as you render)

### Skin visual properties

| Property | Value |
|----------|-------|
| RGBA | 0.25 0.55 0.85 0.35 (translucent blue) |
| Collision | Disabled (contype=0, conaffinity=0) |
| Mesh format | STL |

---

## Known Issues and Workarounds

### 1. mujoco-filament rendering crash in mlspaces

**Problem**: The `mlspaces` conda env has `mujoco-filament 3.5.1` which overwrites the vanilla `mujoco 3.4.0` `libmujoco.so`. All rendering (`MjrContext` creation) fails with:
```
mujoco.FatalError: Error opening file 'filament:pbr.filamat'
```

**Impact**: Cannot render in mlspaces env. Model loading and physics work fine.

**Workaround**: Use the Pi0 venv for rendering:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jaydv/code/molmo/MolmoBot/MolmoBot-Pi0/.venv/bin/python your_script.py
```

**Permanent fix** (if you want rendering in mlspaces):
```bash
conda activate mlspaces
pip uninstall mujoco-filament
pip install mujoco==3.4.0
```

Warning: this may break other molmospaces features that depend on the filament renderer. The production datagen pipeline (on remote machines) uses filament.

### 2. franka_droid_skin is not auto-downloaded

Unlike `franka_droid`, the skin model is not in molmospaces' resource manager. The `skin_meshes/` directory and `model.xml` must be present at `resources/robots/franka_droid_skin/`. If the directory is missing after a fresh clone, recreate it per the installation instructions.

### 3. Offscreen buffer size

The skin model's `<visual><global offwidth="1280" offheight="960"/>` sets the max render resolution when used standalone. When composed into a scene, the scene's visual settings take precedence. If you get `ValueError: Image height > framebuffer height`, bump the offscreen buffer before compiling:

```python
scene_spec.visual.global_.offwidth = 1280
scene_spec.visual.global_.offheight = 960
model = scene_spec.compile()
```

### 4. Proximity sensors not in SensorSuite (yet)

The 29 proximity cameras are not automatically registered in molmospaces' `SensorSuite` / `get_core_sensors()`. They exist as MuJoCo cameras in the compiled model but the datagen pipeline doesn't render them by default. To collect proximity data during datagen, you need either:

- **Option A**: Add custom depth sensors to `get_core_sensors()` for each proximity camera (proper integration, requires molmospaces code changes)
- **Option B**: Post-hoc render from saved qpos trajectories (no pipeline changes needed)
- **Option C**: Hook into the rollout loop and render proximity depth manually (see [Reading Proximity Sensor Data](#reading-proximity-sensor-data))

Option B is recommended for initial experiments:

```python
"""Post-hoc proximity data extraction from saved HDF5 trajectories."""
import h5py
import mujoco
import numpy as np

def extract_proximity_from_trajectory(model_xml_path, hdf5_path, traj_idx):
    """Read qpos from a saved trajectory and render proximity depth at each step."""
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 8, 8)
    renderer.enable_depth_rendering()

    # Get sensor camera IDs
    sensor_cams = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and 'sensor' in name:
            sensor_cams.append((i, name))

    with h5py.File(hdf5_path, "r") as f:
        qpos_data = f[f"traj_{traj_idx}/obs/agent/qpos"][:]  # (T, nq)

    all_readings = []
    for t in range(len(qpos_data)):
        data.qpos[:] = qpos_data[t]
        mujoco.mj_forward(model, data)

        step_readings = {}
        for cam_id, cam_name in sensor_cams:
            renderer.update_scene(data, camera=cam_name)
            depth = renderer.render().copy()
            depth[(depth < 0.02) | (depth > 4.0)] = 0.0
            step_readings[cam_name] = depth
        all_readings.append(step_readings)

    return all_readings  # list of T dicts, each with 29 x (8,8) arrays
```

---

## Development Log

### Session 1: Initial URDF Conversion (pre-existing)
- `fr3_skin_mujoco.py` converted `fr3_full_skin.urdf` to `fr3_skin.xml` (MJCF)
- Key bugs fixed: hardcoded `state_dim`, `qpos_dim` mismatch, dataset format mismatch
- See `memory/act_pipeline.md` for details

### Session 2: Robotiq URDF + Model Integration (2026-04-12)

1. **Attempted URDF conversion of `robotiq.urdf`** -- Fixed mesh path stripping, mimic joints, degenerate inertials, keyframe qpos padding. Produced `fr3_robotiq.xml` but approach was wrong (should use existing molmospaces MJCF).

2. **Verified mesh compatibility** -- Compared bounding boxes of `franka_droid` vs `gentact` link meshes. All 8 links match within 6mm. Skin fits.

3. **Built `franka_droid_skin/model.xml`** -- Generated programmatically by parsing `franka_droid/model.xml` and injecting 29 sensor bodies from `robotiq.urdf` pose data (rpy-to-quaternion conversion via scipy).

4. **Rendering verification** -- Confirmed in Pi0 venv: scene render + 29 depth grids + in-scene kitchen render. All sensors producing valid depth readings.

5. **molmospaces integration** -- Added `FrankaSkinRobotConfig` to `robot_configs.py`. Verified config instantiation, model loading, joint/actuator name matching.

6. **In-scene verification** -- Loaded skin robot into FloorPlan14 (iTHOR kitchen) via `MjSpec` composition. All 29 sensors detecting kitchen geometry at 0.02-3.7m ranges.

### How model.xml was generated

The model was NOT hand-written. A Python script:
1. Parsed `franka_droid/model.xml` with `xml.etree.ElementTree`
2. Parsed `fr3_skin_mujoco/robotiq.urdf` for all 29 sensor joint origins (rpy, xyz)
3. Converted each sensor's URDF rpy to MuJoCo quaternion via `scipy.spatial.transform.Rotation`
4. For each sensor camera, computed a 180-degree X-flip quaternion so camera -Z points outward
5. Injected skin bodies as children of `fr3_link{2,3,5,6}` in the parsed XML tree
6. Added skin mesh assets, default class, visual settings, and home keyframe
7. Wrote the result with `ET.indent()` for readability

The generation script was run inline (not saved as a standalone file). To regenerate, re-run the model generation from the conversation history or adapt `fr3_skin_mujoco.py`.

---

## File Reference

| File | Purpose |
|------|---------|
| `resources/robots/franka_droid_skin/model.xml` | Production MJCF model |
| `resources/robots/franka_droid_skin/skin_meshes/*.stl` | Skin mesh geometry (4 files) |
| `resources/robots/franka_droid_skin/README.md` | Quick-reference README |
| `molmospaces/molmo_spaces/configs/robot_configs.py` | `FrankaSkinRobotConfig` class |
| `fr3_skin_mujoco/WORK_LOG.md` | This document |
| `fr3_skin_mujoco/robotiq.urdf` | Source URDF with skin sensor poses |
| `fr3_skin_mujoco/fr3_skin_mujoco.py` | Legacy URDF-to-MJCF converter |
| `fr3_skin_mujoco/franka_skin_in_kitchen.png` | Verification: robot in kitchen |
| `fr3_skin_mujoco/franka_droid_skin_scene.png` | Verification: standalone robot |
| `fr3_skin_mujoco/franka_droid_skin_depths.png` | Verification: sensor depth grid |

### Source mesh origins

| Mesh | Source path |
|------|------------|
| Skin STLs | `~/code/proximity_learning/assets/.cache/gentact_ros_tools/meshes/skin/self-cap/` |
| FR3 arm meshes | `resources/robots/franka_droid/assets/` (auto-downloaded) |
| Robotiq gripper meshes | `resources/robots/franka_droid/robotiq_2f85_v4/assets/` (auto-downloaded) |
