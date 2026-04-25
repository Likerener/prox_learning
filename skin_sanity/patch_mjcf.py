"""Patch the datagen MJCF (`resources/robots/franka_droid_skin/model.xml`) to
use the fixed sensor orientations from `fr3_full_skin_fixed.urdf`.

The MJCF has one `<body name="linkN_sensor_K" pos="..." quat="w x y z">` per
patch, whose pos/quat are the same values as the URDF joint's xyz/rpy (the
MJCF was auto-generated from the URDF). We only need to rewrite those two
attributes on each sensor body.

We do NOT touch the inner `<camera quat="0 1 0 0" fovy="45" .../>` — that's
the camera-to-body orientation and stays identical.

Output:
  resources/robots/franka_droid_skin/model_fixed.xml
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

URDF_FIXED = Path('/home/jaydv/code/molmo/fr3_skin_mujoco/fr3_full_skin_fixed.urdf')
MJCF_IN = Path('/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml')
MJCF_OUT = Path('/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model_fixed.xml')


def rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    x, y, z, w = R.from_euler('xyz', rpy).as_quat()
    return np.array([w, x, y, z])


def load_urdf_sensor_poses(urdf: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tree = ET.parse(str(urdf))
    root = tree.getroot()
    out = {}
    for j in root.findall('joint'):
        n = j.get('name', '')
        m = re.match(r'link(\d+)_sensor_(\d+)_joint$', n)
        if not m: continue
        sensor_name = n.replace('_joint', '')
        o = j.find('origin')
        xyz = np.array([float(x) for x in o.get('xyz').split()])
        rpy = np.array([float(x) for x in o.get('rpy').split()])
        out[sensor_name] = (xyz, rpy)
    return out


def main():
    poses = load_urdf_sensor_poses(URDF_FIXED)
    print(f'loaded {len(poses)} fixed sensor poses')
    text = MJCF_IN.read_text()

    patched = 0
    for name, (xyz, rpy) in poses.items():
        quat = rpy_to_quat_wxyz(rpy)
        xyz_s = ' '.join(f'{v:.6f}' for v in xyz)
        quat_s = ' '.join(f'{v:.8f}' for v in quat)
        # Match the <body name="link?_sensor_?" pos="..." quat="...">
        pat = rf'(<body\s+name="{re.escape(name)}")\s+pos="[^"]*"\s+quat="[^"]*"'
        repl = rf'\1 pos="{xyz_s}" quat="{quat_s}"'
        new_text, nsub = re.subn(pat, repl, text, count=1)
        if nsub != 1:
            raise RuntimeError(f'did not find body "{name}" in MJCF')
        text = new_text
        patched += 1

    MJCF_OUT.write_text(text)
    print(f'patched {patched}/29 sensor bodies -> {MJCF_OUT}')


if __name__ == '__main__':
    main()
