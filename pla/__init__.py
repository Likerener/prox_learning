"""PLA — Peripersonal Language-Action policies via whole-body ToF proximity sensing.

Subpackages:
  data    — trajectory collection, HDF5 schema, dataset stats, torch Dataset wrappers
  sim     — MJCF/URDF generation, sensor-orientation fixes, ToF rendering
  models  — proximity encoder, full PLA, baselines (VLM-only ACT, prop-only MLP), CVAE
  train   — training entry points + losses
  eval    — eval runner, task definitions, bootstrap CI, failure analysis
  checks  — pre-training sanity checks (depth reconstruction, MJCF replay, gradients)
  viz     — composite videos, point clouds, ToF heatmaps, sensor importance maps
"""

__version__ = "0.1.0"
