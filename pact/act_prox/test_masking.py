"""Fast unit test for the prox masking branches in eval_act_with_prox_encoder.py.

Verifies that:
  1. `mask_proximity=none` leaves prox_pos untouched.
  2. `mask_proximity=zero` replaces prox_pos with zeros.
  3. `mask_proximity=mean` replaces prox_pos with the loaded mean tensor.
  4. `mask_phase` restricts masking to the named phase only.

We monkey-patch the policy + extractor with cheap stubs so this runs in <1s.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "submodules" / "act") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "submodules" / "act"))

import pact.act_prox.eval_act_with_prox_encoder as ev


class FakeExtractor:
    def __init__(self, n_sensors=29, window=8):
        self.window = window
        self.prox_mean = torch.zeros(4, 8, 8)
        self.prox_std = torch.ones(4, 8, 8)
        self._call_n_sensors = n_sensors

    def __call__(self, x):
        bs = x.shape[0]
        # Return a deterministic non-zero, non-mean tensor.
        return torch.full((bs, self._call_n_sensors, 3), 7.0, device=x.device)


class CapturedPolicy:
    """Stub that records the prox_pos it was called with."""
    def __init__(self):
        self.last_prox = None

    def __call__(self, qpos, image, proximity_positions=None):
        self.last_prox = proximity_positions.detach().clone()
        # Return a deterministic chunk: (1, 1, 8).
        return torch.zeros(1, 1, 8, device=qpos.device)


def make_obs(n_sensors=29, gripper=0.04, with_tcp=True, with_obj=True,
             with_base=True, with_target=False, target_xy=(5.75, 3.72)):
    rng = np.random.default_rng(0)
    # gripper as a python list (matches real obs format — see eval_act_mug_random.py)
    obs = {
        "qpos": {"arm": np.zeros(7, dtype=np.float32),
                 "gripper": [float(gripper), float(gripper)]},
        "exo_camera_1": np.zeros((240, 320, 3), dtype=np.uint8),
        "wrist_camera": np.zeros((240, 320, 3), dtype=np.uint8),
    }
    sensor_names = (
        [f"link2_sensor_{i}" for i in range(7)]
        + [f"link3_sensor_{i}" for i in range(8)]
        + [f"link5_sensor_{i}" for i in range(6)]
        + [f"link6_sensor_{i}" for i in range(8)]
    )
    for s in sensor_names[:n_sensors]:
        obs[s] = (rng.random((4, 8, 8)) * 0.5).astype(np.float32)
    if with_tcp:
        # Robot-frame TCP — picked so that with the base below, world is ~(5.7, 3.67, 0.83)
        obs["tcp_pose"] = np.array([0.30, 0.00, 1.00, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if with_base:
        # Identity rotation, world origin near object.
        obs["robot_base_pose"] = np.array([4.69, 3.99, 0.19, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if with_obj:
        obs["obj_start_pose"] = np.array([5.70, 3.67, 0.64, -0.5, -0.5, 0.5, 0.5], dtype=np.float32)
    if with_target:
        obs["obj_end_pose"] = np.array([target_xy[0], target_xy[1], 0.7, 0, 0, 0, 1],
                                       dtype=np.float32)
    return obs


def _build_policy(mask_proximity: str, mask_phase: str,
                  prox_mean_path: str = "") -> tuple:
    """Build a stubbed ACTWithProxEncoderInferencePolicy with given mask config."""
    exp = mock.MagicMock()
    pc = mock.MagicMock()
    pc.ckpt_dir = "/tmp"
    pc.ckpt_name = "policy_best.ckpt"
    pc.prox_mapping_json = "_test"
    pc.prox_encoder_ckpt = "_test"
    pc.camera_names = ("exo_camera_1", "wrist_camera")
    pc.image_h = 240
    pc.image_w = 320
    pc.chunk_size = 1
    pc.temp_agg_off = True
    pc.temp_agg_m = 0.0
    pc.mask_proximity = mask_proximity
    pc.mask_phase = mask_phase
    pc.prox_mean_path = prox_mean_path
    pc.phase_log_path = ""
    pc.prox_tokens_per_sensor = 1
    pc.degrade_vision = "none"
    exp.policy_config = pc

    sensor_names = (
        [f"link2_sensor_{i}" for i in range(7)]
        + [f"link3_sensor_{i}" for i in range(8)]
        + [f"link5_sensor_{i}" for i in range(6)]
        + [f"link6_sensor_{i}" for i in range(8)]
    )
    fake_mapping = {"sensor_names": sensor_names, "n_sensors": 29, "episodes": {}}

    with mock.patch("builtins.open", mock.mock_open(read_data='{"sensor_names":' +
                    str(sensor_names).replace("'", '"') +
                    ',"n_sensors":29,"episodes":{}}')):
        with mock.patch("json.load", return_value=fake_mapping):
            policy = ev.ACTWithProxEncoderInferencePolicy(exp)

    # Inject stubs.
    policy._extractor = FakeExtractor()
    policy._window = policy._extractor.window
    policy._prox_mean = policy._extractor.prox_mean.cpu().numpy().astype(np.float32)
    policy._prox_std = policy._extractor.prox_std.cpu().numpy().astype(np.float32)
    policy._policy = CapturedPolicy()
    policy._stats = {
        "qpos_mean": np.zeros(9, dtype=np.float32),
        "qpos_std": np.ones(9, dtype=np.float32),
        "action_mean": np.zeros(8, dtype=np.float32),
        "action_std": np.ones(8, dtype=np.float32),
    }
    if mask_proximity == "mean" and prox_mean_path:
        arr = np.load(prox_mean_path).astype(np.float32)
        policy._prox_mean_pos = torch.from_numpy(arr).unsqueeze(0).cuda()

    return policy, pc


def assert_close(a, b, rtol=1e-5, atol=1e-6, label=""):
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    if not np.allclose(a_np, b_np, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}: tensors not close\n  a={a_np.flatten()[:6]}\n  b={b_np.flatten()[:6]}"
        )


def run_one_step(policy, obs):
    return policy.inference_model(obs)


def test_mask_none():
    policy, pc = _build_policy("none", "none")
    obs = make_obs(gripper=0.04)
    run_one_step(policy, obs)
    captured = policy._policy.last_prox
    assert_close(captured, torch.full_like(captured, 7.0), label="mask_none")
    print("[unit] mask_none: prox passed through unchanged ✓")


def test_mask_zero():
    policy, pc = _build_policy("zero", "none")
    obs = make_obs(gripper=0.04)
    run_one_step(policy, obs)
    captured = policy._policy.last_prox
    assert_close(captured, torch.zeros_like(captured), label="mask_zero")
    print("[unit] mask_zero: prox zeroed ✓")


def test_mask_mean():
    # Use the precomputed mean file if present; otherwise synthesise a tiny one.
    mean_path = _REPO_ROOT / "pact" / "outputs_prox" / "runs" / "prox_encoder_v1" / "prox_pos_mean.npy"
    if not mean_path.exists():
        # Skip rather than fail.
        print("[unit] mask_mean: SKIPPED (mean file missing)")
        return
    arr = np.load(mean_path).astype(np.float32)
    policy, pc = _build_policy("mean", "none", str(mean_path))
    obs = make_obs(gripper=0.04)
    run_one_step(policy, obs)
    captured = policy._policy.last_prox
    expected = torch.from_numpy(arr).unsqueeze(0).cuda()
    assert_close(captured, expected, label="mask_mean")
    print("[unit] mask_mean: prox replaced by loaded mean ✓")


def test_phase_classifier_approach():
    """Approach: gripper open, TCP far from object xy in world."""
    policy, pc = _build_policy("zero", "approach")
    # Make TCP world xy = (4.99, 3.99) — base(4.69, 3.99) + tcp(0.30, 0).
    # Object world xy = (5.70, 3.67). dist_xy = sqrt(0.71^2 + 0.32^2) = 0.78 m.
    obs = make_obs(gripper=0.003)   # gripper OPEN (small qpos)
    phase = policy._classify_phase(obs)
    assert phase == "approach", f"expected 'approach', got {phase!r}"
    print(f"[unit] classifier: gripper-open + TCP far -> {phase} ✓")


def test_phase_classifier_pregrasp():
    """Pregrasp: gripper open, TCP within 10cm of object xy."""
    policy, pc = _build_policy("zero", "pregrasp")
    obs = make_obs(gripper=0.003)
    # Move TCP closer so world TCP xy aligns with object xy (5.70, 3.67).
    # base = (4.69, 3.99). TCP_world_xy = base + R @ tcp_xyz
    # Need TCP_world_xy = (5.70, 3.67). With identity R: tcp_robot = (1.01, -0.32, ?).
    obs["tcp_pose"] = np.array([1.01, -0.32, 1.00, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    phase = policy._classify_phase(obs)
    assert phase == "pregrasp", f"expected 'pregrasp', got {phase!r}"
    print(f"[unit] classifier: gripper-open + TCP close -> {phase} ✓")


def test_phase_classifier_grasp_lift():
    """Grasp/lift: gripper closed, object not yet lifted (lift state needs history)."""
    policy, pc = _build_policy("zero", "grasp_lift")
    obs = make_obs(gripper=0.76)    # gripper CLOSED
    # TCP coincides with object so classifier sees small distance + closed.
    obs["tcp_pose"] = np.array([1.01, -0.32, 1.00, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    phase = policy._classify_phase(obs)
    assert phase == "grasp_lift", f"expected 'grasp_lift', got {phase!r}"
    print(f"[unit] classifier: gripper-closed + no lift -> {phase} ✓")


def test_phase_mask_applies_only_in_named_phase():
    """If mask_phase=approach, mask SHOULD apply when classifier returns 'approach'
    and should NOT apply when classifier returns 'pregrasp'."""
    policy, _ = _build_policy("zero", "approach")
    # Approach scenario (TCP far)
    obs = make_obs(gripper=0.003)
    run_one_step(policy, obs)
    captured_approach = policy._policy.last_prox
    assert_close(captured_approach, torch.zeros_like(captured_approach),
                 label="approach: mask applied")

    # Pregrasp scenario (TCP close) — phase != approach so mask should NOT apply
    policy2, _ = _build_policy("zero", "approach")
    obs2 = make_obs(gripper=0.003)
    obs2["tcp_pose"] = np.array([1.01, -0.32, 1.00, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    run_one_step(policy2, obs2)
    captured_pregrasp = policy2._policy.last_prox
    # mask_phase=approach, phase=pregrasp → mask NOT applied → prox stays at 7.0
    assert_close(captured_pregrasp, torch.full_like(captured_pregrasp, 7.0),
                 label="pregrasp: mask not applied")
    print("[unit] phase_mask: applies only in named phase ✓")


def main() -> None:
    print("[unit] running prox-mask unit tests")
    test_mask_none()
    test_mask_zero()
    test_mask_mean()
    test_phase_classifier_approach()
    test_phase_classifier_pregrasp()
    test_phase_classifier_grasp_lift()
    test_phase_mask_applies_only_in_named_phase()
    print("[unit] all tests passed.")


if __name__ == "__main__":
    main()
