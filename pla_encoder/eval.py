"""Eval entry point: load a PLA / VLM-only ACT checkpoint and evaluate on
`FrankaPickandPlaceHardBench` (procthor-objaverse val split).

Implementation route: thin wrapper around `molmo_spaces.evaluation.run_evaluation`.
Configs are defined at MODULE level (not inside a function) so they're picklable
by the upstream `runner.run()` which serializes the config to disk. Runtime CLI
overrides land in `_EVAL_OVERRIDES` and are applied in `model_post_init`.

Per ../TODO.md §6:
  - 200 episodes
  - procthor-objaverse val split
  - report success rate + 95% CI
  - write `eval_output/{run_name}/results.json`

Usage:
  python -m pla.eval --checkpoint runs/pla_v1/latest.pt \
                     --run_name pla_v1 --max_episodes 200
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaSkinRobotConfig
from molmo_spaces.configs.camera_configs import FrankaSkinCameraSystem
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig

from pla.eval_policy import PLAInferencePolicy


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "eval_output"
DEFAULT_BENCHMARK = (
    Path.home()
    / ".cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415"
    / "procthor-objaverse/FrankaPickandPlaceHardBench"
    / "FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark"
)


# Populated in main() before run_evaluation instantiates the config class.
# model_post_init on PLABenchmarkEvalConfig copies these onto the live instance.
_EVAL_OVERRIDES: dict[str, Any] = {}


class PLAPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = ""
    use_proximity: bool = True
    image_h: int = 224
    image_w: int = 320
    depth_max_m: float = 4.0
    gripper_schedule: str = "binary_t"
    gripper_close_step: int = 120
    gripper_threshold: float = 0.5
    policy_cls: type = PLAInferencePolicy
    policy_type: str = "learned"


class PLABenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaSkinRobotConfig = FrankaSkinRobotConfig()
    camera_config: FrankaSkinCameraSystem = FrankaSkinCameraSystem()
    policy_config: PLAPolicyConfig = PLAPolicyConfig()
    policy_dt_ms: float = 66.0
    terminate_upon_success: bool = True
    task_horizon: int = 500

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        for key, value in _EVAL_OVERRIDES.items():
            if key == "task_horizon":
                self.task_horizon = value
            else:
                setattr(self.policy_config, key, value)


def parse_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y", "t")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint produced by pla.train.")
    p.add_argument("--benchmark_dir", type=str, default=str(DEFAULT_BENCHMARK),
                   help=f"Directory containing benchmark.json. Default: {DEFAULT_BENCHMARK}")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--use_proximity", type=parse_bool, default=True,
                   help="Override the checkpoint's use_proximity flag (must match train-time).")
    p.add_argument("--max_episodes", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--task_horizon_steps", type=int, default=500)
    p.add_argument("--gripper_schedule", type=str, default="binary_t",
                   choices=("open", "qpos", "binary_t"))
    p.add_argument("--gripper_close_step", type=int, default=120)
    p.add_argument("--image_h", type=int, default=224)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--depth_max_m", type=float, default=4.0)
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi)."""
    if total == 0:
        return (0.0, 0.0)
    p = success / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def main() -> None:
    args = get_args()
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] writing results to {out_dir}")

    # Stash CLI overrides for model_post_init to apply at instantiation.
    _EVAL_OVERRIDES.clear()
    _EVAL_OVERRIDES.update({
        "use_proximity": args.use_proximity,
        "image_h": args.image_h,
        "image_w": args.image_w,
        "depth_max_m": args.depth_max_m,
        "gripper_schedule": args.gripper_schedule,
        "gripper_close_step": args.gripper_close_step,
        "task_horizon": args.task_horizon_steps,
    })

    from molmo_spaces.evaluation import run_evaluation
    # JsonBenchmarkEvalConfig sets `camera_config: None = None` and lets each
    # benchmark episode declare its own cameras (typically DROID-style randomized
    # ones, e.g. wrist_camera_zed_mini). Our policy was trained on
    # FrankaSkinCameraSystem (exo_camera_1 + wrist_camera), so override the
    # benchmark cameras to match training. Departs from the canonical bench but
    # is the closest honest eval we can run without retraining on DROID cameras.
    results = run_evaluation(
        eval_config_cls=PLABenchmarkEvalConfig,
        benchmark_dir=Path(args.benchmark_dir),
        checkpoint_path=args.checkpoint,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        output_dir=out_dir,
        task_horizon_steps=args.task_horizon_steps,
        camera_config_override=FrankaSkinCameraSystem(),
    )

    success = int(getattr(results, "success_count", 0))
    total = int(getattr(results, "total_count", 0))
    sr = (success / total) if total else 0.0
    lo, hi = wilson_ci(success, total)
    summary = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "benchmark_dir": args.benchmark_dir,
        "use_proximity": args.use_proximity,
        "n_episodes": total,
        "n_success": success,
        "success_rate": sr,
        "wilson_95_ci": [lo, hi],
        "task_horizon_steps": args.task_horizon_steps,
        "gripper_schedule": args.gripper_schedule,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] success {success}/{total} = {sr:.3f}  "
          f"(95% CI [{lo:.3f}, {hi:.3f}])  -> {out_path}")


if __name__ == "__main__":
    main()
