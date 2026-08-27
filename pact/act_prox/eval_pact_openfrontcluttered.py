"""Closed-loop eval of PACT in the open-front cluttered env.

Reuses the policy, prox-encoder plumbing and the degrade_vision / mask_proximity
machinery from eval_act_with_prox_encoder.py; swaps only the env config.
Architecture dims are read back off the checkpoint. Env seed is fixed and
num_workers=1, so runs are paired across policies and conditions.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ckpt_dims  # noqa: E402
import pact.act_prox.eval_act_with_prox_encoder as base  # noqa: E402

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner  # noqa: E402
from utils import set_seed  # noqa: E402

DEFAULT_HOUSES = "12,13,14,15,16,17,23,25"


class PACTOpenFrontClutteredEvalConfig(FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig):
    policy_config: base.ACTProxEncoderPolicyConfig = base.ACTProxEncoderPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_encoder_ckpt", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--house_inds", default=DEFAULT_HOUSES)
    p.add_argument("--samples_per_house", type=int, default=1)
    p.add_argument("--no_videos", action="store_true")
    p.add_argument("--degrade_vision", default="none",
                   choices=("none", "blackout", "dim", "noise", "wrist_off", "exo_off"))
    p.add_argument("--mask_proximity", default="none",
                   choices=("none", "zero", "mean", "noise", "shuffle"))
    p.add_argument("--mask_phase", default="none",
                   choices=("none", "approach", "pregrasp", "grasp_lift", "transit", "place"))
    p.add_argument("--prox_mean_path", default="")
    p.add_argument("--phase_log_path", default="")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-prox-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    dims = ckpt_dims.infer(str(ckpt_dir / args.ckpt_name))
    print(f"[act-prox-eval] dims from checkpoint: {dims}")
    if dims["n_proximity_sensors"] == 0:
        raise SystemExit(f"{ckpt_dir} has no proximity tokens — use the baseline eval script.")

    eval_cfg = PACTOpenFrontClutteredEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.no_videos:
        eval_cfg.save_videos = False

    sampler = eval_cfg.task_sampler_config
    sampler.house_inds = [int(h) for h in args.house_inds.split(",")]
    sampler.samples_per_house = args.samples_per_house

    set_seed(int(eval_cfg.seed) if eval_cfg.seed is not None else 7)

    pc = eval_cfg.policy_config
    pc.ckpt_dir = str(ckpt_dir)
    pc.ckpt_name = args.ckpt_name
    pc.prox_encoder_ckpt = str(Path(args.prox_encoder_ckpt).resolve())
    pc.prox_mapping_json = str(Path(args.prox_mapping_json).resolve())
    pc.image_h, pc.image_w = args.image_h, args.image_w
    pc.temp_agg_off, pc.temp_agg_m = args.temp_agg_off, args.temp_agg_m
    for key in ("chunk_size", "hidden_dim", "dim_feedforward", "enc_layers",
                "dec_layers", "state_dim", "action_dim", "prox_tokens_per_sensor"):
        setattr(pc, key, dims[key])
    pc.mask_proximity = args.mask_proximity
    pc.mask_phase = args.mask_phase
    pc.prox_mean_path = str(Path(args.prox_mean_path).resolve()) if args.prox_mean_path else ""
    pc.phase_log_path = str(Path(args.phase_log_path).resolve()) if args.phase_log_path else ""
    pc.degrade_vision = args.degrade_vision

    eval_cfg.save_config()
    print(f"[act-prox-eval] env=OpenFrontCluttered seed={eval_cfg.seed} "
          f"houses={sampler.house_inds} samples/house={sampler.samples_per_house} "
          f"degrade_vision={pc.degrade_vision} mask_proximity={pc.mask_proximity}")

    runner = ParallelRolloutRunner(eval_cfg)
    success, total = runner.run()
    print(f"[act-prox-eval] success {success}/{total}")


if __name__ == "__main__":
    main()
