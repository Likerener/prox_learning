"""Training entry point for PLA / VLM-only ACT.

Usage:
  python -m pla.train --use_proximity true  --run_name pla_v1
  python -m pla.train --use_proximity false --run_name vlm_only_act

Per ../TODO.md §4:
  Loss: L1(actions) + 10 * KL
  Optimizer: Adam, lr=1e-5, batch_size=8
  WandB logging
  Checkpoint every 1000 steps
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pla.dataset import FrankaSkinHDF5Dataset, FrankaSkinDatasetConfig
from pla.policy import PLAPolicy, PLAConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "assets/datagen/pick_and_place_skin_pilot_v1/"
    / "FrankaSkinPickAndPlacePilotConfig/20260508_122042"
)


def parse_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y", "t")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--use_proximity", type=parse_bool, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT),
                   help="Path to a dataset run dir (contains house_*/...).")
    p.add_argument("--ckpt_dir", type=str,
                   default=str(REPO_ROOT / "runs"),
                   help="Parent dir; checkpoints land in <ckpt_dir>/<run_name>/.")
    p.add_argument("--num_steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    # num_workers=2 chosen for ~62 GB RAM box; each MuJoCo/h5 worker is ~6-7 GB RSS.
    # Increase only on a >128 GB machine.
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image_h", type=int, default=224)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--use_wandb", type=parse_bool, default=True)
    p.add_argument("--wandb_project", type=str, default="pla")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to resume from.")
    return p.parse_args()


def make_loader(args: argparse.Namespace) -> DataLoader:
    cfg = FrankaSkinDatasetConfig(
        root_dirs=[Path(args.data_root)],
        chunk_size=100,
        use_proximity=args.use_proximity,
        return_image=True,
        return_language=False,  # not consumed by the policy yet
        image_resolution=(args.image_h, args.image_w),
    )
    ds = FrankaSkinHDF5Dataset(cfg)
    print(f"[dataset] indexed {len(ds._index)} trajectories, {len(ds)} timesteps")
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)

    run_dir = Path(args.ckpt_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing checkpoints under {run_dir}")

    # ------------------- model + optimizer -------------------
    pcfg = PLAConfig(use_proximity=args.use_proximity)
    policy = PLAPolicy(pcfg).cuda()
    optim = policy.configure_optimizer(lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None:
        ck = torch.load(args.resume, map_location="cuda")
        policy.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck.get("step", 0)
        print(f"[resume] loaded {args.resume} at step {start_step}")

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[model] use_proximity={args.use_proximity}, params={n_params/1e6:.2f}M")

    # ------------------- wandb -------------------
    wandb = None
    if args.use_wandb:
        try:
            import wandb as _w
            wandb = _w
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config={**vars(args), **asdict(pcfg)},
            )
        except ImportError:
            print("[wandb] not installed; falling back to stdout-only logging.")
            wandb = None

    # ------------------- data -------------------
    loader = make_loader(args)

    # ------------------- train loop -------------------
    policy.train()
    step = start_step
    t_start = time.time()
    running = {"l1": 0.0, "kl": 0.0, "loss": 0.0}
    running_n = 0

    while step < args.num_steps:
        for batch in loader:
            qpos = batch["qpos"].cuda(non_blocking=True)
            image = batch["image"].cuda(non_blocking=True)
            actions = batch["action"].cuda(non_blocking=True)
            is_pad = batch["is_pad"].cuda(non_blocking=True)
            prox = batch["proximity"].cuda(non_blocking=True) if args.use_proximity else None

            loss_dict = policy(qpos, image, prox, actions, is_pad)
            optim.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optim.step()

            for k in running:
                running[k] += float(loss_dict[k])
            running_n += 1
            step += 1

            if step % args.log_every == 0:
                avg = {k: v / running_n for k, v in running.items()}
                throughput = (step - start_step) * args.batch_size / (time.time() - t_start)
                print(
                    f"[step {step:>7d}] loss={avg['loss']:.4f} l1={avg['l1']:.4f} "
                    f"kl={avg['kl']:.4f}  {throughput:.1f} samp/s"
                )
                if wandb is not None:
                    wandb.log({"train/" + k: v for k, v in avg.items()}, step=step)
                    wandb.log({"train/throughput_samp_per_s": throughput}, step=step)
                running = {k: 0.0 for k in running}
                running_n = 0

            if step % args.ckpt_every == 0:
                ck_path = run_dir / f"step_{step:08d}.pt"
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "args": vars(args),
                        "policy_cfg": asdict(pcfg),
                    },
                    ck_path,
                )
                # also write a stable "latest" pointer
                latest = run_dir / "latest.pt"
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(ck_path.name)
                print(f"[ckpt] saved {ck_path}")

            if step >= args.num_steps:
                break

    # final checkpoint
    ck_path = run_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "args": vars(args),
            "policy_cfg": asdict(pcfg),
        },
        ck_path,
    )
    print(f"[done] final ckpt {ck_path}")


if __name__ == "__main__":
    main()
