from datetime import datetime
from pathlib import Path

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinFumehoodSmokeConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = FrankaSkinFumehoodSmokeConfig()
    cfg.num_workers = 1
    cfg.seed = 2026
    cfg.use_wandb = False

    # Small fumehood batch: 2 houses x 3 episodes = 6 episodes.
    cfg.task_sampler_config.house_inds = [0, 1]
    cfg.task_sampler_config.samples_per_house = 3
    cfg.task_sampler_config.max_total_attempts_multiplier = 5
    cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 100
    cfg.task_sampler_config.max_allowed_sequential_rollout_failures = 100
    cfg.task_sampler_config.max_allowed_sequential_irrecoverable_failures = 1000

    cfg.output_dir = (
        Path("assets/datagen")
        / "roger_fumehood_nonhybrid_smoke6"
        / "FrankaSkinFumehoodSmokeConfig"
        / run_id
    )

    print("output_dir =", cfg.output_dir, flush=True)
    print("tag =", cfg.tag, flush=True)
    print("robot =", type(cfg.robot_config).__name__, flush=True)
    print("sampler =", cfg.task_sampler_config.task_sampler_class.__name__, flush=True)
    print("policy =", type(cfg.policy_config).__name__, flush=True)
    print("house_inds =", cfg.task_sampler_config.house_inds, flush=True)
    print("samples_per_house =", cfg.task_sampler_config.samples_per_house, flush=True)

    success_count, total_count = ParallelRolloutRunner(cfg).run()
    print("Final Success count:", success_count, "Total count:", total_count, flush=True)


if __name__ == "__main__":
    main()
