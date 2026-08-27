from datetime import datetime
from pathlib import Path

from scratch.run_prox_necessity_pilot import CustomParallelRolloutRunner
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig,
)


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig()

    cfg.num_workers = 1
    cfg.seed = 2026
    cfg.use_wandb = False
    cfg.collision_free_pose_limit = 30
    cfg.filter_for_successful_trajectories = True

    # Focus only on the promising houses from smoke5 audit.
    cfg.task_sampler_config.house_inds = [9, 11]
    cfg.task_sampler_config.samples_per_house = 8

    # Narrow object set to improve success rate and proximity relevance.
    cfg.task_sampler_config.pickup_types = [
        "mug",
        "cup",
        "apple",
        "cellphone",
        "remote",
        "spoon",
        "fork",
        "knife",
    ]

    cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_rollout_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_irrecoverable_failures = 10000
    cfg.task_sampler_config.max_total_attempts_multiplier = 30
    cfg.task_sampler_config.max_asset_failures = 10000
    cfg.task_sampler_config.max_robot_placement_attempts = 100
    cfg.task_sampler_config.base_pose_sampling_radius_range = (0.0, 1.2)

    cfg.policy_config.filter_colliding_grasps = False

    cfg.output_dir = (
        Path("assets/datagen")
        / "roger_open_front_cluttered_improved_h9_h11"
        / "FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig"
        / run_id
    )

    print("output_dir =", cfg.output_dir, flush=True)
    print("house_inds =", cfg.task_sampler_config.house_inds, flush=True)
    print("samples_per_house =", cfg.task_sampler_config.samples_per_house, flush=True)
    print("pickup_types =", cfg.task_sampler_config.pickup_types, flush=True)
    print("source_surface_types =", cfg.task_sampler_config.source_surface_types, flush=True)

    CustomParallelRolloutRunner.target_successes = 8
    success_count, total_count = CustomParallelRolloutRunner(cfg).run()
    print("Final Success count:", success_count, "Total count:", total_count, flush=True)


if __name__ == "__main__":
    main()
