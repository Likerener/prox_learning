from datetime import datetime
from pathlib import Path

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinLowSurfacePickAndPlacePilotConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = FrankaSkinLowSurfacePickAndPlacePilotConfig()

    cfg.num_workers = 1
    cfg.seed = 2026
    cfg.use_wandb = False
    cfg.collision_free_pose_limit = 30
    cfg.filter_for_successful_trajectories = True

    cfg.task_sampler_config.house_inds = list(range(50))
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.pickup_types = ["mug"]

    cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_rollout_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_irrecoverable_failures = 10000
    cfg.task_sampler_config.max_total_attempts_multiplier = 25
    cfg.task_sampler_config.max_asset_failures = 10000
    cfg.task_sampler_config.max_robot_placement_attempts = 80
    cfg.task_sampler_config.base_pose_sampling_radius_range = (0.0, 1.2)

    cfg.policy_config.filter_colliding_grasps = False

    cfg.output_dir = (
        Path("assets/datagen")
        / "roger_low_surface_mug_sweep"
        / "FrankaSkinLowSurfacePickAndPlacePilotConfig"
        / run_id
    )

    print("output_dir =", cfg.output_dir, flush=True)
    print("scene_dataset =", cfg.scene_dataset, flush=True)
    print("data_split =", cfg.data_split, flush=True)
    print("house_inds =", cfg.task_sampler_config.house_inds, flush=True)
    print("pickup_types =", cfg.task_sampler_config.pickup_types, flush=True)
    print("source_surface_types =", cfg.task_sampler_config.source_surface_types, flush=True)
    print("samples_per_house =", cfg.task_sampler_config.samples_per_house, flush=True)
    print("collision_free_pose_limit =", cfg.collision_free_pose_limit, flush=True)
    print("max_robot_placement_attempts =", cfg.task_sampler_config.max_robot_placement_attempts, flush=True)
    print("base_pose_sampling_radius_range =", cfg.task_sampler_config.base_pose_sampling_radius_range, flush=True)
    print("filter_colliding_grasps =", cfg.policy_config.filter_colliding_grasps, flush=True)

    success_count, total_count = ParallelRolloutRunner(cfg).run()
    print("Final Success count:", success_count, "Total count:", total_count, flush=True)


if __name__ == "__main__":
    main()
