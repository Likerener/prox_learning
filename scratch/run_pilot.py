from datetime import datetime
from pathlib import Path
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import FrankaSkinLowSurfacePickAndPlacePilotConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg = FrankaSkinLowSurfacePickAndPlacePilotConfig()
    cfg.num_workers = 1
    cfg.use_wandb = False
    cfg.collision_free_pose_limit = 30
    cfg.task_sampler_config.house_inds = list(range(30))
    cfg.task_sampler_config.pickup_types = ["mug"]
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_rollout_failures = 2000
    cfg.task_sampler_config.max_total_attempts_multiplier = 10
    cfg.task_sampler_config.max_asset_failures = 10_000
    cfg.task_sampler_config.max_robot_placement_attempts = 80
    cfg.task_sampler_config.base_pose_sampling_radius_range = (0.0, 1.2)
    cfg.policy_config.filter_colliding_grasps = False
    cfg.filter_for_successful_trajectories = True
    cfg.output_dir = Path('/mnt/d/Machines virtueles/prox_learning/assets/datagen/pick_and_place_skin_low_surface_mug_pilot_v1') / run_id
    cfg.save_config()
    print('output_dir=', cfg.output_dir, flush=True)
    success_count, total_count = ParallelRolloutRunner(cfg).run()
    print('Success count:', success_count, 'Total count:', total_count, flush=True)

if __name__ == '__main__':
    main()
