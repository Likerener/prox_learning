from pathlib import Path

from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
from molmo_spaces.policy.act_debug_policy import ACTDebugPolicy

benchmark_dir = Path("/home/qinzhengfangli/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260327/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark")
output_dir = Path("eval_output/ACTDebugEval_1ep")

dummy_config = PiPolicyEvalConfig()
policy = ACTDebugPolicy(dummy_config, dummy_config.task_type)

results = run_evaluation(
    eval_config_cls=PiPolicyEvalConfig,
    benchmark_dir=benchmark_dir,
    checkpoint_path=None,
    task_horizon_steps=3,
    output_dir=output_dir,
    num_workers=1,
    use_wandb=False,
    preloaded_policy=policy,
    max_episodes=1,
    episode_idx=0,
)

print("success_count:", results.success_count)
print("total_count:", results.total_count)
print("output_dir:", results.output_dir)
