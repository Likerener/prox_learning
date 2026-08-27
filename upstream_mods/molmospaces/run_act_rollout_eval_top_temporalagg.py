from pathlib import Path
import json

from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
from molmo_spaces.policy.act_molmo_policy import ACTMolmoPolicy

benchmark_dir = Path("/home/qinzhengfangli/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260327/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark")

dummy_config = PiPolicyEvalConfig()
dummy_config.policy_config.checkpoint_path = "/home/qinzhengfangli/act/ckpt_molmo_wandb_1000ep/policy_best.ckpt"

policy = ACTMolmoPolicy(dummy_config, dummy_config.task_type)

results = run_evaluation(
    eval_config_cls=PiPolicyEvalConfig,
    benchmark_dir=benchmark_dir,
    checkpoint_path="/home/qinzhengfangli/act/ckpt_molmo_wandb_1000ep/policy_best.ckpt",
    task_horizon_steps=100,
    output_dir=Path("eval_output/ACTMolmoRollout_1ep_top_temporalagg"),
    num_workers=1,
    use_wandb=False,
    preloaded_policy=policy,
    max_episodes=1,
    episode_idx=0,
)

summary = {
    "success_count": results.success_count,
    "total_count": results.total_count,
    "success_rate": results.success_count / results.total_count if results.total_count else 0.0,
    "output_dir": str(results.output_dir),
}

print(json.dumps(summary, indent=2))

out = Path("/home/qinzhengfangli/act/ckpt_molmo_wandb_1000ep/rollout_eval_1ep_top_temporalagg.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))
print("saved:", out)
