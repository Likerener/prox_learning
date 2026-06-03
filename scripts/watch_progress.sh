#!/usr/bin/env bash
# Prints a brief progress summary for all running experiments.
# Use: bash scripts/watch_progress.sh

cd /home/jaydv/code/prox_learning

echo "[$(date +%H:%M:%S)] === PROGRESS ==="
echo "--- subprocess count ---"
ps -ef | grep eval_act_with_prox_encoder.py | grep -v grep | wc -l

echo "--- memory / GPU ---"
free -h | head -2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

for d in eval_output/exp1_mask_zero_n50 \
         eval_output/exp1_mask_mean_n50 \
         eval_output/exp2_mask_approach_n50 \
         eval_output/exp2_mask_pregrasp_n50 \
         eval_output/exp2_mask_grasp_lift_n50 \
         eval_output/exp2_mask_transit_n50 \
         eval_output/exp2_mask_place_n50 \
         eval_output/epoch_sweep; do
    if [ -d "$d" ]; then
        if [ "$d" = "eval_output/epoch_sweep" ]; then
            # Special handling
            for ep_dir in "$d"/epoch_*; do
                [ -d "$ep_dir" ] || continue
                tot=$(ls "$ep_dir"/run_*/house_1/trajectories_batch_1_of_1.h5 2>/dev/null | wc -l)
                if [ -f "$ep_dir/summary.json" ]; then
                    rate=$(/opt/conda/envs/mlspaces/bin/python -c "import json; s=json.load(open('$ep_dir/summary.json')); print(f\"{s['pooled_success_rate']:.0%}  ({s['total_successes']}/{s['total_episodes']})\")" 2>/dev/null || echo "—")
                    printf "  %s : DONE  rate=%s\n" "$(basename "$ep_dir")" "$rate"
                else
                    printf "  %s : %d rollouts in flight\n" "$(basename "$ep_dir")" "$tot"
                fi
            done
        else
            tot=$(ls "$d"/run_*/house_1/trajectories_batch_1_of_1.h5 2>/dev/null | wc -l)
            if [ -f "$d/summary.json" ]; then
                rate=$(/opt/conda/envs/mlspaces/bin/python -c "import json; s=json.load(open('$d/summary.json')); print(f\"{s['pooled_success_rate']:.0%}  ({s['total_successes']}/{s['total_episodes']})\")" 2>/dev/null || echo "—")
                printf "  %s : DONE  rate=%s\n" "$(basename "$d")" "$rate"
            else
                printf "  %s : %d rollouts in flight\n" "$(basename "$d")" "$tot"
            fi
        fi
    fi
done
