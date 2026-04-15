#!/bin/bash
# Run all benchmark experiments for the PD disaggregation blog.
#
# Usage:
#   ./scripts/run_experiments.sh              # run all experiments
#   ./scripts/run_experiments.sh exp1a_pd     # run a single experiment
#
# Prerequisites:
#   - Ray cluster running with 4 nodes × 8 MI325X GPUs
#   - ray serve CLI available
#   - python -m ray.llm._internal.serve.benchmark available
#
# Each experiment:
#   1. Deploys a serve config
#   2. Waits for RUNNING status
#   3. Runs a QPS sweep via interactive benchmark
#   4. Saves results to results/qwen3_235b/<exp_name>/
#   5. Shuts down the service

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/qwen3_235b"
BENCHMARK="python -m ray.llm._internal.serve.benchmark"
MODEL="qwen3-235b"
TOKENIZER="Qwen/Qwen3-235B-A22B-FP8"
BASE_URL="http://localhost:8000"

deploy_and_wait() {
    local config="$1"
    echo ">>> Deploying ${config}..."
    cd "${REPO_ROOT}"
    ray serve deploy "serve_configs/${config}" 2>&1
    echo ">>> Waiting for RUNNING status..."
    while true; do
        status=$(ray serve status 2>/dev/null | grep -m1 'status:' | awk '{print $2}')
        if [ "$status" = "RUNNING" ]; then
            echo ">>> Service is RUNNING"
            break
        elif [ "$status" = "DEPLOY_FAILED" ]; then
            echo ">>> DEPLOY_FAILED — aborting"
            return 1
        fi
        sleep 10
    done
    # Extra settle time for vLLM engine warm-up
    sleep 15
    # Smoke test
    echo ">>> Smoke test..."
    ${BENCHMARK} -u ${BASE_URL} -m ${MODEL} --tokenizer ${TOKENIZER} -s 2>&1 | tail -3
}

shutdown_service() {
    echo ">>> Shutting down service..."
    ray serve shutdown -y 2>&1
    sleep 10
}

run_sweep() {
    local exp_name="$1"
    local isl="$2"
    local osl="$3"
    local hit_rate="$4"
    local num_turns="$5"
    local qps_list="$6"
    local num_sessions="${7:-50}"

    local save_dir="${RESULTS_DIR}/${exp_name}"
    mkdir -p "${save_dir}"

    echo ">>> Running benchmark: ${exp_name}"
    echo ">>>   Workload: ISL=${isl}, OSL=${osl}, HR=${hit_rate}, turns=${num_turns}"
    echo ">>>   QPS sweep: ${qps_list}"
    echo ">>>   Sessions per point: ${num_sessions}"
    echo ">>>   Save dir: ${save_dir}"

    # Run each QPS point individually for clean results
    IFS=',' read -ra QPS_ARRAY <<< "${qps_list}"
    for qps in "${QPS_ARRAY[@]}"; do
        local qps_tag=$(echo "$qps" | tr '.' 'p')
        local save_file="${save_dir}/qps_${qps_tag}.json"

        echo ">>>   QPS=${qps} → ${save_file}"
        ${BENCHMARK} \
            -u ${BASE_URL} \
            -m ${MODEL} \
            --tokenizer ${TOKENIZER} \
            --isl ${isl} \
            --osl ${osl} \
            --hit-rate ${hit_rate} \
            --num-turns ${num_turns} \
            --request-rate ${qps} \
            --num-sessions ${num_sessions} \
            --save-result "${save_file}" \
            2>&1 | tail -5

        echo ">>>   Done QPS=${qps}"
    done

    echo ">>> Experiment ${exp_name} complete. Results in ${save_dir}/"
}

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

exp1a_pd() {
    echo "=== EXP-1a PD: 2P1D TP8 (24 GPU), ISL=16K, OSL=1K, 0% HR ==="
    deploy_and_wait "pd/qwen235b_2p1d_tp8.yaml"
    run_sweep "exp1a_2p1d_tp8" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

exp1a_agg() {
    echo "=== EXP-1a Agg: 3Agg TP8 (24 GPU), ISL=16K, OSL=1K, 0% HR ==="
    deploy_and_wait "agg/qwen235b_3agg_tp8.yaml"
    run_sweep "exp1a_3agg_tp8" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    # Reuse for EXP-1d Agg (same config, different workload)
    run_sweep "exp1d_3agg_tp8" 8000 1024 0.0 1 "0.5,1.0,2.0,3.0,4.0,5.0"
    shutdown_service
}

exp1d_pd() {
    echo "=== EXP-1d PD: 1P2D TP8 (24 GPU), ISL=8K, OSL=1K, 0% HR ==="
    deploy_and_wait "pd/qwen235b_1p2d_tp8.yaml"
    run_sweep "exp1d_1p2d_tp8" 8000 1024 0.0 1 "0.5,1.0,2.0,3.0,4.0,5.0"
    # Reuse for EXP-3c (same config, different workload)
    run_sweep "exp3c_1p2d_tp8" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

exp1c_pd() {
    echo "=== EXP-1c PD: 1P2D TP8 + cache (24 GPU), ISL=8K, OSL=1K, 80% HR, 5 turns ==="
    deploy_and_wait "pd/qwen235b_1p2d_tp8_cache.yaml"
    run_sweep "exp1c_1p2d_tp8_cache" 8000 1024 0.8 5 "0.5,1.0,2.0,3.0,4.0,5.0,6.0" 30
    shutdown_service
}

exp1c_agg() {
    echo "=== EXP-1c Agg: 3Agg TP8 (24 GPU), ISL=8K, OSL=1K, 80% HR, 5 turns ==="
    # Note: 3Agg config does not have prefix_caching — need to verify if the
    # default enables it or if we need a separate config
    deploy_and_wait "agg/qwen235b_3agg_tp8.yaml"
    run_sweep "exp1c_3agg_tp8" 8000 1024 0.8 5 "0.5,1.0,2.0,3.0,4.0,5.0,6.0" 30
    shutdown_service
}

exp1b_pd() {
    echo "=== EXP-1b PD: 1P3D TP8 (32 GPU), ISL=16K, OSL=4K, 0% HR ==="
    deploy_and_wait "pd/qwen235b_1p3d_tp8.yaml"
    run_sweep "exp1b_1p3d_tp8" 16000 4096 0.0 1 "0.25,0.5,0.75,1.0,1.25,1.5"
    # Reuse for EXP-3f (same config, different workload)
    run_sweep "exp3f_1p3d_tp8" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

exp1b_agg() {
    echo "=== EXP-1b Agg: 4Agg TP8 (32 GPU), ISL=16K, OSL=4K, 0% HR ==="
    deploy_and_wait "agg/qwen235b_4agg_tp8.yaml"
    run_sweep "exp1b_4agg_tp8" 16000 4096 0.0 1 "0.25,0.5,0.75,1.0,1.25,1.5"
    shutdown_service
}

exp3d() {
    echo "=== EXP-3d: 2P2D TP8 (32 GPU), ISL=16K, OSL=1K, 0% HR ==="
    deploy_and_wait "pd/qwen235b_2p2d_tp8.yaml"
    run_sweep "exp3d_2p2d_tp8" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

# ============================================================
# MAIN
# ============================================================

if [ $# -eq 0 ]; then
    echo "Running ALL experiments sequentially..."
    echo "Estimated time: 4-6 hours (8 deploys × ~30-45 min each)"
    echo ""
    exp1a_pd
    exp1a_agg
    exp1d_pd
    exp1c_pd
    exp1c_agg
    exp1b_pd
    exp1b_agg
    exp3d
    echo ""
    echo "=== ALL EXPERIMENTS COMPLETE ==="
    echo "Results in: ${RESULTS_DIR}/"
else
    # Run specific experiment(s)
    for exp in "$@"; do
        ${exp}
    done
fi
