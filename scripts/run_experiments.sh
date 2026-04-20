#!/bin/bash
# Run benchmark experiments for PD disaggregation on AMD MI325X.
#
# Usage:
#   ./scripts/run_experiments.sh                          # run all
#   ./scripts/run_experiments.sh pd_2p1d_isl16k_osl1k     # run one
#
# Prerequisites:
#   - Ray cluster with 2-4 nodes × 8 MI325X GPUs
#   - python -m ray.llm._internal.serve.benchmark available
#
# Each experiment deploys a config, sweeps QPS, saves results, shuts down.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/qwen3_235b"
BENCHMARK="python -m ray.llm._internal.serve.benchmark"
MODEL="qwen3-235b"
TOKENIZER="Qwen/Qwen3-235B-A22B-FP8"
BASE_URL="http://localhost:8000"

deploy_and_wait() {
    local config="$1"
    local max_wait=600
    local elapsed=0
    echo ">>> Deploying ${config}..."
    cd "${REPO_ROOT}"
    ray serve deploy "serve_configs/${config}" 2>&1
    echo ">>> Waiting for RUNNING (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        local app_status=$(ray serve status 2>/dev/null | grep -A2 'applications:' | grep 'status:' | head -1 | awk '{print $2}')
        if [ "$app_status" = "RUNNING" ]; then
            echo ">>> RUNNING after ${elapsed}s"
            break
        elif [ "$app_status" = "DEPLOY_FAILED" ]; then
            echo ">>> DEPLOY_FAILED"; ray serve status 2>/dev/null | head -20; return 1
        fi
        sleep 15; elapsed=$((elapsed + 15))
    done
    [ $elapsed -ge $max_wait ] && { echo ">>> Timed out"; return 1; }
    sleep 15
    echo ">>> Smoke test..."
    ${BENCHMARK} -u ${BASE_URL} -m ${MODEL} --tokenizer ${TOKENIZER} -s 2>&1 | tail -3
}

shutdown_service() {
    echo ">>> Shutting down..."
    ray serve shutdown -y 2>&1
    sleep 10
}

run_sweep() {
    local result_name="$1" isl="$2" osl="$3" hr="$4" turns="$5" qps_list="$6"
    local sessions="${7:-50}"
    local save_dir="${RESULTS_DIR}/${result_name}"
    mkdir -p "${save_dir}"
    echo ">>> Sweep: ${result_name} (ISL=${isl}, OSL=${osl}, HR=${hr}, turns=${turns})"
    IFS=',' read -ra QPS_ARRAY <<< "${qps_list}"
    for qps in "${QPS_ARRAY[@]}"; do
        local qps_tag=$(echo "$qps" | tr '.' 'p')
        echo "  QPS=${qps}"
        ${BENCHMARK} -u ${BASE_URL} -m ${MODEL} --tokenizer ${TOKENIZER} \
            --isl ${isl} --osl ${osl} --hit-rate ${hr} --num-turns ${turns} \
            --request-rate ${qps} --num-sessions ${sessions} \
            --save-result "${save_dir}/qps_${qps_tag}.json" 2>&1 | tail -3
    done
}

# ============================================================
# EXPERIMENTS — naming: {config}_{workload}
# ============================================================

# --- ISL=16K, OSL=1K, 0% HR (24 GPU) ---
pd_2p1d_isl16k_osl1k() {
    deploy_and_wait "pd/qwen235b_2p1d_tp8.yaml"
    run_sweep "pd_2p1d_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
agg_3x_isl16k_osl1k() {
    deploy_and_wait "agg/qwen235b_3agg_tp8.yaml"
    run_sweep "agg_3x_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

# --- ISL=8K, OSL=1K, 0% HR (24 GPU) ---
pd_1p2d_isl8k_osl1k() {
    deploy_and_wait "pd/qwen235b_1p2d_tp8.yaml"
    run_sweep "pd_1p2d_isl8k_osl1k_hr0" 8000 1024 0.0 1 "0.5,1.0,2.0,3.0,4.0,5.0"
    shutdown_service
}
agg_3x_isl8k_osl1k() {
    deploy_and_wait "agg/qwen235b_3agg_tp8.yaml"
    run_sweep "agg_3x_isl8k_osl1k_hr0" 8000 1024 0.0 1 "0.5,1.0,2.0,3.0,4.0,5.0"
    shutdown_service
}

# --- ISL=8K, OSL=1K, 60% HR (24 GPU) ---
pd_1p2d_isl8k_osl1k_hr60() {
    deploy_and_wait "pd/qwen235b_1p2d_tp8_cache.yaml"
    run_sweep "pd_1p2d_isl8k_osl1k_hr60" 8000 1024 0.6 1 "0.5,1.0,2.0,3.0,4.0,5.0,6.0"
    shutdown_service
}
agg_3x_isl8k_osl1k_hr60() {
    deploy_and_wait "agg/qwen235b_3agg_tp8.yaml"
    run_sweep "agg_3x_isl8k_osl1k_hr60" 8000 1024 0.6 1 "0.5,1.0,2.0,3.0,4.0,5.0,6.0"
    shutdown_service
}

# --- ISL=16K, OSL=4K, 0% HR (32 GPU) ---
pd_1p3d_isl16k_osl4k() {
    deploy_and_wait "pd/qwen235b_1p3d_tp8.yaml"
    run_sweep "pd_1p3d_isl16k_osl4k_hr0" 16000 4096 0.0 1 "0.25,0.5,0.75,1.0,1.25,1.5" 20
    shutdown_service
}
agg_4x_isl16k_osl4k() {
    deploy_and_wait "agg/qwen235b_4agg_tp8.yaml"
    run_sweep "agg_4x_isl16k_osl4k_hr0" 16000 4096 0.0 1 "0.25,0.5,0.75,1.0,1.25,1.5" 10
    shutdown_service
}

# --- Scaling curve: ISL=16K, OSL=1K, 0% HR (various GPU counts) ---
pd_1p1d_isl16k_osl1k() {
    deploy_and_wait "pd/qwen235b_1p1d_tp8.yaml"
    run_sweep "pd_1p1d_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
agg_2x_isl16k_osl1k() {
    deploy_and_wait "agg/qwen235b_2agg_tp8.yaml"
    run_sweep "agg_2x_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
agg_4x_isl16k_osl1k() {
    deploy_and_wait "agg/qwen235b_4agg_tp8.yaml"
    run_sweep "agg_4x_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
pd_1p2d_isl16k_osl1k() {
    deploy_and_wait "pd/qwen235b_1p2d_tp8.yaml"
    run_sweep "pd_1p2d_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
pd_2p2d_isl16k_osl1k() {
    deploy_and_wait "pd/qwen235b_2p2d_tp8.yaml"
    run_sweep "pd_2p2d_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}
pd_1p3d_isl16k_osl1k() {
    deploy_and_wait "pd/qwen235b_1p3d_tp8.yaml"
    run_sweep "pd_1p3d_isl16k_osl1k_hr0" 16000 1024 0.0 1 "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0"
    shutdown_service
}

# ============================================================
# DeepSeek-V3 — ISL=5.4K, OSL=140 (16 GPU, prefix caching)
# ============================================================
# Each function temporarily overrides MODEL/TOKENIZER/RESULTS_DIR so the
# shared deploy_and_wait/run_sweep helpers route to the DeepSeek model and
# results directory.

deepseek_pd_1p1d_isl5k_osl140_hr30() {
    local MODEL="deepseek-v3"
    local TOKENIZER="deepseek-ai/DeepSeek-V3-0324"
    local RESULTS_DIR="${REPO_ROOT}/results/deepseek_v3"
    deploy_and_wait "pd/deepseek_v3_1p1d_tp8.yaml"
    run_sweep "pd_1p1d_isl5k_osl140_hr30" 5400 140 0.3 1 "3.0,4.0,4.5,5.0,6.0,7.0"
    shutdown_service
}
deepseek_pd_1p1d_isl5k_osl140_hr60() {
    local MODEL="deepseek-v3"
    local TOKENIZER="deepseek-ai/DeepSeek-V3-0324"
    local RESULTS_DIR="${REPO_ROOT}/results/deepseek_v3"
    deploy_and_wait "pd/deepseek_v3_1p1d_tp8.yaml"
    run_sweep "pd_1p1d_isl5k_osl140_hr60" 5400 140 0.6 1 "3.0,4.0,4.5,5.0,6.0,7.0"
    shutdown_service
}
deepseek_agg_2x_isl5k_osl140_hr30() {
    local MODEL="deepseek-v3"
    local TOKENIZER="deepseek-ai/DeepSeek-V3-0324"
    local RESULTS_DIR="${REPO_ROOT}/results/deepseek_v3"
    deploy_and_wait "agg/deepseek_v3_2agg_tp8.yaml"
    run_sweep "agg_2x_isl5k_osl140_hr30" 5400 140 0.3 1 "3.0,4.0,4.5,5.0,6.0,7.0"
    shutdown_service
}
deepseek_agg_2x_isl5k_osl140_hr60() {
    local MODEL="deepseek-v3"
    local TOKENIZER="deepseek-ai/DeepSeek-V3-0324"
    local RESULTS_DIR="${REPO_ROOT}/results/deepseek_v3"
    deploy_and_wait "agg/deepseek_v3_2agg_tp8.yaml"
    run_sweep "agg_2x_isl5k_osl140_hr60" 5400 140 0.6 1 "3.0,4.0,4.5,5.0,6.0,7.0"
    shutdown_service
}

# ============================================================
# MAIN
# ============================================================
if [ $# -eq 0 ]; then
    echo "Running all experiments sequentially..."
    pd_2p1d_isl16k_osl1k
    agg_3x_isl16k_osl1k
    pd_1p2d_isl8k_osl1k
    agg_3x_isl8k_osl1k
    pd_1p2d_isl8k_osl1k_hr60
    agg_3x_isl8k_osl1k_hr60
    pd_1p3d_isl16k_osl4k
    agg_4x_isl16k_osl4k
    pd_1p1d_isl16k_osl1k
    agg_2x_isl16k_osl1k
    agg_4x_isl16k_osl1k
    pd_1p2d_isl16k_osl1k
    pd_2p2d_isl16k_osl1k
    pd_1p3d_isl16k_osl1k
    deepseek_pd_1p1d_isl5k_osl140_hr30
    deepseek_pd_1p1d_isl5k_osl140_hr60
    deepseek_agg_2x_isl5k_osl140_hr30
    deepseek_agg_2x_isl5k_osl140_hr60
    echo "=== ALL EXPERIMENTS COMPLETE ==="
else
    for exp in "$@"; do "${exp}"; done
fi
