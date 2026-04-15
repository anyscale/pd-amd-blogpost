#!/bin/bash
# QPS sweep helper for interactive benchmark mode
# Usage: bash run_qps_sweep.sh <qps_list> <save_prefix> <measure_count> <tpot_limit> <ttft_limit>
# Example: bash run_qps_sweep.sh "0.5 1.0 1.5 2.0" "qps" 100 35 2000

QPS_LIST="$1"
SAVE_PREFIX="${2:-qps}"
MEASURE_COUNT="${3:-100}"
TPOT_LIMIT="${4:-35}"
TTFT_LIMIT="${5:-2000}"

BM="python -m ray.llm._internal.serve.benchmark -i --client --cmd"

for qps in $QPS_LIST; do
    echo "=== QPS=$qps ==="

    # Set rate
    $BM "rate $qps" 2>&1
    echo "Rate set to $qps, waiting for steady state..."

    # Wait for steady state (30s)
    for i in $(seq 1 6); do
        sleep 5
        STATUS=$($BM status 2>&1)
        echo "  [$i/6] $STATUS"
    done

    # Measure
    echo "Starting measurement ($MEASURE_COUNT requests)..."
    $BM "measure $MEASURE_COUNT" 2>&1

    # Wait for measurement to complete
    while true; do
        sleep 10
        STATUS=$($BM status 2>&1)
        echo "  Measuring... $STATUS"
        if echo "$STATUS" | grep -q "active=False"; then
            break
        fi
    done

    # Save
    SAVE_FILE="${SAVE_PREFIX}_${qps}.json"
    echo "Saving to $SAVE_FILE..."
    $BM "save $SAVE_FILE" 2>&1

    echo "=== QPS=$qps DONE ==="
    echo ""
done

echo "=== SWEEP COMPLETE ==="
