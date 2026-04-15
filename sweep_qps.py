#!/usr/bin/env python3
"""Automated QPS sweep for interactive benchmark mode.

Usage:
    python sweep_qps.py --qps 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --measure 100 \
        --tpot-limit 35 --ttft-limit 2000 --prefix qps
"""
import argparse
import json
import re
import subprocess
import sys
import time


def run_cmd(cmd_str):
    """Send a command to the benchmark interactive server."""
    result = subprocess.run(
        ["python", "-m", "ray.llm._internal.serve.benchmark", "-i", "--client", "--cmd", cmd_str],
        capture_output=True, text=True, timeout=30,
    )
    output = result.stdout.strip() + result.stderr.strip()
    return output


def get_status():
    """Get current benchmark status."""
    return run_cmd("status")


def parse_measurement(output):
    """Extract measurement JSON from command output."""
    # Look for JSON block in output
    match = re.search(r'\{[^{}]*"requests"[^{}]*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def wait_for_steady_state(target_qps, wait_seconds=30):
    """Wait for the system to reach steady state."""
    print(f"  Waiting {wait_seconds}s for steady state...")
    intervals = max(1, wait_seconds // 10)
    for i in range(intervals):
        time.sleep(min(10, wait_seconds))
        status = get_status()
        print(f"  [{i+1}/{intervals}] {status}")


def wait_for_measurement(timeout=600):
    """Wait until measurement completes. Returns the full output."""
    start = time.time()
    all_output = ""
    while time.time() - start < timeout:
        time.sleep(15)
        status = get_status()
        all_output += status + "\n"
        print(f"  {status}")
        if "active=False" in status:
            return all_output
    print("  WARNING: Measurement timed out!")
    return all_output


def check_sla(result, tpot_limit, ttft_limit):
    """Check if SLA is met. Returns (passed, reason)."""
    if result is None:
        return False, "no measurement data"
    tpot = result.get("p50_tpot_ms", float("inf"))
    ttft = result.get("p50_ttft_ms", float("inf"))
    reasons = []
    if tpot > tpot_limit:
        reasons.append(f"TPOT p50={tpot:.1f}ms > {tpot_limit}ms")
    if ttft > ttft_limit:
        reasons.append(f"TTFT p50={ttft:.1f}ms > {ttft_limit}ms")
    if reasons:
        return False, "; ".join(reasons)
    return True, f"TPOT p50={tpot:.1f}ms, TTFT p50={ttft:.1f}ms"


def check_saturation(status_output):
    """Check if the system is saturated (inflight growing unbounded)."""
    inflight_vals = re.findall(r"inflight=(\d+)", status_output)
    if len(inflight_vals) >= 3:
        vals = [int(v) for v in inflight_vals[-3:]]
        if all(v > 50 for v in vals) and vals[-1] > vals[0] * 1.2:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", nargs="+", type=float, required=True)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--tpot-limit", type=float, default=35)
    parser.add_argument("--ttft-limit", type=float, default=2000)
    parser.add_argument("--prefix", default="qps")
    parser.add_argument("--steady-wait", type=int, default=30,
                        help="Seconds to wait for steady state")
    args = parser.parse_args()

    results_summary = {}
    max_sla_qps = 0
    stop_sweep = False

    for qps in args.qps:
        if stop_sweep:
            print(f"\n=== Skipping QPS={qps} (previous point saturated) ===")
            continue

        print(f"\n{'='*60}")
        print(f"QPS = {qps}")
        print(f"{'='*60}")

        # Set rate
        output = run_cmd(f"rate {qps}")
        print(f"  {output}")

        # Wait for steady state
        wait_for_steady_state(qps, args.steady_wait)

        # Start measurement
        output = run_cmd(f"measure {args.measure}")
        print(f"  {output}")

        # Wait for measurement
        all_output = wait_for_measurement(timeout=max(300, int(args.measure / qps * 3)))

        # Parse result
        result = parse_measurement(all_output)
        if result:
            sla_passed, sla_msg = check_sla(result, args.tpot_limit, args.ttft_limit)
            print(f"\n  Results: {json.dumps(result, indent=2)}")
            print(f"  SLA: {'PASS' if sla_passed else 'FAIL'} - {sla_msg}")

            if sla_passed:
                max_sla_qps = qps

            results_summary[str(qps)] = {
                "sla_passed": sla_passed,
                "sla_msg": sla_msg,
                **result,
            }
        else:
            print("  WARNING: Could not parse measurement results")
            results_summary[str(qps)] = {"sla_passed": False, "error": "parse_failed"}

        # Save
        save_file = f"{args.prefix}_{qps}.json"
        output = run_cmd(f"save {save_file}")
        print(f"  {output}")

        # Check for saturation
        if result and result.get("request_rate", 0) < qps * 0.7:
            print(f"  WARNING: Effective QPS ({result['request_rate']:.2f}) << target ({qps})")
            print(f"  System may be saturated. Stopping sweep.")
            stop_sweep = True

        if check_saturation(all_output):
            print(f"  WARNING: Inflight growing unbounded. Stopping sweep.")
            stop_sweep = True

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Max QPS under SLA: {max_sla_qps}")
    print(f"{'='*60}")

    # Print summary
    for qps_str, data in results_summary.items():
        sla = "PASS" if data.get("sla_passed") else "FAIL"
        tpot = data.get("p50_tpot_ms", "N/A")
        ttft = data.get("p50_ttft_ms", "N/A")
        rate = data.get("request_rate", "N/A")
        print(f"  QPS={qps_str}: {sla} | eff_qps={rate} | TPOT_p50={tpot} | TTFT_p50={ttft}")

    return max_sla_qps


if __name__ == "__main__":
    max_qps = main()
    # Write max_qps to a temp file for the caller
    with open("/tmp/sweep_result.txt", "w") as f:
        f.write(str(max_qps))
