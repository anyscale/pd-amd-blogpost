#!/usr/bin/env python3
"""
Extract maximum sustainable QPS under SLA constraints from benchmark JSON files.

SLA Constraints:
  - TPOT (avg) < 35 ms
  - TTFT (avg) < 3000 ms

This script scans benchmark result directories for various PD-disaggregated
and aggregated TP8 configurations, extracts per-QPS metrics, and finds the
maximum QPS that satisfies both SLA constraints.

Output: A summary table printed to stdout and a JSON file saved for downstream
plotting (Figure 5: QPS capacity vs GPU count scaling curve).
"""

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SLA thresholds
# ---------------------------------------------------------------------------
TPOT_THRESHOLD_MS = 35.0
TTFT_THRESHOLD_MS = 3000.0

# ---------------------------------------------------------------------------
# Config definitions: (label, directory_path, gpu_count, mode)
# mode is "pd" for prefill-decode disaggregated, "agg" for aggregated
# ---------------------------------------------------------------------------
BASE = Path("/home/ray/default/work_pd")

CONFIGS = [
    {
        "label": "1P1D TP8",
        "dir": BASE / "thesis_3_qwen_pd_ratio_tp_exploration/results/1p1d_tp8_opt_16k1k",
        "gpu_count": 16,
        "mode": "pd",
        "prefill_nodes": 1,
        "decode_nodes": 1,
    },
    {
        "label": "2P1D TP8",
        "dir": BASE / "thesis_5_tp8_pd_ratio_scaling/results/2p1d_tp8_opt",
        "gpu_count": 24,
        "mode": "pd",
        "prefill_nodes": 2,
        "decode_nodes": 1,
    },
    {
        "label": "3P1D TP8",
        "dir": BASE / "thesis_5_tp8_pd_ratio_scaling/results/3p1d_tp8_opt",
        "gpu_count": 32,
        "mode": "pd",
        "prefill_nodes": 3,
        "decode_nodes": 1,
    },
    {
        "label": "2Agg TP8",
        "dir": BASE / "thesis_4_pareto_and_tp8_agg/results/2agg_tp8_16k1k",
        "gpu_count": 16,
        "mode": "agg",
        "prefill_nodes": 0,
        "decode_nodes": 0,
        "agg_nodes": 2,
    },
    {
        "label": "3Agg TP8",
        "dir": BASE / "thesis_5_tp8_pd_ratio_scaling/results/3agg_tp8",
        "gpu_count": 24,
        "mode": "agg",
        "prefill_nodes": 0,
        "decode_nodes": 0,
        "agg_nodes": 3,
    },
    {
        "label": "4Agg TP8",
        "dir": BASE / "thesis_5_tp8_pd_ratio_scaling/results/4agg_tp8",
        "gpu_count": 32,
        "mode": "agg",
        "prefill_nodes": 0,
        "decode_nodes": 0,
        "agg_nodes": 4,
    },
]


def parse_qps_from_filename(filename: str) -> float:
    """Extract QPS value from filenames like qps_1p0.json or qps_1.0.json."""
    name = Path(filename).stem  # e.g. "qps_1p0" or "qps_1.0"
    # Remove the "qps_" prefix
    val_str = name.replace("qps_", "")
    # Handle both "1p0" (underscore/p notation) and "1.0" (dot notation)
    val_str = val_str.replace("p", ".")
    return float(val_str)


def extract_metrics_from_file(filepath: str) -> dict:
    """Load a benchmark JSON file and return the key metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    window = data.get("window", {})
    spec = data.get("spec", {})
    config = data.get("config", {})

    # NOTE: spec.request_rate is always 1.0 due to a bug in the benchmark tool's
    # save logic. The actual target QPS is encoded in the filename (e.g. qps_2p0.json
    # means target 2.0 QPS). window.request_rate shows the achieved rate.
    target_qps = parse_qps_from_filename(filepath)
    achieved_qps = window.get("request_rate", target_qps)

    return {
        "qps": target_qps,
        "achieved_qps": achieved_qps,
        "avg_tpot_ms": window.get("avg_tpot_ms"),
        "avg_ttft_ms": window.get("avg_ttft_ms"),
        "p50_tpot_ms": window.get("p50_tpot_ms"),
        "p90_tpot_ms": window.get("p90_tpot_ms"),
        "p99_tpot_ms": window.get("p99_tpot_ms"),
        "p50_ttft_ms": window.get("p50_ttft_ms"),
        "p90_ttft_ms": window.get("p90_ttft_ms"),
        "p99_ttft_ms": window.get("p99_ttft_ms"),
        "throughput_tok_s": window.get("throughput_tok_s"),
        "avg_latency_ms": window.get("avg_latency_ms"),
        "requests": window.get("requests"),
        "isl": config.get("isl"),
        "osl": config.get("osl"),
        "hit_rate": config.get("hit_rate"),
    }


def scan_directory(dir_path: str) -> list:
    """Scan a directory for all qps_*.json files and extract metrics."""
    results = []
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"  WARNING: Directory does not exist: {dir_path}", file=sys.stderr)
        return results

    json_files = sorted(dir_path.glob("qps_*.json"))
    if not json_files:
        print(f"  WARNING: No qps_*.json files found in: {dir_path}", file=sys.stderr)
        return results

    for jf in json_files:
        try:
            metrics = extract_metrics_from_file(str(jf))
            metrics["file"] = str(jf)
            results.append(metrics)
        except Exception as e:
            print(f"  ERROR reading {jf}: {e}", file=sys.stderr)

    # Sort by QPS
    results.sort(key=lambda x: x["qps"])
    return results


def find_max_sla_qps(qps_points: list) -> tuple:
    """
    Find the maximum QPS where both SLA constraints hold:
      - avg_tpot_ms < TPOT_THRESHOLD_MS
      - avg_ttft_ms < TTFT_THRESHOLD_MS

    Returns (max_qps, metrics_at_max_qps) or (None, None) if no point passes.
    """
    max_qps = None
    max_metrics = None

    for pt in qps_points:
        tpot = pt.get("avg_tpot_ms")
        ttft = pt.get("avg_ttft_ms")
        if tpot is None or ttft is None:
            continue
        if tpot < TPOT_THRESHOLD_MS and ttft < TTFT_THRESHOLD_MS:
            if max_qps is None or pt["qps"] > max_qps:
                max_qps = pt["qps"]
                max_metrics = pt

    return max_qps, max_metrics


def main():
    print("=" * 90)
    print("SLA Capacity Analysis: Max QPS under TPOT < {:.0f}ms AND TTFT < {:.0f}ms".format(
        TPOT_THRESHOLD_MS, TTFT_THRESHOLD_MS))
    print("=" * 90)
    print()

    all_results = {}  # For JSON export

    for cfg in CONFIGS:
        label = cfg["label"]
        dir_path = cfg["dir"]
        gpu_count = cfg["gpu_count"]

        print(f"--- {label} ({gpu_count} GPUs) ---")
        print(f"    Source: {dir_path}")

        qps_points = scan_directory(dir_path)
        if not qps_points:
            print(f"    No data found.\n")
            continue

        # Print all QPS points
        print(f"    {'QPS':>6s}  {'TPOT(avg)':>10s}  {'TTFT(avg)':>10s}  {'Tput(tok/s)':>12s}  {'SLA Pass':>10s}")
        print(f"    {'----':>6s}  {'---------':>10s}  {'---------':>10s}  {'-----------':>12s}  {'--------':>10s}")
        for pt in qps_points:
            tpot = pt.get("avg_tpot_ms", 0)
            ttft = pt.get("avg_ttft_ms", 0)
            tput = pt.get("throughput_tok_s", 0)
            sla_pass = tpot < TPOT_THRESHOLD_MS and ttft < TTFT_THRESHOLD_MS
            marker = "PASS" if sla_pass else "FAIL"
            # Show which constraint fails
            fail_reasons = []
            if tpot >= TPOT_THRESHOLD_MS:
                fail_reasons.append("TPOT")
            if ttft >= TTFT_THRESHOLD_MS:
                fail_reasons.append("TTFT")
            if fail_reasons:
                marker = f"FAIL({','.join(fail_reasons)})"
            print(f"    {pt['qps']:6.2f}  {tpot:10.2f}  {ttft:10.2f}  {tput:12.1f}  {marker:>10s}")

        max_qps, max_metrics = find_max_sla_qps(qps_points)
        if max_qps is not None:
            print(f"\n    >> Max SLA QPS: {max_qps:.2f}  "
                  f"(TPOT={max_metrics['avg_tpot_ms']:.2f}ms, "
                  f"TTFT={max_metrics['avg_ttft_ms']:.2f}ms, "
                  f"Throughput={max_metrics['throughput_tok_s']:.1f} tok/s)")
        else:
            print(f"\n    >> No QPS point meets SLA constraints!")
        print()

        # Store results for JSON
        config_result = {
            "label": label,
            "gpu_count": gpu_count,
            "mode": cfg["mode"],
            "max_sla_qps": max_qps,
            "all_qps_points": [],
        }
        if cfg["mode"] == "pd":
            config_result["prefill_nodes"] = cfg["prefill_nodes"]
            config_result["decode_nodes"] = cfg["decode_nodes"]
        else:
            config_result["agg_nodes"] = cfg.get("agg_nodes", 0)

        if max_metrics:
            config_result["at_max_sla"] = {
                "avg_tpot_ms": max_metrics["avg_tpot_ms"],
                "avg_ttft_ms": max_metrics["avg_ttft_ms"],
                "throughput_tok_s": max_metrics["throughput_tok_s"],
            }

        for pt in qps_points:
            config_result["all_qps_points"].append({
                "qps": pt["qps"],
                "achieved_qps": pt.get("achieved_qps"),
                "avg_tpot_ms": pt["avg_tpot_ms"],
                "avg_ttft_ms": pt["avg_ttft_ms"],
                "p50_tpot_ms": pt["p50_tpot_ms"],
                "p90_tpot_ms": pt["p90_tpot_ms"],
                "p99_tpot_ms": pt["p99_tpot_ms"],
                "p50_ttft_ms": pt["p50_ttft_ms"],
                "p90_ttft_ms": pt["p90_ttft_ms"],
                "p99_ttft_ms": pt["p99_ttft_ms"],
                "throughput_tok_s": pt["throughput_tok_s"],
                "avg_latency_ms": pt["avg_latency_ms"],
                "sla_pass": (pt["avg_tpot_ms"] < TPOT_THRESHOLD_MS
                             and pt["avg_ttft_ms"] < TTFT_THRESHOLD_MS),
            })

        all_results[label] = config_result

    # ---------------------------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY TABLE: Max Sustainable QPS under SLA")
    print(f"  SLA: TPOT(avg) < {TPOT_THRESHOLD_MS:.0f}ms, TTFT(avg) < {TTFT_THRESHOLD_MS:.0f}ms")
    print(f"  Workload: ISL=16K, OSL=1K, 0% Hit Rate")
    print("=" * 90)
    print(f"{'Config':<15s} {'Mode':<6s} {'GPUs':>5s} {'Max QPS':>8s} {'TPOT(ms)':>9s} {'TTFT(ms)':>9s} {'Tput(tok/s)':>12s}")
    print(f"{'-'*15:<15s} {'-'*6:<6s} {'-'*5:>5s} {'-'*8:>8s} {'-'*9:>9s} {'-'*9:>9s} {'-'*12:>12s}")

    for cfg in CONFIGS:
        label = cfg["label"]
        r = all_results.get(label, {})
        max_qps = r.get("max_sla_qps")
        gpu = cfg["gpu_count"]
        mode = cfg["mode"].upper()
        if max_qps is not None:
            at_max = r.get("at_max_sla", {})
            tpot = at_max.get("avg_tpot_ms", 0)
            ttft = at_max.get("avg_ttft_ms", 0)
            tput = at_max.get("throughput_tok_s", 0)
            print(f"{label:<15s} {mode:<6s} {gpu:>5d} {max_qps:>8.2f} {tpot:>9.2f} {ttft:>9.2f} {tput:>12.1f}")
        else:
            print(f"{label:<15s} {mode:<6s} {gpu:>5d} {'N/A':>8s} {'N/A':>9s} {'N/A':>9s} {'N/A':>12s}")

    print()

    # ---------------------------------------------------------------------------
    # Save JSON output
    # ---------------------------------------------------------------------------
    output_path = Path(
        "/home/ray/default/agents_artifacts/pd_disaggregation_amd_blog_20260408"
        "/repo/results/qwen3_235b/exp3_reprocessed_scaling_data.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "description": "Max sustainable QPS under SLA for PD-disaggregated and aggregated TP8 configs",
        "model": "Qwen3-235B-A22B-FP8",
        "workload": {
            "isl": 16000,
            "osl": 1024,
            "hit_rate": 0.0,
        },
        "sla": {
            "tpot_avg_ms": TPOT_THRESHOLD_MS,
            "ttft_avg_ms": TTFT_THRESHOLD_MS,
        },
        "configs": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # ---------------------------------------------------------------------------
    # Quick scaling summary for Figure 5
    # ---------------------------------------------------------------------------
    print("=" * 90)
    print("Figure 5 Data Points (QPS Capacity vs GPU Count)")
    print("=" * 90)
    print()
    print("PD-Disaggregated:")
    for cfg in CONFIGS:
        if cfg["mode"] == "pd":
            r = all_results.get(cfg["label"], {})
            qps = r.get("max_sla_qps", "N/A")
            print(f"  {cfg['gpu_count']} GPUs ({cfg['label']}): {qps} QPS")

    print()
    print("Aggregated:")
    for cfg in CONFIGS:
        if cfg["mode"] == "agg":
            r = all_results.get(cfg["label"], {})
            qps = r.get("max_sla_qps", "N/A")
            print(f"  {cfg['gpu_count']} GPUs ({cfg['label']}): {qps} QPS")


if __name__ == "__main__":
    main()
