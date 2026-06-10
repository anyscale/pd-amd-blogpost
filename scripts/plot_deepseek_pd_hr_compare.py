#!/usr/bin/env python3
"""One-off: TPOT vs QPS for DeepSeek-V3 PD 1P1D at 30% vs 60% hit rate.

Sanity check that hit rate (a prefill-side knob) does not move the decode-side
TPOT curve when prefill and decode are physically disaggregated.
Output: figures/fig_pd_tpot_hr30_vs_hr60.png (NOT used in the blog post).
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "results", "deepseek_v3")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

COLOR_HR30 = "#1f77b4"  # blue
COLOR_HR60 = "#9467bd"  # purple
SLA_DS = 30  # ms

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 300,
})


def load_series(subdir):
    pts = []
    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, subdir, "qps_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        pts.append((d["config"]["request_rate"], d["stats"]["avg_tpot_ms"]))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    qps_30, tpot_30 = load_series("pd_1p1d_isl5k_osl140_hr30")
    qps_60, tpot_60 = load_series("pd_1p1d_isl5k_osl140_hr60")

    print("PD 1P:1D, 30% HR:")
    for q, t in zip(qps_30, tpot_30):
        print(f"  QPS={q:>4}  TPOT={t:.2f} ms")
    print("PD 1P:1D, 60% HR:")
    for q, t in zip(qps_60, tpot_60):
        print(f"  QPS={q:>4}  TPOT={t:.2f} ms")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(qps_30, tpot_30, color=COLOR_HR30, marker="o",
            linewidth=2.5, markersize=8, label="PD 1P:1D, 30% HR", zorder=3)
    ax.plot(qps_60, tpot_60, color=COLOR_HR60, marker="D",
            linewidth=2.5, markersize=7, label="PD 1P:1D, 60% HR", zorder=3)
    ax.axhline(SLA_DS, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"SLA = {SLA_DS}ms", zorder=2)

    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("DeepSeek-V3 PD 1P:1D — TPOT vs QPS\n(ISL=5.4K, OSL=140, 16 GPU)")
    ax.legend(loc="upper left", fontsize=10)

    y_top = max(max(tpot_30), max(tpot_60)) * 1.25
    ax.set_ylim(bottom=0, top=y_top)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_pd_tpot_hr30_vs_hr60.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
