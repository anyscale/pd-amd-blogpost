#!/usr/bin/env python3
"""One-off: TPOT vs QPS for DeepSeek-V3 at 60% hit rate (PD 1P1D vs 2xAgg).

Mirrors the styling of the right panel of fig_4_tpot_vs_qps.png but uses the
60% hit-rate run data. Output is written to figures/fig_4_tpot_vs_qps_deepseek_hr60.png
and is intentionally NOT wired into the blog post.
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

COLOR_PD = "#1f77b4"
COLOR_AGG = "#ff7f0e"
COLOR_WINNER = "#2ca02c"
MARKER_PD = "o"
MARKER_AGG = "s"

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
    qps = [p[0] for p in pts]
    tpot = [p[1] for p in pts]
    return qps, tpot


def interp_crossing(qps_list, tpot_list, sla):
    for i in range(len(qps_list) - 1):
        if tpot_list[i] <= sla < tpot_list[i + 1]:
            frac = (sla - tpot_list[i]) / (tpot_list[i + 1] - tpot_list[i])
            return qps_list[i] + frac * (qps_list[i + 1] - qps_list[i])
    if all(t <= sla for t in tpot_list):
        return qps_list[-1]
    return 0


def main():
    qps_agg, tpot_agg = load_series("agg_2x_isl5k_osl140_hr60")
    qps_pd, tpot_pd = load_series("pd_1p1d_isl5k_osl140_hr60")

    print("Agg 2x TP8 (60% HR):")
    for q, t in zip(qps_agg, tpot_agg):
        print(f"  QPS={q:>4}  TPOT={t:.2f} ms")
    print("PD 1P:1D TP8 (60% HR):")
    for q, t in zip(qps_pd, tpot_pd):
        print(f"  QPS={q:>4}  TPOT={t:.2f} ms")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(qps_agg, tpot_agg, color=COLOR_AGG, marker=MARKER_AGG,
            linewidth=2.5, markersize=8, label="Agg 2×Agg TP8", zorder=3)
    ax.plot(qps_pd, tpot_pd, color=COLOR_PD, marker=MARKER_PD,
            linewidth=2.5, markersize=8, label="PD 1P:1D TP8", zorder=3)
    ax.axhline(SLA_DS, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"SLA = {SLA_DS}ms", zorder=2)

    above = [(q, t) for q, t in zip(qps_agg, tpot_agg) if t > SLA_DS]
    if len(above) >= 2:
        qs, ts = zip(*above)
        ax.fill_between(qs, SLA_DS, ts, color="red", alpha=0.08, zorder=1)

    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("DeepSeek-V3  (ISL=5.4K, OSL=140, 60% HR)")
    ax.legend(loc="upper left", fontsize=9)
    y_top = max(max(tpot_agg), max(tpot_pd)) * 1.25
    ax.set_ylim(bottom=0, top=y_top)

    agg_max = interp_crossing(qps_agg, tpot_agg, SLA_DS)
    pd_max = interp_crossing(qps_pd, tpot_pd, SLA_DS)
    print(f"\nSLA crossings (TPOT={SLA_DS} ms): Agg={agg_max:.2f} QPS, PD={pd_max:.2f} QPS")

    if agg_max > 0 and pd_max > 0 and pd_max > agg_max:
        mult = pd_max / agg_max
        ax.plot([agg_max, agg_max], [0, SLA_DS], color=COLOR_AGG,
                linestyle=":", linewidth=1.0, alpha=0.5, zorder=2)
        ax.plot([pd_max, pd_max], [0, SLA_DS], color=COLOR_PD,
                linestyle=":", linewidth=1.0, alpha=0.5, zorder=2)

        arrow_y = SLA_DS * 0.45
        ax.annotate("", xy=(pd_max, arrow_y), xytext=(agg_max, arrow_y),
                    arrowprops=dict(arrowstyle="<->", color=COLOR_WINNER,
                                    lw=2.2, shrinkA=2, shrinkB=2))

        mid_qps = (agg_max + pd_max) / 2
        ax.text(mid_qps, arrow_y - SLA_DS * 0.12,
                f"PD serves {mult:.1f}× more QPS\nunder SLA",
                ha="center", va="top", fontsize=9.5, fontweight="bold",
                color=COLOR_WINNER,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLOR_WINNER, alpha=0.9, linewidth=0.8))

        ax.text(agg_max, -SLA_DS * 0.12, f"Agg\n{agg_max:.1f}",
                ha="center", va="top", fontsize=8, color=COLOR_AGG,
                fontweight="bold")
        ax.text(pd_max, -SLA_DS * 0.12, f"PD\n{pd_max:.1f}",
                ha="center", va="top", fontsize=8, color=COLOR_PD,
                fontweight="bold")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_4_tpot_vs_qps_deepseek_hr60.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
