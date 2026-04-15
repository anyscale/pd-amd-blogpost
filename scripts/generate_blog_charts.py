#!/usr/bin/env python3
"""Generate publication-quality charts for PD disaggregation AMD blog post."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
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

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR_PD = "#1f77b4"      # blue
COLOR_AGG = "#ff7f0e"      # orange
COLOR_WINNER = "#2ca02c"   # green
COLOR_NEUTRAL = "#888888"  # gray
MARKER_PD = "o"
MARKER_AGG = "s"


# ===================================================================
# Figure 3 — TTFT vs QPS (dual panel)
# ===================================================================
def fig3_ttft_vs_qps():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: Qwen3-235B ---
    qps_agg = [0.25, 1.0, 2.0, 3.0]
    ttft_agg = [560, 559, 557, 579]
    qps_pd = [0.25, 1.0, 2.0, 3.0]
    ttft_pd = [704, 697, 686, 7021]

    ax_left.plot(qps_agg, ttft_agg, color=COLOR_AGG, marker=MARKER_AGG,
                 linewidth=2, markersize=7, label="Agg TP8 (2 replicas)", zorder=3)
    ax_left.plot(qps_pd, ttft_pd, color=COLOR_PD, marker=MARKER_PD,
                 linewidth=2, markersize=7, label="PD 1P1D TP8", zorder=3)
    ax_left.set_yscale("log")
    ax_left.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_left.set_xlabel("QPS")
    ax_left.set_ylabel("TTFT (ms)")
    ax_left.set_title("Qwen3-235B on MI325X")
    ax_left.legend(loc="best")

    # --- Right panel: DeepSeek-V3 ---
    qps_agg2 = [3.0, 4.0, 4.5, 5.0, 6.0]
    ttft_agg2 = [253.7, 254.2, 268.9, 271.0, 281.8]
    qps_pd2 = [3.0, 4.0, 4.5, 5.0, 6.0]
    ttft_pd2 = [264.3, 260.4, 323.6, 262.9, 1521.2]

    ax_right.plot(qps_agg2, ttft_agg2, color=COLOR_AGG, marker=MARKER_AGG,
                  linewidth=2, markersize=7, label="Agg 2Agg TP8", zorder=3)
    ax_right.plot(qps_pd2, ttft_pd2, color=COLOR_PD, marker=MARKER_PD,
                  linewidth=2, markersize=7, label="PD 1P1D TP8", zorder=3)
    ax_right.set_yscale("log")
    ax_right.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_right.set_xlabel("QPS")
    ax_right.set_ylabel("TTFT (ms)")
    ax_right.set_title("DeepSeek-V3 on MI325X")
    ax_right.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_3_ttft_vs_qps.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 4 — TPOT vs QPS (dual panel) — THE MOST IMPORTANT CHART
# ===================================================================
def fig4_tpot_vs_qps():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    SLA_MS = 30  # reference SLA line

    # --- Left panel: Qwen3-235B (ISL=16K, OSL=1K) ---
    qps_agg = [0.25, 1.0, 2.0, 3.0]
    tpot_agg = [15.7, 23.2, 46.1, 70.4]
    qps_pd = [0.25, 1.0, 2.0, 3.0]
    tpot_pd = [15.8, 22.8, 28.5, 30.8]

    ax_left.plot(qps_agg, tpot_agg, color=COLOR_AGG, marker=MARKER_AGG,
                 linewidth=2.5, markersize=8, label="Agg (best, 32K budget)", zorder=3)
    ax_left.plot(qps_pd, tpot_pd, color=COLOR_PD, marker=MARKER_PD,
                 linewidth=2.5, markersize=8, label="PD 2P2D TP4", zorder=3)
    ax_left.axhline(SLA_MS, color="red", linestyle="--", linewidth=1.2,
                     alpha=0.7, label=f"SLA = {SLA_MS} ms", zorder=2)
    ax_left.set_xlabel("QPS")
    ax_left.set_ylabel("TPOT (ms)")
    ax_left.set_title("Qwen3-235B  (ISL=16K, OSL=1K)")
    ax_left.legend(loc="upper left")
    # Ensure y-axis starts near 0 to emphasize the divergence
    ax_left.set_ylim(bottom=0, top=max(tpot_agg) * 1.15)

    # --- Right panel: DeepSeek-V3 (ISL=5.4K, OSL=140, 30% HR) ---
    qps_agg2 = [3.0, 4.0, 4.5, 5.0, 6.0]
    tpot_agg2 = [24.14, 32.81, 35.33, 38.12, 49.95]
    qps_pd2 = [3.0, 4.0, 4.5, 5.0, 6.0]
    tpot_pd2 = [12.28, 11.88, 12.72, 12.60, 12.93]

    ax_right.plot(qps_agg2, tpot_agg2, color=COLOR_AGG, marker=MARKER_AGG,
                  linewidth=2.5, markersize=8, label="Agg 2Agg TP8", zorder=3)
    ax_right.plot(qps_pd2, tpot_pd2, color=COLOR_PD, marker=MARKER_PD,
                  linewidth=2.5, markersize=8, label="PD 1P1D TP8", zorder=3)
    ax_right.axhline(SLA_MS, color="red", linestyle="--", linewidth=1.2,
                      alpha=0.7, label=f"SLA = {SLA_MS} ms", zorder=2)
    ax_right.set_xlabel("QPS")
    ax_right.set_ylabel("TPOT (ms)")
    ax_right.set_title("DeepSeek-V3  (ISL=5.4K, OSL=140, 30% HR)")
    ax_right.legend(loc="upper left")
    ax_right.set_ylim(bottom=0, top=max(tpot_agg2) * 1.15)

    # Add shaded region above SLA to visually emphasize the death spiral
    for ax, qps_a, tpot_a in [(ax_left, qps_agg, tpot_agg),
                                (ax_right, qps_agg2, tpot_agg2)]:
        above = [(q, t) for q, t in zip(qps_a, tpot_a) if t > SLA_MS]
        if len(above) >= 2:
            qs, ts = zip(*above)
            ax.fill_between(qs, SLA_MS, ts, color="red", alpha=0.08, zorder=1)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_4_tpot_vs_qps.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 6 — Wrong P:D Ratio (horizontal bar chart)
# ===================================================================
def fig6_wrong_pd_ratio():
    configs = [
        ("3P:1D\n(32 GPU)", 227.0, "pd_bad"),
        ("2P:1D\n(24 GPU)", 205.3, "pd_bad"),
        ("1P:1D\n(16 GPU)", 192.2, "pd_bad"),
        ("4\u00d7Agg\n(32 GPU)", 135.8, "agg"),
        ("1P:3D\n(32 GPU)", 117.0, "pd_good"),
    ]

    # Sort worst (highest) at top
    configs.sort(key=lambda x: x[1], reverse=True)

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    kinds = [c[2] for c in configs]

    agg_value = 135.8  # Agg baseline

    # Color map
    # For the "pd_bad" bars, use a red-to-orange gradient based on severity
    bad_values = [v for v, k in zip(values, kinds) if k == "pd_bad"]
    bad_min = min(bad_values) if bad_values else 0
    bad_max = max(bad_values) if bad_values else 1

    colors = []
    for v, k in zip(values, kinds):
        if k == "agg":
            colors.append(COLOR_NEUTRAL)
        elif k == "pd_good":
            colors.append(COLOR_WINNER)
        else:
            # Gradient from orange (least bad) to red (worst)
            if bad_max == bad_min:
                frac = 0.5
            else:
                frac = (v - bad_min) / (bad_max - bad_min)
            r = 0.9 + 0.1 * frac
            g = 0.45 * (1 - frac)
            b = 0.1 * (1 - frac)
            colors.append((r, g, b))

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5,
                   height=0.6, zorder=3)

    # Agg baseline vertical dashed line
    ax.axvline(agg_value, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Agg baseline ({agg_value:.0f}s)", zorder=2)

    # Annotate each bar with percentage vs Agg
    for i, (val, bar) in enumerate(zip(values, bars)):
        pct = (val - agg_value) / agg_value * 100
        sign = "+" if pct > 0 else ""
        if pct == 0:
            label_text = f"  {val:.1f}s (baseline)"
        else:
            label_text = f"  {val:.1f}s ({sign}{pct:.0f}% vs Agg)"
        ax.text(val + 2, i, label_text, va="center", fontsize=10, fontweight="medium")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("End-to-End Latency (s)")
    ax.set_title("Wrong P:D Ratio Can Be Worse Than Aggregated\n"
                 "(QPS=1.5, ISL=16K, OSL=4K)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, max(values) * 1.35)
    ax.invert_yaxis()  # worst at top

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_6_wrong_pd_ratio.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    fig3_ttft_vs_qps()
    fig4_tpot_vs_qps()
    fig6_wrong_pd_ratio()
    print("\nAll figures generated successfully.")
