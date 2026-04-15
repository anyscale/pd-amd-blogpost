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
# Figure 1 — Hero Bar Chart: QPS Capacity Under SLA
# ===================================================================
def fig1_hero_bar():
    # Each scenario: (label, pd_config, pd_qps, agg_config, agg_qps, sla)
    # QPS = max sustainable QPS for the ENTIRE SERVICE (not per-GPU)
    data = [
        {
            "label": "Qwen3-235B\n24 GPU, ISL=16K\nOSL=1K, 0% HR",
            "sla": "TPOT < 25ms",
            "pd_config": "2P:1D TP8", "pd_qps": 4.0,
            "agg_config": "3×Agg TP8", "agg_qps": 1.5,
        },
        {
            "label": "Qwen3-235B\n24 GPU, ISL=8K\nOSL=1K, 0% HR",
            "sla": "TPOT < 25ms",
            "pd_config": "1P:2D TP8", "pd_qps": 5.0,
            "agg_config": "3×Agg TP8", "agg_qps": 3.5,
        },
        {
            "label": "DeepSeek-V3\n16 GPU, ISL=5.4K\nOSL=140, 30% HR",
            "sla": "TPOT < 30ms",
            "pd_config": "1P:1D TP8", "pd_qps": 7.0,
            "agg_config": "2×Agg TP8", "agg_qps": 3.0,
        },
        {
            "label": "DeepSeek-V3\n16 GPU, ISL=5.4K\nOSL=140, 60% HR",
            "sla": "TPOT < 30ms",
            "pd_config": "1P:1D TP8", "pd_qps": 7.0,
            "agg_config": "2×Agg TP8", "agg_qps": 5.0,
        },
        {   # PD loses case: 32 GPU with enough Agg replicas
            "label": "Qwen3-235B\n32 GPU, ISL=16K\nOSL=1K, 0% HR",
            "sla": "TPOT < 25ms",
            "pd_config": "2P:2D TP8", "pd_qps": 3.0,
            "agg_config": "4×Agg TP8", "agg_qps": 4.0,
        },
        {   # PD neutral case: high hit rate makes prefill cheap
            "label": "Qwen3-235B\n24 GPU, ISL=8K\nOSL=1K, 60% HR",
            "sla": "TPOT < 25ms",
            "pd_config": "1P:2D TP8", "pd_qps": 6.0,
            "agg_config": "3×Agg TP8", "agg_qps": 6.0,
        },
    ]

    n = len(data)
    x = np.arange(n)
    width = 0.30

    pd_qps = [d["pd_qps"] for d in data]
    agg_qps = [d["agg_qps"] for d in data]

    fig, ax = plt.subplots(figsize=(16, 7))

    bars_pd = ax.bar(x - width / 2, pd_qps, width, color=COLOR_PD,
                      edgecolor="white", linewidth=0.5, zorder=3)
    bars_agg = ax.bar(x + width / 2, agg_qps, width, color=COLOR_AGG,
                       edgecolor="white", linewidth=0.5, zorder=3)

    # Label bars with config name + QPS value
    for i, bar in enumerate(bars_pd):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{data[i]['pd_config']}\n{pd_qps[i]:.1f} QPS",
                ha="center", va="bottom", fontsize=8, color=COLOR_PD,
                fontweight="medium")
    for i, bar in enumerate(bars_agg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{data[i]['agg_config']}\n{agg_qps[i]:.1f} QPS",
                ha="center", va="bottom", fontsize=8, color=COLOR_AGG,
                fontweight="medium")

    # Dashed extension lines + double-headed arrow showing gap
    for i in range(n):
        mult = pd_qps[i] / agg_qps[i]
        pd_wins = mult >= 1.0
        color = COLOR_WINNER if pd_wins else "#d62728"  # green or red
        label = f"{mult:.1f}×" if pd_wins else f"{1/mult:.1f}× Agg"

        lo = min(pd_qps[i], agg_qps[i])
        hi = max(pd_qps[i], agg_qps[i])

        # Right edge of the agg bar (rightmost bar)
        agg_right = x[i] + width / 2 + width * 0.02
        # Where the annotation bracket sits
        bracket_x = x[i] + width / 2 + width * 0.55

        # Dashed horizontal lines from each bar top to the bracket
        ax.plot([agg_right, bracket_x + 0.05], [hi, hi],
                color=color, linestyle=":", linewidth=1.0, alpha=0.6, zorder=2)
        ax.plot([agg_right, bracket_x + 0.05], [lo, lo],
                color=color, linestyle=":", linewidth=1.0, alpha=0.6, zorder=2)

        # Double-headed arrow between hi and lo at bracket_x
        ax.annotate("", xy=(bracket_x, hi), xytext=(bracket_x, lo),
                     arrowprops=dict(arrowstyle="<->", color=color, lw=1.8,
                                     shrinkA=1, shrinkB=1))

        # Multiplier label next to the arrow
        mid_y = (lo + hi) / 2
        ax.text(bracket_x + 0.13, mid_y, label,
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.8))

    # X-axis: scenario + SLA
    labels = [f"{d['label']}\n({d['sla']})" for d in data]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.1)

    ax.set_ylabel("Max Sustainable QPS Under SLA\n(entire service, not per-GPU)", fontsize=11)
    ax.set_title("PD vs Aggregated: Max Sustainable QPS Under SLA (Same GPU Count)",
                  fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_PD, label="PD (best config)"),
        Patch(facecolor=COLOR_AGG, label="Aggregated"),
        Line2D([0], [0], color=COLOR_WINNER, lw=2, label="PD wins"),
        Line2D([0], [0], color="#d62728", lw=2, label="Agg wins"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.set_ylim(0, max(max(pd_qps), max(agg_qps)) * 1.35)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_1_hero_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


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

    # --- Right panel: DeepSeek-V3 (re-run data, current cluster) ---
    qps_agg2 = [3.0, 4.0, 4.5, 5.0, 6.0, 7.0]
    ttft_agg2 = [260.2, 256.1, 255.9, 255.2, 276.8, 268.9]
    qps_pd2 = [3.0, 4.0, 4.5, 5.0, 6.0, 7.0]
    ttft_pd2 = [331.5, 331.2, 326.5, 339.9, 721.3, 2006.2]

    ax_right.plot(qps_agg2, ttft_agg2, color=COLOR_AGG, marker=MARKER_AGG,
                  linewidth=2, markersize=7, label="Agg 2×Agg TP8", zorder=3)
    ax_right.plot(qps_pd2, ttft_pd2, color=COLOR_PD, marker=MARKER_PD,
                  linewidth=2, markersize=7, label="PD 1P:1D TP8", zorder=3)
    ax_right.set_yscale("log")
    ax_right.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_right.set_xlabel("QPS")
    ax_right.set_ylabel("TTFT (ms)")
    ax_right.set_title("DeepSeek-V3 on MI325X (30% HR)")
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

    SLA_QWEN = 25   # Qwen SLA: TPOT < 25ms
    SLA_DS = 30     # DeepSeek SLA: TPOT < 30ms

    # --- Left panel: Qwen3-235B (ISL=16K, OSL=1K) — 24 GPU (2P1D vs 3Agg) ---
    qps_agg = [1.0, 2.0, 3.0, 4.0]
    tpot_agg = [23.1, 25.5, 27.6, 29.9]
    qps_pd = [1.0, 2.0, 3.0, 4.0]
    tpot_pd = [19.5, 22.8, 24.2, 24.8]

    ax_left.plot(qps_agg, tpot_agg, color=COLOR_AGG, marker=MARKER_AGG,
                 linewidth=2.5, markersize=8, label="Agg 3Agg TP8", zorder=3)
    ax_left.plot(qps_pd, tpot_pd, color=COLOR_PD, marker=MARKER_PD,
                 linewidth=2.5, markersize=8, label="PD 2P1D TP8", zorder=3)
    ax_left.axhline(SLA_QWEN, color="red", linestyle="--", linewidth=1.2,
                     alpha=0.7, label=f"SLA = {SLA_QWEN} ms", zorder=2)
    ax_left.set_xlabel("QPS")
    ax_left.set_ylabel("TPOT (ms)")
    ax_left.set_title("Qwen3-235B  (ISL=16K, OSL=1K, 24 GPU)")
    ax_left.legend(loc="upper left")
    # Ensure y-axis starts near 0 to emphasize the divergence
    ax_left.set_ylim(bottom=0, top=max(tpot_agg) * 1.15)

    # --- Right panel: DeepSeek-V3 (ISL=5.4K, OSL=140, 30% HR) ---
    qps_agg2 = [3.0, 4.0, 5.0, 6.0, 7.0]
    tpot_agg2 = [23.7, 30.2, 35.1, 43.6, 50.6]
    qps_pd2 = [3.0, 4.0, 5.0, 6.0, 7.0]
    tpot_pd2 = [22.6, 23.2, 26.1, 26.8, 27.3]

    ax_right.plot(qps_agg2, tpot_agg2, color=COLOR_AGG, marker=MARKER_AGG,
                  linewidth=2.5, markersize=8, label="Agg 2Agg TP8", zorder=3)
    ax_right.plot(qps_pd2, tpot_pd2, color=COLOR_PD, marker=MARKER_PD,
                  linewidth=2.5, markersize=8, label="PD 1P1D TP8", zorder=3)
    ax_right.axhline(SLA_DS, color="red", linestyle="--", linewidth=1.2,
                      alpha=0.7, label=f"SLA = {SLA_DS} ms", zorder=2)
    ax_right.set_xlabel("QPS")
    ax_right.set_ylabel("TPOT (ms)")
    ax_right.set_title("DeepSeek-V3  (ISL=5.4K, OSL=140, 30% HR)")
    ax_right.legend(loc="upper left")
    ax_right.set_ylim(bottom=0, top=max(tpot_agg2) * 1.15)

    # Add shaded region above SLA to visually emphasize the death spiral
    for ax, qps_a, tpot_a, sla in [(ax_left, qps_agg, tpot_agg, SLA_QWEN),
                                     (ax_right, qps_agg2, tpot_agg2, SLA_DS)]:
        above = [(q, t) for q, t in zip(qps_a, tpot_a) if t > sla]
        if len(above) >= 2:
            qs, ts = zip(*above)
            ax.fill_between(qs, sla, ts, color="red", alpha=0.08, zorder=1)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_4_tpot_vs_qps.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 5 — Scaling Curve: QPS vs GPU Count
# ===================================================================
def fig5_scaling_curve():
    gpu_counts = [16, 24, 32]

    # All data from current cluster (vLLM 0.18.0, MI325X)
    # SLA: TPOT < 25ms AND TTFT < 3000ms (tight SLA where PD advantage shows)
    # Best PD config at each GPU count
    pd_qps  = [2.0, 4.0, 3.0]
    pd_labels = ["1P1D", "2P1D", "2P2D"]

    # Aggregated
    agg_qps = [1.5, 1.5, 4.0]
    agg_labels = ["2Agg", "3Agg", "4Agg"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(gpu_counts, pd_qps, color=COLOR_PD, marker=MARKER_PD,
            linewidth=2.5, markersize=9, label="PD (best config)", zorder=3)
    ax.plot(gpu_counts, agg_qps, color=COLOR_AGG, marker=MARKER_AGG,
            linewidth=2.5, markersize=9, label="Aggregated", zorder=3)

    # Annotate PD data points
    offsets_pd = [(8, 12), (8, 12), (8, -22)]
    for i, (g, q) in enumerate(zip(gpu_counts, pd_qps)):
        ax.annotate(f"{q:.1f} QPS\n({pd_labels[i]})",
                    xy=(g, q), xytext=offsets_pd[i],
                    textcoords="offset points", fontsize=9,
                    color=COLOR_PD, fontweight="medium",
                    arrowprops=dict(arrowstyle="-", color=COLOR_PD, lw=0.5))

    # Annotate Agg data points
    offsets_agg = [(8, -22), (-60, -22), (8, 12)]
    for i, (g, q) in enumerate(zip(gpu_counts, agg_qps)):
        ax.annotate(f"{q:.1f} QPS\n({agg_labels[i]})",
                    xy=(g, q), xytext=offsets_agg[i],
                    textcoords="offset points", fontsize=9,
                    color=COLOR_AGG, fontweight="medium",
                    arrowprops=dict(arrowstyle="-", color=COLOR_AGG, lw=0.5))

    # Add advantage annotations
    for i, g in enumerate(gpu_counts):
        if pd_qps[i] > agg_qps[i]:
            mult = pd_qps[i] / agg_qps[i]
            mid_y = (pd_qps[i] + agg_qps[i]) / 2
            ax.text(g - 1.5, mid_y, f"{mult:.1f}x", fontsize=10,
                    fontweight="bold", color=COLOR_PD, ha="right")

    ax.set_xticks(gpu_counts)
    ax.set_xlabel("GPU Count")
    ax.set_ylabel("Max QPS Under SLA (TPOT < 25ms)")
    ax.set_title("QPS Capacity Scaling: PD vs Aggregated\n(ISL=16K, OSL=1K, Qwen3-235B, MI325X)")
    ax.legend(loc="upper left", fontsize=11)
    ax.set_ylim(0, max(max(pd_qps), max(agg_qps)) * 1.3)
    ax.set_xlim(14, 34)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_5_scaling_curve.png")
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
    fig1_hero_bar()
    fig3_ttft_vs_qps()
    fig4_tpot_vs_qps()
    fig5_scaling_curve()
    fig6_wrong_pd_ratio()
    print("\nAll figures generated successfully.")
