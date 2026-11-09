"""Visualization module for RANSAC line fitting and OLS comparison.

Generates presentation-ready Matplotlib figures saved as high-resolution PNGs:
1. ransac_vs_ols_fitting.png: Scatter plot with inliers/outliers and fitted lines.
2. comparison_summary.png: Side-by-side comparative line trajectories and error bar chart.
"""

import os
from typing import Optional
import matplotlib
# Use non-interactive backend for headless execution in Linux terminal
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_generator import Dataset
from line_fitting import LineModel
from metrics import LineFittingMetrics


def plot_line_fitting(
    dataset: Dataset,
    ols_model: LineModel,
    ransac_model: LineModel,
    output_path: str = "ransac_vs_ols_fitting.png",
    show_plot: bool = False,
) -> str:
    """Generate and save the main scatter and line fitting visualization.

    Displays:
    - Ground-truth inliers and outliers with distinct markers and colors.
    - Ground-truth line (dashed black).
    - Ordinary Least Squares line (dash-dotted red/orange).
    - RANSAC estimated line (solid green).

    Parameters:
        dataset: Dataset holding points and ground truth.
        ols_model: Fitted Ordinary Least Squares model.
        ransac_model: Fitted RANSAC model.
        output_path: File path to save the generated PNG.
        show_plot: Whether to invoke plt.show() (default False for headless mode).

    Returns:
        Absolute file path of the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # 1. Plot Inliers and Outliers
    inliers = dataset.inliers
    outliers = dataset.outliers

    ax.scatter(
        inliers[:, 0],
        inliers[:, 1],
        c="#2563eb",
        marker="o",
        s=35,
        alpha=0.75,
        edgecolors="none",
        label=f"Inliers (N={len(inliers)})",
        zorder=3,
    )

    if len(outliers) > 0:
        ax.scatter(
            outliers[:, 0],
            outliers[:, 1],
            c="#ef4444",
            marker="x",
            s=45,
            linewidths=1.5,
            alpha=0.85,
            label=f"Outliers (N={len(outliers)})",
            zorder=4,
        )

    # 2. Plot Lines across domain
    x_min, x_max = dataset.x_range
    padding = (x_max - x_min) * 0.05
    x_line = np.linspace(x_min - padding, x_max + padding, 200)

    y_gt = dataset.true_slope * x_line + dataset.true_intercept
    y_ols = ols_model.predict(x_line)
    y_ransac = ransac_model.predict(x_line)

    ax.plot(
        x_line,
        y_gt,
        color="#111827",
        linestyle="--",
        linewidth=2.2,
        label=f"Ground Truth (y = {dataset.true_slope:.2f}x + {dataset.true_intercept:.2f})",
        zorder=5,
    )

    ax.plot(
        x_line,
        y_ols,
        color="#ea580c",
        linestyle="-.",
        linewidth=2.2,
        label=f"Ordinary Least Squares (y = {ols_model.slope:.2f}x + {ols_model.intercept:.2f})",
        zorder=6,
    )

    ax.plot(
        x_line,
        y_ransac,
        color="#16a34a",
        linestyle="-",
        linewidth=2.6,
        label=f"RANSAC Robust (y = {ransac_model.slope:.2f}x + {ransac_model.intercept:.2f})",
        zorder=7,
    )

    # Styling and formatting
    ax.set_title("Robust Line Fitting: RANSAC vs. Ordinary Least Squares (OLS)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X coordinate", fontsize=11, labelpad=8)
    ax.set_ylabel("Y coordinate", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6, color="#9ca3af")
    ax.set_xlim(x_min - padding, x_max + padding)

    # Highlight RANSAC inlier count in legend
    inlier_str = f"RANSAC consensus inliers: {ransac_model.inlier_count}/{dataset.n_points}"
    ax.legend(loc="upper left", framealpha=0.92, fancybox=True, shadow=True, fontsize=10)

    # Informational annotation box
    ax.text(
        0.98,
        0.04,
        f"{inlier_str}\nOutliers: {len(outliers)} ({len(outliers)/dataset.n_points*100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#d1d5db", alpha=0.9),
    )

    plt.tight_layout()

    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True) if os.path.dirname(out_abs) else None
    plt.savefig(out_abs, dpi=150)

    if show_plot:
        plt.show()

    plt.close(fig)
    return out_abs


def plot_comparison_summary(
    dataset: Dataset,
    ols_metrics: LineFittingMetrics,
    ransac_metrics: LineFittingMetrics,
    output_path: str = "comparison_summary.png",
    show_plot: bool = False,
) -> str:
    """Generate a multi-panel comparison figure showing trajectories and error bars.

    Subplot 1: Isolated comparison of Ground Truth, OLS, and RANSAC lines.
    Subplot 2: Bar chart comparing Slope Error, Intercept Error, and Line RMSE.

    Parameters:
        dataset: Dataset holding domain range and true parameters.
        ols_metrics: Computed metrics for OLS.
        ransac_metrics: Computed metrics for RANSAC.
        output_path: File path to save the generated PNG.
        show_plot: Whether to invoke plt.show().

    Returns:
        Absolute file path of the saved PNG.
    """
    fig, (ax_lines, ax_bars) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # --- Subplot 1: Line Trajectories ---
    x_min, x_max = dataset.x_range
    x_line = np.linspace(x_min, x_max, 200)

    y_gt = dataset.true_slope * x_line + dataset.true_intercept
    y_ols = ols_metrics.estimated_slope * x_line + ols_metrics.estimated_intercept
    y_ransac = ransac_metrics.estimated_slope * x_line + ransac_metrics.estimated_intercept

    ax_lines.plot(x_line, y_gt, "k--", linewidth=2.2, label="Ground Truth Line")
    ax_lines.plot(x_line, y_ols, color="#ea580c", linestyle="-.", linewidth=2.2, label="OLS (Outlier Distorted)")
    ax_lines.plot(x_line, y_ransac, color="#16a34a", linestyle="-", linewidth=2.5, label="RANSAC (Inlier Refined)")

    ax_lines.set_title("Estimated Line Trajectories", fontsize=12, fontweight="bold")
    ax_lines.set_xlabel("X coordinate", fontsize=10)
    ax_lines.set_ylabel("Y coordinate", fontsize=10)
    ax_lines.grid(True, linestyle=":", alpha=0.6)
    ax_lines.legend(loc="upper left", framealpha=0.9)

    # --- Subplot 2: Quantitative Error Metrics Bar Chart ---
    categories = ["Slope Error", "Intercept Error", "Line RMSE"]
    ols_values = [ols_metrics.slope_error, ols_metrics.intercept_error, ols_metrics.rmse]
    ransac_values = [ransac_metrics.slope_error, ransac_metrics.intercept_error, ransac_metrics.rmse]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax_bars.bar(x - width / 2, ols_values, width, label="Ordinary Least Squares", color="#f97316", alpha=0.85)
    rects2 = ax_bars.bar(x + width / 2, ransac_values, width, label="RANSAC Robust", color="#22c55e", alpha=0.85)

    ax_bars.set_title("Estimation Error Comparison", fontsize=12, fontweight="bold")
    ax_bars.set_ylabel("Error Value (Lower is Better)", fontsize=10)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(categories, fontsize=10)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.6)
    ax_bars.legend(loc="upper right", framealpha=0.9)

    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_bars.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True) if os.path.dirname(out_abs) else None
    plt.savefig(out_abs, dpi=150)

    if show_plot:
        plt.show()

    plt.close(fig)
    return out_abs

