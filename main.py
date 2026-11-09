"""Main command-line entry point for RANSAC line fitting and outlier handling project.

Demonstrates robust geometric estimation by comparing Ordinary Least Squares (OLS)
against manual RANSAC on noisy 2D data containing significant outliers.
"""

import argparse
import os
import sys
import time

from data_generator import generate_line_data
from line_fitting import fit_ols, fit_ransac
from metrics import compare_models
from visualization import plot_line_fitting, plot_comparison_summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with sensible college-project defaults."""
    parser = argparse.ArgumentParser(
        description="Robust Line Fitting and Outlier Handling using the RANSAC Algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inliers", type=int, default=100, help="Number of linear inlier points")
    parser.add_argument("--outliers", type=int, default=40, help="Number of dispersed outlier points")
    parser.add_argument("--slope", type=float, default=2.0, help="Ground-truth line slope (m)")
    parser.add_argument("--intercept", type=float, default=5.0, help="Ground-truth line intercept (c)")
    parser.add_argument("--noise", type=float, default=1.0, help="Standard deviation of Gaussian noise added to inliers")
    parser.add_argument("--threshold", type=float, default=2.5, help="RANSAC perpendicular distance threshold (tau)")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum RANSAC iterations to execute (N)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save generated comparison charts")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating and saving visualization figures")
    return parser.parse_args()


def print_banner() -> None:
    """Display program header."""
    print("=" * 76)
    print("  ROBUST LINE FITTING & OUTLIER HANDLING USING THE RANSAC ALGORITHM")
    print("  College-Level Classical Computer Vision Project")
    print("=" * 76)


def print_summary_table(ols_metrics, ransac_metrics, dataset, ransac_model, ransac_time_ms: float) -> None:
    """Print a clean ASCII table comparing OLS and RANSAC performance."""
    print("\n--- EXPERIMENTAL CONFIGURATION ---")
    print(f"  Ground-Truth Line:      y = {dataset.true_slope:.2f}x + {dataset.true_intercept:.2f}")
    print(f"  Total Data Points:      {dataset.n_points} ({dataset.n_inliers} inliers, {dataset.n_outliers} outliers)")
    outlier_pct = (dataset.n_outliers / dataset.n_points) * 100.0
    print(f"  Outlier Contamination:  {outlier_pct:.1f}%")
    print(f"  RANSAC Iterations:      {ransac_model.n_iterations} ({ransac_time_ms:.2f} ms execution)")

    print("\n--- QUANTITATIVE COMPARISON TABLE ---")
    header = f"{'Metric':<30} | {'Ordinary Least Squares':<22} | {'RANSAC (Robust)':<20}"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    print(f"{'Estimated Slope (m)':<30} | {ols_metrics.estimated_slope:<22.4f} | {ransac_metrics.estimated_slope:<20.4f}")
    print(f"{'Slope Absolute Error':<30} | {ols_metrics.slope_error:<22.4f} | {ransac_metrics.slope_error:<20.4f}")
    print(f"{'Estimated Intercept (c)':<30} | {ols_metrics.estimated_intercept:<22.4f} | {ransac_metrics.estimated_intercept:<20.4f}")
    print(f"{'Intercept Absolute Error':<30} | {ols_metrics.intercept_error:<22.4f} | {ransac_metrics.intercept_error:<20.4f}")
    print(f"{'Domain Prediction MAE':<30} | {ols_metrics.mae:<22.4f} | {ransac_metrics.mae:<20.4f}")
    print(f"{'Domain Prediction RMSE':<30} | {ols_metrics.rmse:<22.4f} | {ransac_metrics.rmse:<20.4f}")
    print(f"{'Detected Inliers':<30} | {f'{ols_metrics.detected_inliers}/{dataset.n_points}':<22} | {f'{ransac_metrics.detected_inliers}/{dataset.n_points}':<20}")

    if ransac_metrics.inlier_precision is not None:
        prec_str = f"{ransac_metrics.inlier_precision * 100:.1f}%"
        rec_str = f"{ransac_metrics.inlier_recall * 100:.1f}%"
        f1_str = f"{ransac_metrics.inlier_f1 * 100:.1f}%"
        acc_str = f"{ransac_metrics.classification_accuracy * 100:.1f}%"
        print(f"{'Inlier Classification Precision':<30} | {'N/A (All Inliers)':<22} | {prec_str:<20}")
        print(f"{'Inlier Classification Recall':<30} | {'100.0%':<22} | {rec_str:<20}")
        print(f"{'Inlier Classification F1-Score':<30} | {'N/A':<22} | {f1_str:<20}")
        print(f"{'Overall Classification Accuracy':<30} | {f'{dataset.n_inliers/dataset.n_points*100:.1f}%':<22} | {acc_str:<20}")

    print(divider)


def main() -> int:
    """Execute the line fitting pipeline."""
    args = parse_arguments()
    print_banner()

    # 1. Synthesize Dataset
    print(f"\n[1/4] Generating synthetic dataset with {args.inliers} inliers and {args.outliers} outliers...")
    dataset = generate_line_data(
        n_inliers=args.inliers,
        n_outliers=args.outliers,
        slope=args.slope,
        intercept=args.intercept,
        noise_std=args.noise,
        seed=args.seed,
    )

    # 2. Fit Ordinary Least Squares baseline
    print("[2/4] Fitting baseline using Ordinary Least Squares (OLS)...")
    ols_model = fit_ols(dataset.points)

    # 3. Fit RANSAC model
    print(f"[3/4] Fitting robust model using manual RANSAC ({args.iterations} iterations, threshold {args.threshold})...")
    t0 = time.perf_counter()
    ransac_model = fit_ransac(
        dataset.points,
        distance_threshold=args.threshold,
        max_iterations=args.iterations,
        seed=args.seed,
    )
    ransac_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Evaluate metrics
    print("[4/4] Evaluating estimation metrics against ground truth...")
    ols_metrics, ransac_metrics = compare_models(ols_model, ransac_model, dataset)

    # Display console comparison
    print_summary_table(ols_metrics, ransac_metrics, dataset, ransac_model, ransac_time_ms)

    # Key observation note
    print("\n--- KEY OBSERVATION ---")
    print("  OLS minimizes squared vertical errors across all points; a few distant outliers")
    print("  drastically leverage the line away from the genuine inlier distribution.")
    print("  In contrast, RANSAC maximizes inlier consensus using perpendicular distances,")
    print("  achieving an accurate line fit even with heavy outlier contamination.")

    # 5. Generate and save plots
    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        path1 = os.path.join(args.output_dir, "ransac_vs_ols_fitting.png")
        path2 = os.path.join(args.output_dir, "comparison_summary.png")

        print("\n--- GENERATED VISUALIZATIONS ---")
        saved_p1 = plot_line_fitting(dataset, ols_model, ransac_model, output_path=path1)
        print(f"  [1] Scatter & Fitted Lines: {saved_p1}")

        saved_p2 = plot_comparison_summary(dataset, ols_metrics, ransac_metrics, output_path=path2)
        print(f"  [2] Comparison Error Summary: {saved_p2}")

    print("\n[Done] Pipeline execution completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
