"""Test suite for RANSAC line fitting and outlier handling project."""

import numpy as np
import pytest

from data_generator import Dataset, generate_line_data
from line_fitting import LineModel, compute_line_from_two_points, fit_ols, fit_ransac


class TestDataGenerator:
    """Unit tests for synthetic dataset generation."""

    def test_dataset_shapes_and_counts(self):
        """Verify generated dataset dimensions, point counts, and inlier mask consistency."""
        n_inliers = 80
        n_outliers = 30
        data = generate_line_data(
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            slope=2.0,
            intercept=5.0,
            noise_std=1.0,
            seed=42,
        )

        assert isinstance(data, Dataset)
        assert data.points.shape == (110, 2)
        assert data.inlier_mask.shape == (110,)
        assert data.inlier_mask.dtype == bool
        assert data.n_points == 110
        assert data.n_inliers == n_inliers
        assert data.n_outliers == n_outliers
        assert len(data.inliers) == n_inliers
        assert len(data.outliers) == n_outliers

    def test_dataset_reproducibility(self):
        """Verify identical dataset generation when using the same random seed."""
        data1 = generate_line_data(n_inliers=50, n_outliers=20, seed=123)
        data2 = generate_line_data(n_inliers=50, n_outliers=20, seed=123)

        np.testing.assert_array_equal(data1.points, data2.points)
        np.testing.assert_array_equal(data1.inlier_mask, data2.inlier_mask)
        assert data1.true_slope == data2.true_slope
        assert data1.true_intercept == data2.true_intercept

    def test_dataset_different_seeds(self):
        """Verify distinct datasets when using different random seeds."""
        data1 = generate_line_data(n_inliers=50, n_outliers=20, seed=10)
        data2 = generate_line_data(n_inliers=50, n_outliers=20, seed=20)

        assert not np.array_equal(data1.points, data2.points)

    def test_inlier_closeness_to_line(self):
        """Verify that inliers lie closely to y = m*x + c within Gaussian noise bounds."""
        slope = 3.0
        intercept = -4.0
        noise_std = 0.5
        data = generate_line_data(
            n_inliers=200,
            n_outliers=10,
            slope=slope,
            intercept=intercept,
            noise_std=noise_std,
            seed=99,
        )

        inliers = data.inliers
        x = inliers[:, 0]
        y = inliers[:, 1]
        residuals = np.abs(y - (slope * x + intercept))

        # 99.7% of Gaussian samples fall within 3*sigma
        assert np.mean(residuals) < 2.0 * noise_std
        assert np.max(residuals) < 4.5 * noise_std

    def test_outlier_separation(self):
        """Verify outliers are significantly displaced from the line."""
        slope = 2.0
        intercept = 5.0
        noise_std = 1.0
        data = generate_line_data(
            n_inliers=100,
            n_outliers=50,
            slope=slope,
            intercept=intercept,
            noise_std=noise_std,
            seed=42,
        )

        outliers = data.outliers
        x_out = outliers[:, 0]
        y_out = outliers[:, 1]
        residuals = np.abs(y_out - (slope * x_out + intercept))

        # Each outlier was forced to have residual >= max(3*sigma, 2.0)
        assert np.all(residuals >= 2.0)

    def test_zero_outliers(self):
        """Verify that generating a dataset with zero outliers works cleanly."""
        data = generate_line_data(n_inliers=50, n_outliers=0, seed=42)

        assert data.n_points == 50
        assert data.n_inliers == 50
        assert data.n_outliers == 0
        assert np.all(data.inlier_mask)
        assert len(data.outliers) == 0

    @pytest.mark.parametrize(
        "kwargs, exc_msg",
        [
            ({"n_inliers": 1}, "At least 2 inliers"),
            ({"n_outliers": -5}, "Outlier count cannot be negative"),
            ({"noise_std": -0.5}, "Noise standard deviation must be non-negative"),
            ({"x_range": (10.0, 5.0)}, r"x_range\[0\] must be strictly less than"),
            ({"x_range": (5.0, 5.0)}, r"x_range\[0\] must be strictly less than"),
        ],
    )
    def test_invalid_parameters_raise(self, kwargs, exc_msg):
        """Verify parameter validation for invalid arguments."""
        with pytest.raises(ValueError, match=exc_msg):
            generate_line_data(**kwargs)


class TestLineModel:
    """Unit tests for LineModel representation, prediction, and orthogonal distance."""

    def test_predict_and_perpendicular_distance(self):
        """Verify perpendicular distance formula against hand-calculated analytical values."""
        # For line y = x (slope=1, intercept=0, standard form x - y = 0)
        # Distance from (0, 2) is |1*(0) - 2 + 0| / sqrt(1^2 + (-1)^2) = 2 / sqrt(2) = sqrt(2)
        model = LineModel(slope=1.0, intercept=0.0)
        points = np.array([
            [0.0, 2.0],   # Distance sqrt(2) ~= 1.41421356
            [2.0, 0.0],   # Distance sqrt(2) ~= 1.41421356
            [1.0, 1.0],   # Point on line, distance 0.0
            [5.0, 5.0],   # Point on line, distance 0.0
            [0.0, -2.0],  # Distance sqrt(2) ~= 1.41421356
        ])

        distances = model.perpendicular_distance(points)
        expected = np.array([np.sqrt(2.0), np.sqrt(2.0), 0.0, 0.0, np.sqrt(2.0)])
        np.testing.assert_allclose(distances, expected, rtol=1e-6)

        preds = model.predict(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_allclose(preds, np.array([0.0, 5.0, 10.0]))

    def test_invalid_points_shape_raises(self):
        """Verify ValueError is raised if points array does not have shape (N, 2)."""
        model = LineModel(slope=1.0, intercept=0.0)
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            model.perpendicular_distance(np.array([1.0, 2.0, 3.0]))


class TestOLS:
    """Unit tests for Ordinary Least Squares baseline."""

    def test_ols_clean_line_recovery(self):
        """Verify OLS recovers exact slope and intercept on noise-free colinear data."""
        x = np.linspace(0.0, 10.0, 25)
        y = 2.5 * x - 3.5
        points = np.column_stack((x, y))

        model = fit_ols(points)
        assert pytest.approx(model.slope, rel=1e-6) == 2.5
        assert pytest.approx(model.intercept, rel=1e-6) == -3.5
        assert model.inlier_count == 25
        assert np.all(model.inlier_mask)

    def test_ols_sensitivity_to_outliers(self):
        """Demonstrate that OLS is strongly corrupted by outliers due to quadratic penalty."""
        data = generate_line_data(
            n_inliers=100,
            n_outliers=40,
            slope=2.0,
            intercept=5.0,
            noise_std=1.0,
            seed=42,
        )
        model = fit_ols(data.points)

        slope_err = abs(model.slope - 2.0)
        intercept_err = abs(model.intercept - 5.0)

        # OLS should deviate noticeably due to the 40 dispersed outliers
        assert slope_err > 0.15 or intercept_err > 1.5

    def test_ols_insufficient_points_raises(self):
        """Verify OLS raises ValueError with fewer than 2 points."""
        with pytest.raises(ValueError, match="At least 2 points"):
            fit_ols(np.array([[1.0, 2.0]]))

    def test_ols_vertical_line_raises(self):
        """Verify OLS raises ValueError on vertically aligned points (division by zero variance)."""
        vertical_points = np.array([[4.0, 1.0], [4.0, 5.0], [4.0, 9.0]])
        with pytest.raises(ValueError, match="vertical line"):
            fit_ols(vertical_points)


class TestRANSAC:
    """Unit tests for manual RANSAC line fitting algorithm."""

    def test_ransac_without_outliers(self):
        """Verify RANSAC accurately estimates line when outliers are absent."""
        data = generate_line_data(
            n_inliers=100,
            n_outliers=0,
            slope=2.0,
            intercept=5.0,
            noise_std=0.5,
            seed=42,
        )
        model = fit_ransac(data.points, distance_threshold=2.0, max_iterations=100, seed=42)

        assert abs(model.slope - 2.0) < 0.05
        assert abs(model.intercept - 5.0) < 0.3
        assert model.inlier_count is not None and model.inlier_count >= 95

    def test_ransac_outperforms_ols_with_outliers(self):
        """Verify RANSAC substantially outperforms OLS when 28.5% outliers are present."""
        data = generate_line_data(
            n_inliers=100,
            n_outliers=40,
            slope=2.0,
            intercept=5.0,
            noise_std=1.0,
            seed=42,
        )

        ransac_model = fit_ransac(data.points, distance_threshold=2.5, max_iterations=150, seed=42)
        ols_model = fit_ols(data.points)

        ransac_slope_err = abs(ransac_model.slope - 2.0)
        ransac_intercept_err = abs(ransac_model.intercept - 5.0)

        ols_slope_err = abs(ols_model.slope - 2.0)
        ols_intercept_err = abs(ols_model.intercept - 5.0)

        # RANSAC slope and intercept errors must be tight and much smaller than OLS
        assert ransac_slope_err < 0.08
        assert ransac_intercept_err < 0.8
        assert ransac_slope_err < 0.5 * ols_slope_err
        assert ransac_intercept_err < 0.5 * ols_intercept_err

    def test_ransac_inlier_classification_accuracy(self):
        """Verify RANSAC detects inliers with high precision and recall against ground truth."""
        data = generate_line_data(
            n_inliers=100,
            n_outliers=40,
            slope=2.0,
            intercept=5.0,
            noise_std=1.0,
            seed=42,
        )
        model = fit_ransac(data.points, distance_threshold=2.5, max_iterations=100, seed=42)

        true_inliers = data.inlier_mask
        pred_inliers = model.inlier_mask
        assert pred_inliers is not None

        tp = np.sum(true_inliers & pred_inliers)
        fp = np.sum((~true_inliers) & pred_inliers)
        fn = np.sum(true_inliers & (~pred_inliers))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        assert recall >= 0.90
        assert precision >= 0.90

    def test_ransac_deterministic_with_seed(self):
        """Verify RANSAC yields identical results when executed with the same random seed."""
        data = generate_line_data(n_inliers=80, n_outliers=30, seed=42)

        m1 = fit_ransac(data.points, distance_threshold=2.5, max_iterations=100, seed=777)
        m2 = fit_ransac(data.points, distance_threshold=2.5, max_iterations=100, seed=777)

        assert m1.slope == m2.slope
        assert m1.intercept == m2.intercept
        np.testing.assert_array_equal(m1.inlier_mask, m2.inlier_mask)
        assert m1.inlier_count == m2.inlier_count

    def test_ransac_handles_degenerate_point_samples(self):
        """Verify RANSAC handles degenerate points (e.g., repeated identical points) without crashing."""
        identical_pts = np.array([[5.0, 15.0]] * 15)
        clean_pts = np.column_stack((np.linspace(0.0, 10.0, 30), 2.0 * np.linspace(0.0, 10.0, 30) + 5.0))
        pts = np.vstack((identical_pts, clean_pts))

        # Should handle sampling degenerate duplicate points without throwing exceptions
        model = fit_ransac(pts, distance_threshold=2.0, max_iterations=60, seed=42)
        assert isinstance(model, LineModel)
        assert model.inlier_count is not None and model.inlier_count > 0

    def test_compute_line_from_two_points_degeneracy(self):
        """Verify compute_line_from_two_points returns None for coincident points."""
        p1 = np.array([3.0, 4.0])
        p2 = np.array([3.0, 4.0])
        res = compute_line_from_two_points(p1, p2)
        assert res is None

    @pytest.mark.parametrize(
        "kwargs, exc_msg",
        [
            ({"points": np.array([[1.0, 2.0]])}, "At least 2 points"),
            ({"points": np.array([[1.0, 2.0], [3.0, 4.0]]), "distance_threshold": 0.0}, "distance_threshold must be strictly positive"),
            ({"points": np.array([[1.0, 2.0], [3.0, 4.0]]), "distance_threshold": -1.0}, "distance_threshold must be strictly positive"),
            ({"points": np.array([[1.0, 2.0], [3.0, 4.0]]), "max_iterations": 0}, "max_iterations must be at least 1"),
        ],
    )
    def test_ransac_invalid_parameters_raise(self, kwargs, exc_msg):
        """Verify parameter validation in fit_ransac."""
        with pytest.raises(ValueError, match=exc_msg):
            fit_ransac(**kwargs)


class TestMetrics:
    """Unit tests for evaluation metrics and OLS vs RANSAC comparison."""

    def test_evaluate_model_perfect_fit(self):
        """Verify metric values when model perfectly matches ground truth."""
        from metrics import evaluate_model
        data = generate_line_data(n_inliers=50, n_outliers=0, slope=2.0, intercept=5.0, seed=42)
        perfect_model = LineModel(
            slope=2.0,
            intercept=5.0,
            inlier_mask=np.ones(50, dtype=bool),
        )

        metrics = evaluate_model(perfect_model, data, model_name="Perfect")

        assert metrics.slope_error == 0.0
        assert metrics.intercept_error == 0.0
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.inlier_precision == 1.0
        assert metrics.inlier_recall == 1.0
        assert metrics.inlier_f1 == 1.0
        assert metrics.classification_accuracy == 1.0

        d = metrics.to_dict()
        assert d["slope_error"] == 0.0
        assert d["inlier_f1"] == 100.0

    def test_compare_models(self):
        """Verify compare_models computes metrics for both models with RANSAC outperforming OLS."""
        from metrics import compare_models
        data = generate_line_data(n_inliers=80, n_outliers=30, slope=2.0, intercept=5.0, seed=42)
        ols_model = fit_ols(data.points)
        ransac_model = fit_ransac(data.points, distance_threshold=2.5, max_iterations=100, seed=42)

        ols_m, ransac_m = compare_models(ols_model, ransac_model, data)

        assert ransac_m.rmse < ols_m.rmse
        assert ransac_m.slope_error < ols_m.slope_error
        assert ransac_m.inlier_precision is not None and ransac_m.inlier_precision > 0.85


class TestVisualization:
    """Unit tests for chart generation."""

    def test_plots_generate_valid_files(self, tmp_path):
        """Verify visualization functions write valid, non-empty PNG files."""
        from visualization import plot_line_fitting, plot_comparison_summary
        from metrics import compare_models

        data = generate_line_data(n_inliers=40, n_outliers=10, seed=42)
        ols_model = fit_ols(data.points)
        ransac_model = fit_ransac(data.points, seed=42)
        ols_m, ransac_m = compare_models(ols_model, ransac_model, data)

        out1 = str(tmp_path / "test_fit.png")
        out2 = str(tmp_path / "test_comp.png")

        path1 = plot_line_fitting(data, ols_model, ransac_model, output_path=out1)
        path2 = plot_comparison_summary(data, ols_m, ransac_m, output_path=out2)

        import os
        assert os.path.exists(path1)
        assert os.path.getsize(path1) > 1000
        assert os.path.exists(path2)
        assert os.path.getsize(path2) > 1000

