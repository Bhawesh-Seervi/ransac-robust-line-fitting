"""Line fitting algorithms: Ordinary Least Squares (OLS) and manual RANSAC.

Provides:
- LineModel dataclass for parameter storage, prediction, and orthogonal distance.
- fit_ols: Standard least-squares line regression baseline.
- fit_ransac: Manual Random Sample Consensus with two-point sampling,
  perpendicular distance classification, degeneracy handling, and inlier refitting.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LineModel:
    """Represents a 2D line defined by y = slope * x + intercept.

    Attributes:
        slope: Slope (m) of the line.
        intercept: Y-intercept (c) of the line.
        inlier_mask: Optional boolean array of shape (N,) indicating consensus inliers.
        n_iterations: Optional count of RANSAC iterations executed.
        inlier_count: Optional number of inliers supporting this model.
    """
    slope: float
    intercept: float
    inlier_mask: Optional[np.ndarray] = None
    n_iterations: Optional[int] = None
    inlier_count: Optional[int] = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calculate predicted y values for given x coordinates: y = m*x + c."""
        return self.slope * np.asarray(x, dtype=np.float64) + self.intercept

    def perpendicular_distance(self, points: np.ndarray) -> np.ndarray:
        """Compute exact perpendicular (orthogonal) distance from points to the line.

        For a line m*x - y + c = 0, the perpendicular distance from point (x0, y0) is:
            d = |m*x0 - y0 + c| / sqrt(m^2 + 1)

        Parameters:
            points: Array of shape (N, 2) containing [x, y] coordinates.

        Returns:
            Array of shape (N,) with non-negative perpendicular distances.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points array must have shape (N, 2), got {pts.shape}")

        x = pts[:, 0]
        y = pts[:, 1]
        numerator = np.abs(self.slope * x - y + self.intercept)
        denominator = np.sqrt(self.slope ** 2 + 1.0)
        return numerator / denominator


def fit_ols(points: np.ndarray) -> LineModel:
    """Fit a straight line using Ordinary Least Squares (OLS) regression.

    Minimizes the sum of squared vertical residuals: sum((y_i - (m*x_i + c))^2).

    Parameters:
        points: (N, 2) array of [x, y] coordinates.

    Returns:
        LineModel containing estimated slope and intercept.

    Raises:
        ValueError: If points has fewer than 2 samples or if all x coordinates are identical.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("At least 2 points are required for Ordinary Least Squares line fitting.")

    x = pts[:, 0]
    y = pts[:, 1]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_diff = x - x_mean
    y_diff = y - y_mean

    denominator = np.sum(x_diff ** 2)
    if denominator < 1e-12:
        raise ValueError("Cannot fit line: all points have virtually identical x coordinates (vertical line).")

    slope = float(np.sum(x_diff * y_diff) / denominator)
    intercept = float(y_mean - slope * x_mean)

    return LineModel(
        slope=slope,
        intercept=intercept,
        inlier_mask=np.ones(len(pts), dtype=bool),
        inlier_count=len(pts),
    )


def compute_line_from_two_points(
    p1: np.ndarray,
    p2: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Compute standard form line coefficients A*x + B*y + C = 0 from two distinct points.

    Returns:
        (A, B, C) coefficients normalized such that sqrt(A^2 + B^2) = 1,
        or None if points are degenerate (coincident or vertical line).
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    dx = x2 - x1
    dy = y2 - y1

    # Check for degenerate coincident points
    if (dx * dx + dy * dy) < 1e-12:
        return None

    # Standard form: (y2 - y1)*x - (x2 - x1)*y + (x2*y1 - x1*y2) = 0
    a = dy
    b = -dx
    c = x2 * y1 - x1 * y2

    norm = np.sqrt(a * a + b * b)
    if norm < 1e-12:
        return None

    return a / norm, b / norm, c / norm


def fit_ransac(
    points: np.ndarray,
    distance_threshold: float = 2.5,
    max_iterations: int = 100,
    min_inliers: int = 2,
    seed: Optional[int] = 42,
) -> LineModel:
    """Robust line fitting using the manual RANSAC algorithm.

    Steps:
    1. Randomly sample two distinct points.
    2. Check for and reject degenerate samples (nearly coincident points).
    3. Calculate the candidate line in standard form.
    4. Compute perpendicular distances from all points to the candidate line.
    5. Count consensus inliers where distance <= distance_threshold.
    6. Retain the candidate hypothesis yielding the maximum inlier set.
    7. Refit the final line using Ordinary Least Squares across all consensus inliers.

    Parameters:
        points: (N, 2) array of [x, y] coordinates.
        distance_threshold: Maximum perpendicular distance for inlier classification.
        max_iterations: Number of random consensus trials to execute.
        min_inliers: Minimum inliers required to accept a model hypothesis.
        seed: Random seed for deterministic reproducibility.

    Returns:
        LineModel containing refit slope, intercept, consensus inlier mask,
        total iterations run, and the final inlier count.

    Raises:
        ValueError: If input points have fewer than 2 samples or if parameters are invalid.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_points = len(pts)

    if n_points < 2:
        raise ValueError("At least 2 points are required for RANSAC line fitting.")
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be strictly positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    rng = np.random.default_rng(seed)

    best_inlier_count = -1
    best_inlier_mask = np.zeros(n_points, dtype=bool)
    best_candidate_model: Optional[Tuple[float, float]] = None

    x_all = pts[:, 0]
    y_all = pts[:, 1]

    for _ in range(max_iterations):
        # 1. Randomly sample two distinct point indices
        sample_indices = rng.choice(n_points, size=2, replace=False)
        p1 = pts[sample_indices[0]]
        p2 = pts[sample_indices[1]]

        # 2. Compute candidate line coefficients (normalized A*x + B*y + C = 0)
        line_params = compute_line_from_two_points(p1, p2)
        if line_params is None:
            continue  # Degenerate sample pair, skip

        a_norm, b_norm, c_norm = line_params

        # 3. Compute perpendicular distance from all points to candidate line:
        # Since sqrt(A^2 + B^2) == 1, distance = |A*x + B*y + C|
        distances = np.abs(a_norm * x_all + b_norm * y_all + c_norm)

        # 4. Inlier classification by distance threshold
        inliers = distances <= distance_threshold
        inlier_count = int(np.sum(inliers))

        # 5. Keep best hypothesis with largest consensus inlier set
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inliers

            # Candidate slope and intercept (if non-vertical: B != 0)
            if np.abs(b_norm) > 1e-8:
                cand_slope = -a_norm / b_norm
                cand_intercept = -c_norm / b_norm
                best_candidate_model = (float(cand_slope), float(cand_intercept))

    # 6. Final Refinement: Refit line using all consensus inliers
    if best_inlier_count >= max(2, min_inliers) and np.sum(best_inlier_mask) >= 2:
        inlier_points = pts[best_inlier_mask]
        try:
            refined_model = fit_ols(inlier_points)
            final_slope = refined_model.slope
            final_intercept = refined_model.intercept
        except ValueError:
            # Fallback to candidate model if inliers are vertically collinear
            if best_candidate_model is not None:
                final_slope, final_intercept = best_candidate_model
            else:
                fallback = fit_ols(pts)
                final_slope, final_intercept = fallback.slope, fallback.intercept
    elif best_candidate_model is not None:
        final_slope, final_intercept = best_candidate_model
    else:
        # Extreme fallback if all sampled pairs were degenerate
        fallback = fit_ols(pts)
        final_slope, final_intercept = fallback.slope, fallback.intercept
        best_inlier_mask = np.ones(n_points, dtype=bool)
        best_inlier_count = n_points

    # Re-evaluate inlier mask on final refined line for consistency
    refined_line = LineModel(slope=final_slope, intercept=final_intercept)
    final_distances = refined_line.perpendicular_distance(pts)
    final_inlier_mask = final_distances <= distance_threshold
    final_inlier_count = int(np.sum(final_inlier_mask))

    return LineModel(
        slope=final_slope,
        intercept=final_intercept,
        inlier_mask=final_inlier_mask,
        n_iterations=max_iterations,
        inlier_count=final_inlier_count,
    )

