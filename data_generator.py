"""Synthetic 2D dataset generator for robust line fitting and outlier handling.

Generates ground-truth linear inliers with Gaussian measurement noise and
dispersed outliers across the region of interest.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Dataset:
    """Container holding 2D points and ground-truth metadata.

    Attributes:
        points: (N, 2) array of [x, y] coordinates.
        inlier_mask: (N,) boolean mask where True indicates a ground-truth inlier.
        true_slope: Ground-truth line slope (m).
        true_intercept: Ground-truth line intercept (c).
        x_range: (min_x, max_x) interval of the domain.
    """
    points: np.ndarray
    inlier_mask: np.ndarray
    true_slope: float
    true_intercept: float
    x_range: Tuple[float, float]

    @property
    def inliers(self) -> np.ndarray:
        """Return array of ground-truth inlier points."""
        return self.points[self.inlier_mask]

    @property
    def outliers(self) -> np.ndarray:
        """Return array of ground-truth outlier points."""
        return self.points[~self.inlier_mask]

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_inliers(self) -> int:
        return int(np.sum(self.inlier_mask))

    @property
    def n_outliers(self) -> int:
        return int(np.sum(~self.inlier_mask))


def generate_line_data(
    n_inliers: int = 100,
    n_outliers: int = 40,
    slope: float = 2.0,
    intercept: float = 5.0,
    noise_std: float = 1.0,
    x_range: Tuple[float, float] = (0.0, 20.0),
    y_margin: float = 15.0,
    seed: Optional[int] = 42,
    enforce_separation: bool = True,
) -> Dataset:
    """Generate synthetic 2D data points with controllable inliers and outliers.

    Parameters:
        n_inliers: Number of inlier points sampled along the true line.
        n_outliers: Number of random outlier points scattered across the scene.
        slope: Slope (m) of the ground-truth line: y = m*x + c.
        intercept: Y-intercept (c) of the ground-truth line.
        noise_std: Standard deviation of zero-mean Gaussian noise added to inliers.
        x_range: (x_min, x_max) span of the horizontal domain.
        y_margin: Vertical padding added above and below the true line range for outliers.
        seed: Random seed for reproducible dataset synthesis.

    Returns:
        Dataset object containing points, inlier mask, and ground-truth parameters.

    Raises:
        ValueError: If sample counts are invalid or x_range is degenerate.
    """
    if n_inliers < 2:
        raise ValueError("At least 2 inliers are required to define a line.")
    if n_outliers < 0:
        raise ValueError("Outlier count cannot be negative.")
    if noise_std < 0.0:
        raise ValueError("Noise standard deviation must be non-negative.")
    if x_range[0] >= x_range[1]:
        raise ValueError("x_range[0] must be strictly less than x_range[1].")

    rng = np.random.default_rng(seed)

    x_min, x_max = x_range

    # 1. Generate Inliers: uniformly sampled along x with Gaussian noise along y
    x_inliers = rng.uniform(x_min, x_max, size=n_inliers)
    y_clean = slope * x_inliers + intercept
    y_noise = rng.normal(loc=0.0, scale=noise_std, size=n_inliers)
    y_inliers = y_clean + y_noise

    inlier_points = np.column_stack((x_inliers, y_inliers))
    inlier_flags = np.ones(n_inliers, dtype=bool)

    # 2. Generate Outliers: scattered uniformly across the bounding region
    if n_outliers > 0:
        # Determine vertical bounding envelope
        y_line_min = slope * x_min + intercept
        y_line_max = slope * x_max + intercept
        y_min = min(y_line_min, y_line_max) - y_margin
        y_max = max(y_line_min, y_line_max) + y_margin

        x_outliers = rng.uniform(x_min, x_max, size=n_outliers)
        y_outliers = rng.uniform(y_min, y_max, size=n_outliers)

        if enforce_separation:
            # Ensure outliers are distinctly separated from the line (at least 3.0 * noise_std)
            separation_threshold = max(3.0 * noise_std, 2.0)
            line_y_at_outliers = slope * x_outliers + intercept
            close_mask = np.abs(y_outliers - line_y_at_outliers) < separation_threshold

            # Shift any points that happen to fall right on the line
            if np.any(close_mask):
                y_outliers[close_mask] += np.where(
                    rng.uniform(size=np.sum(close_mask)) > 0.5,
                    separation_threshold + 5.0,
                    -(separation_threshold + 5.0),
                )

        outlier_points = np.column_stack((x_outliers, y_outliers))
        outlier_flags = np.zeros(n_outliers, dtype=bool)

        all_points = np.vstack((inlier_points, outlier_points))
        all_flags = np.concatenate((inlier_flags, outlier_flags))
    else:
        all_points = inlier_points
        all_flags = inlier_flags

    # 3. Shuffle dataset to eliminate ordering bias
    indices = rng.permutation(len(all_points))
    shuffled_points = all_points[indices]
    shuffled_flags = all_flags[indices]

    return Dataset(
        points=shuffled_points,
        inlier_mask=shuffled_flags,
        true_slope=float(slope),
        true_intercept=float(intercept),
        x_range=x_range,
    )

