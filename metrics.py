"""Evaluation metrics for linear regression and RANSAC line fitting.

Calculates:
- Inlier and outlier classification accuracy (Precision, Recall, F1-Score).
- Geometric line estimation errors (Absolute Slope Error, Absolute Intercept Error).
- Domain prediction errors against ground-truth line (MAE, RMSE).
- Comparative performance summary between Ordinary Least Squares and RANSAC.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from data_generator import Dataset
from line_fitting import LineModel


@dataclass
class LineFittingMetrics:
    """Quantitative performance metrics for an estimated line model.

    Attributes:
        model_name: Identifier name (e.g., 'Ordinary Least Squares' or 'RANSAC').
        estimated_slope: Slope estimated by the model.
        estimated_intercept: Intercept estimated by the model.
        slope_error: Absolute difference from ground-truth slope: |m_est - m_true|.
        intercept_error: Absolute difference from ground-truth intercept: |c_est - c_true|.
        mae: Mean Absolute Error between model prediction and true line across domain.
        rmse: Root Mean Squared Error between model prediction and true line across domain.
        detected_inliers: Number of points classified as inliers by the model.
        detected_outliers: Number of points classified as outliers by the model.
        inlier_precision: TP / (TP + FP) against ground-truth inliers (if GT available).
        inlier_recall: TP / (TP + FN) against ground-truth inliers (if GT available).
        inlier_f1: Harmonic mean of precision and recall.
        classification_accuracy: Overall binary classification accuracy (TP + TN) / Total.
    """
    model_name: str
    estimated_slope: float
    estimated_intercept: float
    slope_error: float
    intercept_error: float
    mae: float
    rmse: float
    detected_inliers: int
    detected_outliers: int
    inlier_precision: Optional[float] = None
    inlier_recall: Optional[float] = None
    inlier_f1: Optional[float] = None
    classification_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a formatted dictionary."""
        return {
            "model_name": self.model_name,
            "estimated_slope": round(self.estimated_slope, 4),
            "estimated_intercept": round(self.estimated_intercept, 4),
            "slope_error": round(self.slope_error, 4),
            "intercept_error": round(self.intercept_error, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "detected_inliers": self.detected_inliers,
            "detected_outliers": self.detected_outliers,
            "inlier_precision": round(self.inlier_precision * 100, 2) if self.inlier_precision is not None else None,
            "inlier_recall": round(self.inlier_recall * 100, 2) if self.inlier_recall is not None else None,
            "inlier_f1": round(self.inlier_f1 * 100, 2) if self.inlier_f1 is not None else None,
            "classification_accuracy": round(self.classification_accuracy * 100, 2) if self.classification_accuracy is not None else None,
        }


def evaluate_model(
    model: LineModel,
    dataset: Dataset,
    model_name: str = "Model",
    eval_samples: int = 500,
) -> LineFittingMetrics:
    """Evaluate an estimated line model against ground truth metadata.

    Note on Ground Truth:
        The ground truth line (true_slope and true_intercept) is strictly an
        evaluation oracle and is NEVER provided to the fitting algorithms.

    Parameters:
        model: Estimated LineModel from OLS or RANSAC.
        dataset: Dataset containing ground truth inlier mask, true slope, and true intercept.
        model_name: Human-readable name for logging.
        eval_samples: Number of uniformly spaced domain samples for MAE/RMSE calculation.

    Returns:
        LineFittingMetrics object holding all computed metrics.
    """
    m_est = model.slope
    c_est = model.intercept
    m_gt = dataset.true_slope
    c_gt = dataset.true_intercept

    # 1. Parameter Estimation Errors
    slope_err = float(np.abs(m_est - m_gt))
    intercept_err = float(np.abs(c_est - c_gt))

    # 2. Prediction Errors against Ground-Truth Line across the Domain
    x_min, x_max = dataset.x_range
    x_eval = np.linspace(x_min, x_max, eval_samples)
    y_true = m_gt * x_eval + c_gt
    y_pred = model.predict(x_eval)

    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # 3. Inlier/Outlier Classification Metrics
    total_pts = dataset.n_points
    if model.inlier_mask is not None:
        pred_inliers = model.inlier_mask
        det_inliers = int(np.sum(pred_inliers))
        det_outliers = int(total_pts - det_inliers)

        true_inliers = dataset.inlier_mask

        tp = int(np.sum(true_inliers & pred_inliers))
        fp = int(np.sum((~true_inliers) & pred_inliers))
        fn = int(np.sum(true_inliers & (~pred_inliers)))
        tn = int(np.sum((~true_inliers) & (~pred_inliers)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = float(2.0 * (precision * recall) / denom) if denom > 0.0 else 0.0
        accuracy = float((tp + tn) / total_pts) if total_pts > 0 else 0.0
    else:
        det_inliers = total_pts
        det_outliers = 0
        precision = None
        recall = None
        f1 = None
        accuracy = None

    return LineFittingMetrics(
        model_name=model_name,
        estimated_slope=m_est,
        estimated_intercept=c_est,
        slope_error=slope_err,
        intercept_error=intercept_err,
        mae=mae,
        rmse=rmse,
        detected_inliers=det_inliers,
        detected_outliers=det_outliers,
        inlier_precision=precision,
        inlier_recall=recall,
        inlier_f1=f1,
        classification_accuracy=accuracy,
    )


def compare_models(
    ols_model: LineModel,
    ransac_model: LineModel,
    dataset: Dataset,
) -> Tuple[LineFittingMetrics, LineFittingMetrics]:
    """Evaluate and compare Ordinary Least Squares and RANSAC side-by-side."""
    ols_metrics = evaluate_model(ols_model, dataset, model_name="Ordinary Least Squares")
    ransac_metrics = evaluate_model(ransac_model, dataset, model_name="RANSAC")
    return ols_metrics, ransac_metrics

