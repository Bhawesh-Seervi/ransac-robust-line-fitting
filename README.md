# Robust Line Fitting and Outlier Handling using the RANSAC Algorithm

A lightweight, college-level computer vision and parameter estimation project demonstrating why the **RANdom SAmple Consensus (RANSAC)** algorithm robustly estimates 2D line parameters in the presence of severe outlier contamination, whereas **Ordinary Least Squares (OLS)** linear regression breaks down.

---

## 1. Project Overview

In real-world computer vision, image processing, and robotics (e.g., lane line detection, horizon tracking, wall boundary extraction in LIDAR scans), raw datasets are frequently corrupted by sensor noise, spurious reflections, and unmodelled clutter (outliers). 

Classical linear regression using **Ordinary Least Squares (OLS)** minimizes the sum of squared vertical residuals. Because large residuals are squared, distant outliers exert an outsized quadratic leverage on the fitted parameters, twisting the estimated line away from the genuine inlier distribution.

**RANSAC** reformulates model fitting as a search for maximum consensus:
1. It repeatedly fits minimal sample sets (2 points for a 2D line).
2. It scores hypotheses by counting consensus inliers within an orthogonal distance threshold $\tau$.
3. It refits the final model over all identified consensus inliers.

This repository provides a clean, pure Python/NumPy implementation comparing OLS and RANSAC side-by-side with automated metrics, high-resolution visualizations, and a unit test suite.

---

## 2. Problem Statement

Given a set of 2D data points $\mathcal{P} = \{(x_i, y_i)\}_{i=1}^N$ where a subset of points (inliers) approximately satisfy a linear relationship:

$$y_i = m \cdot x_i + c + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

and the remaining points (outliers) are distributed arbitrarily across the domain:
- How does Ordinary Least Squares behave when outliers are present?
- How does RANSAC effectively isolate the inliers and recover the true parameters $(m, c)$?

---

## 3. Technologies Used

- **Language**: Python 3.12 / 3.13
- **Numerical Computation**: NumPy
- **Visualizations**: Matplotlib (headless `Agg` backend)
- **Baseline Modeling**: scikit-learn / NumPy analytical least-squares
- **Testing**: pytest

---

## 4. Mathematical Formulation

### 4.1 Ordinary Least Squares (OLS)

OLS estimates slope $m$ and intercept $c$ by minimizing the sum of squared vertical residuals:

$$S(m, c) = \sum_{i=1}^N \left( y_i - (m x_i + c) \right)^2$$

Setting the partial derivatives $\frac{\partial S}{\partial m} = 0$ and $\frac{\partial S}{\partial c} = 0$ yields the closed-form analytical solution:

$$m_{\text{OLS}} = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}, \quad c_{\text{OLS}} = \bar{y} - m_{\text{OLS}} \bar{x}$$

where $\bar{x} = \frac{1}{N}\sum x_i$ and $\bar{y} = \frac{1}{N}\sum y_i$.

**Why OLS Fails with Outliers:**
The penalty function is quadratic ($L_2$ norm). A single outlier located at distance $D \gg \sigma$ contributes $D^2$ to the error sum, heavily distorting the solution.

### 4.2 Standard Line Representation & Perpendicular Distance

A candidate line passing through two distinct points $(x_1, y_1)$ and $(x_2, y_2)$ is expressed in standard form:

$$A x + B y + C = 0$$

where:
$$A = y_2 - y_1, \quad B = -(x_2 - x_1), \quad C = x_2 y_1 - x_1 y_2$$

The exact orthogonal (perpendicular) Euclidean distance from any point $(x_0, y_0)$ to this line is:

$$d = \frac{|A x_0 + B y_0 + C|}{\sqrt{A^2 + B^2}}$$

Equivalently, for a line parameterized by $y = m x + c$ ($m x - y + c = 0$):

$$d = \frac{|m x_0 - y_0 + c|}{\sqrt{m^2 + 1}}$$

*Note: Perpendicular distance is geometrically invariant to coordinate rotations and correctly measures distance to the line, whereas vertical distance $|y_0 - (m x_0 + c)|$ over-penalizes steep lines and becomes infinite for vertical lines.*

### 4.3 Theoretical RANSAC Iteration Bound

Let:
- $s = 2$ be the minimal sample size required to instantiate a line.
- $w = \frac{N_{\text{inliers}}}{N_{\text{total}}}$ be the true inlier ratio.
- $\epsilon = 1 - w$ be the outlier ratio.
- $p$ be the desired probability that at least one sampled pair consists entirely of inliers (typically $p = 0.99$).

The probability that all $s$ points in a random sample are inliers is $w^s = (1 - \epsilon)^s$. The probability that at least one of the $s$ points is an outlier is $1 - (1 - \epsilon)^s$.

Across $N_{\text{iter}}$ independent trials, the probability that *every* sample contains at least one outlier is:

$$1 - p = \left( 1 - (1 - \epsilon)^s \right)^{N_{\text{iter}}}$$

Taking natural logarithms gives the minimum required iteration count:

$$N_{\text{iter}} = \frac{\ln(1 - p)}{\ln\left(1 - (1 - \epsilon)^s\right)}$$

For our default parameters ($w = \frac{100}{140} \approx 0.714$, $s = 2$, $p = 0.99$):

$$N_{\text{iter}} = \frac{\ln(0.01)}{\ln(1 - 0.714^2)} = \frac{-4.605}{\ln(0.4897)} \approx 6.45 \implies \approx 7 \text{ iterations}$$

Executing $N = 100$ iterations provides a theoretical success confidence exceeding $99.999\%$.

---

## 5. RANSAC Algorithm Steps

```text
Input: Data points P = {(x_i, y_i)}, distance threshold tau, max iterations N
Output: Robust line (m*, c*), consensus inlier mask

1. best_inliers = empty set
2. For iteration = 1 to N:
     a. Randomly select 2 distinct points p1, p2 from P without replacement.
     b. If distance(p1, p2) < epsilon (degenerate pair), reject and resample.
     c. Compute normalized line coefficients (A, B, C).
     d. For each point p_i in P:
          Compute perpendicular distance d_i = |A*x_i + B*y_i + C| / sqrt(A^2 + B^2)
     e. Form candidate inlier set: S = { p_i in P | d_i <= tau }
     f. If |S| > |best_inliers|:
          best_inliers = S
3. If |best_inliers| >= 2:
     Refit final (m*, c*) via Least Squares using ONLY best_inliers.
4. Return (m*, c*) and inlier mask.
```

---

## 6. Project Structure

```text
ransac-line-fitting/
├── data_generator.py      # Synthetic 2D data generator (inliers + Gaussian noise + outliers)
├── line_fitting.py        # Ordinary Least Squares (OLS) and manual RANSAC implementations
├── metrics.py             # Evaluation metrics (MAE, RMSE, parameter errors, inlier precision/recall)
├── visualization.py       # Publication-ready Matplotlib chart generation (saved as PNGs)
├── main.py                # Command-line interface and end-to-end execution pipeline
├── tests/
│   └── test_ransac.py     # Automated unit tests covering generation, fitting, metrics, and charts
├── requirements.txt       # Project dependency specifications
├── pytest.ini            # Pytest path configuration
└── README.md              # Complete theoretical and practical guide
```

---

## 7. Installation & Execution

### 7.1 Virtual Environment Setup

```bash
# Navigate to the project directory
cd ransac-line-fitting

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Running the Application

Execute the pipeline with default parameters ($m=2.0, c=5.0$, 100 inliers, 40 outliers):

```bash
python main.py
```

#### Configurable CLI Arguments:

```bash
python main.py \
  --inliers 120 \
  --outliers 50 \
  --slope 1.5 \
  --intercept 3.0 \
  --noise 0.8 \
  --threshold 2.0 \
  --iterations 150 \
  --seed 42 \
  --output-dir ./charts
```

### 7.3 Running the Automated Tests

Run the complete 30-test verification suite:

```bash
pytest -v
```

---

## 8. Experimental Results & Comparison

### Default Benchmark Run

- **Ground-Truth Equation**: $y = 2.00x + 5.00$
- **Total Points**: 140 ($100$ inliers with $\sigma=1.0$, $40$ dispersed outliers $\approx 28.6\%$ contamination)
- **RANSAC Configuration**: $\tau = 2.5$, $N = 100$ iterations, seed = 42

| Evaluation Metric | Ordinary Least Squares (OLS) | RANSAC (Robust) | Improvement / Note |
| :--- | :---: | :---: | :--- |
| **Estimated Slope ($m$)** | `0.9934` | **`1.9680`** | Ground truth: `2.0000` |
| **Slope Absolute Error** | `1.0066` | **`0.0320`** | **31.5x lower error** |
| **Estimated Intercept ($c$)** | `13.1544` | **`5.1408`** | Ground truth: `5.0000` |
| **Intercept Absolute Error**| `8.1544` | **`0.1408`** | **57.9x lower error** |
| **Domain Prediction MAE** | `5.2245` px | **`0.2108` px** | **24.8x lower error** |
| **Domain Prediction RMSE** | `6.1293` px | **`0.2581` px** | **23.7x lower error** |
| **Consensus Inliers** | `140/140` (Assumes all valid) | **`105/140`** | True inliers: 100 |
| **Inlier Precision** | N/A | **`95.2%`** | Only 5 false alarms |
| **Inlier Recall** | `100.0%` | **`100.0%`** | All true inliers retained |
| **Inlier F1-Score** | N/A | **`97.6%`** | High classification fidelity |
| **Execution Time** | `< 1.0 ms` | **`~5.0 ms`** | Fast CPU execution |

### Generated Visualizations

1. **`ransac_vs_ols_fitting.png`**:
   Shows inlier points (blue circles), outliers (red crosses), ground truth (black dashed), OLS fit (orange dash-dot), and RANSAC fit (green solid). OLS is heavily biased downward, while RANSAC cleanly overlays the ground-truth line.
2. **`comparison_summary.png`**:
   Displays a side-by-side trajectory divergence plot and a quantitative error bar chart comparing slope error, intercept error, and line RMSE.

---

## 9. Advantages of RANSAC

1. **Robust to Heavy Outlier Contamination**: RANSAC succeeds even when outliers comprise 30% to 50%+ of the dataset.
2. **Minimal Model Hypothesis**: Only requires sampling 2 points per trial for a line, ensuring a high probability of drawing a pure inlier sample within a modest iteration budget.
3. **Model Independence**: The same consensus paradigm applies to circles, planes, homographies, and fundamental matrices in multi-view geometry.
4. **Final Refinement**: Re-fitting with standard least squares over the consensus inlier set eliminates sample discretization noise and yields statistically efficient estimates.

---

## 10. Limitations

1. **Non-Deterministic Nature**: Because it relies on random sampling, different random seeds can produce slight parameter variations unless a fixed seed or sufficient iterations are used.
2. **Sensitivity to Distance Threshold ($\tau$)**:
   - If $\tau$ is chosen too small, true inliers with Gaussian noise are rejected.
   - If $\tau$ is chosen too large, outliers are incorrectly admitted into the consensus set.
3. **Exponential Iteration Scaling at Low Inlier Ratios**: If inliers drop to $10\%$, $N_{\text{iter}}$ for 99% confidence increases from 7 to $\approx 460$ iterations.
4. **Computational Cost**: More computationally intensive than closed-form analytical OLS regression.

---

## 11. Conclusion

Ordinary Least Squares regression is effective for Gaussian-distributed measurement errors but lacks robustness when contaminated with arbitrary outliers. RANSAC decouples model estimation from outliers by evaluating candidate models through inlier consensus. 

This project demonstrates that combining **random minimal sampling**, **perpendicular Euclidean distance gating**, and **least-squares inlier refinement** produces an accurate, reliable estimator capable of isolating linear structure under severe outlier contamination.

