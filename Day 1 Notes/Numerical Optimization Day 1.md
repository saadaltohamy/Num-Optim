---
share_link: https://share.note.sx/vzv14pyz#CQzKRpOCh8HeQWP+8ClQ7T2VdU/bNTTvlVoEZf9yUX8
share_updated: 2026-01-14T14:44:02+02:00
---
# Optimization Techniques

## Introduction

We want to find the optimal parameters of the model.

Equation:
$$
X \theta = Y
$$

Goal: Find $\theta$.

Algebraically, you can express it as:
$$
\theta = X^{-1} Y
$$

**Condition**: $X$ must be square and invertible. If not, we look for other solutions like OLS or Numerical Optimization.

---

## 1. Ordinary Least Squares (OLS)

O.L.S is the **Algebraic Solution** (Closed Form Solution).

$$
\theta = (X^T X)^{-1} X^T Y
$$

### Problems with OLS
1.  **Validity**: Valid for **Linear equations only**.
2.  **Scalability**: Calculating the inverse of a matrix $(X^T X)$ is computationally expensive ($O(n^3)$). If the matrix is too big, it becomes infeasible.
3.  **Limitations**: There is no algebraic solution for any equation with a degree greater than four.

> **Note**: In such cases, the only path is using **Numerical Solutions**, not closed-form equations.

### Scikit-Learn Implementation Notes

> *   Scikit-learn uses **SVD (Singular Value Decomposition)** as the closed-form solution for linear regression.
> *   It also offers **SGD (Stochastic Gradient Descent)** as a numerical solution.
> *   **Tip**: If you have the choice and the dataset size permits, use SVD (LinearRegression class) as it's exact.

---

## 2. Numerical Solutions (Gradient Descent)

Numerical solutions are iterative. You start with a guess and improve it step-by-step.

### General Steps:

1.  **Assumptions**: Initialize all parameters (e.g., to zeros or random values).
2.  **Prediction**: Calculate the output "predicted y": $\bar{y}$.
3.  **Evaluation**: Calculate the cost/loss (compare $\bar{y}$ with actual $y$).
4.  **Check**: Is this the optimal solution? (Stop if yes, continue if no).
5.  **Update**:
    *   Decide the direction of the update (positive or negative).
    *   Calculate the amount of the update.
    *   Update rule: $w_{new} = w_{old} + \text{update}$.
6.  **Loop**: Repeat steps 2-5 until convergence.

---

### Gradient Descent in Detail

**Scenario**: You have centered data (mean = 0). If you sum the simple errors $(\bar{y} - y)$, the result might be 0 due to positive and negative errors canceling out.

**Solution**: Use a Loss Function like L1 norm or L2 norm.

#### L1 Norm vs. L2 Norm

| Metric      | Description        | Formula                     | Geometry                          |
| :---------- | :----------------- | :-------------------------- | :-------------------------------- |
| **L2 Norm** | Euclidean Distance | $\sqrt{\sum (p_i - q_i)^2}$ | Straight line (shortest path)     |
| **L1 Norm** | Manhattan Distance | $\sum \|p_i - q_i\|$        | Grid-like path (moves along axes) |
![[l1_vs_l2_comparison.png]]
#### Cost Functions: MSE vs. MAE vs. RMSE

1.  **MSE (Mean Squared Error)**: L2 norm squared (averaged).
    *   Shape: Quadratic (Parabola). Smooth, differentiable.
    *   Value: Greater than RMSE.
2.  **RMSE (Root Mean Squared Error)**: L2 norm (averaged).
    *   Scale: Same unit as the target variable.
3.  **MAE (Mean Absolute Error)**: L1 norm (averaged).
    *   Shape: V-shape. Not differentiable at 0.

> **Optimization Note**: In optimization, we care about **THE SHAPE of the function**, not just the values. MSE is preferred because its gradient decreases as it approaches the minimum, allowing for finer updates. MAE gradients are constant, which can lead to *oscillation around the minimum*.

---

### Steps of Gradient Descent Algorithm

1.  **Initialize**: Set all parameters $\theta = 0$.
2.  **Predict**: Calculate $\bar{y}$.
3.  **Loss**: Calculate Cost (MSE).
    $$J(\theta) = \frac {1}{2m} \sum_{i=0}^m (\bar{y}^{(i)}-y^{(i)})^2$$
4.  **Direction & Magnitude**: Calculate the gradient (derivative).
5.  **Update**:
    $$
    \theta_{t+1} = \theta_{t} - \alpha \frac{dJ}{d \theta}
    $$
    *   Here, $\alpha$ is the **Learning Rate** (hyperparameter, usually 0 to 1).
    *   It controls the step size.

#### Calculating the Update (Step 4 & 5 logic)

*   Calculate derivative $\frac{dJ}{d \theta}$ (Slope).
*   **Logic**:
    *   If derivative is **positive** (slope up), we need to **decrease** $\theta$ (go left).
    *   If derivative is **negative** (slope down), we need to **increase** $\theta$ (go right).
    *   This is why we subtract the gradient: $\theta_{new} = \theta_{old} - \text{gradient}$.
*   **Overshooting**: If $\frac{dJ}{d \theta}$ is too large, you might overshoot the minimum. We multiply by $\alpha$ (learning rate) to scale it down.

---

### Convergence & Stopping Criteria (Loss Graph)

In your loss function graph, you ideally reach the global minimum where the gradient is 0.

**Stopping Conditions:**
1.  **Gradient Tolerance**: If gradient $< 0.001$ (close to 0), stop.
2.  **Cost Tolerance**: If change in cost $(J_{old} - J_{new}) < \text{tolerance}$, stop. This helps if you are stuck in a **Saddle Point** or a very flat plateau.

> **Vanishing Gradient**: When the gradient is very small but not zero (plateau), making learning extremely slow.

---

### Gradient Vector vs. Slope

*   $\frac{dJ}{d \theta}$ is the partial derivative with respect to one parameter.
*   **Gradient ($\nabla J$)**: A vector containing derivatives for all parameters.
    $$
    \nabla J(\theta) =
    \begin{bmatrix}
    \frac{\partial J}{\partial \theta_0} \\
    \frac{\partial J}{\partial \theta_1} \\
    \vdots
    \end{bmatrix}
    $$
*   **Direction**: The gradient vector points in the direction of steepest **ascent**.
*   **Update**: We move in the opposite direction (steepest **descent**).

### Vectorized update of params

1.  Calculate your predicted $y$ vector for all samples.
2.  Calculate your error vector for all samples $\dot e_{m \times 1}$.
3.  Calculate the cost function: L2 Norm of error vector *squared* divided by $2m$.
4.  Calculate the partial derivative $$\frac{dJ}{d \theta_n} = \frac{\text{error vector} \cdot x_n}{m}$$
5.  Calculate the entire gradient vector:
    $$ \nabla J_{n\times1} = \frac{1}{m} X^{T}_{n\times m} \times \dot e_{m\times1} $$
6.  Obtain the gradient vector $\nabla J_{n\times1}(\theta_0, \theta_1, \dots \theta_n)$.
7.  Update your parameters:
    $$ \theta_{new} = \theta_{old} - \alpha \nabla J_{n\times1} $$

---

## 3. Feature Scaling

### Problem

When features have different scales (e.g., "Age" 0-100 vs. "Salary" 1000-100000), it causes issues for Gradient Descent. Here is the step-by-step logic:

1.  **Different Ranges**: Feature $x_1$ ranges from 0 to 100, while $x_2$ ranges from 0 to 100,000.
2.  **Different Gradients**: The gradient with respect to a weight is proportional to the feature value ($\frac{\partial J}{\partial \theta_j} \propto x_j \cdot \text{error}$).
    *   Large feature values  $\Rightarrow$ Very large gradients. $\Rightarrow$ Small Weights
    *   Small feature values $\Rightarrow$ Small gradients. $\Rightarrow$ Large Weights
3.  **Elongated Shape**: This inequality creates a cost function surface that looks like a steep valley (elongated ellipse). It is steep in one direction and flat in the other.
4.  **Oscillation**:
    *   Gradient Descent takes huge steps in the steep direction (causing overshooting and bouncing).
    *   It takes tiny, slow steps in the flat direction.
    *   **Result**: The path oscillates back and forth, taking a very long time to converge to the minimum.

### Solutions

#### 1) Min-Max Scaling (Normalization)
Rescales data to a fixed range, usually [0, 1].

*   **Equation**:
    $$ X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
*   **Range**: $[0, 1]$
*   **Note**: Sensitive to outliers. An outlier can squish all other data into a tiny range.

#### 2) Standardization (Z-Score)
Centers data around 0 with a standard deviation of 1.

*   **Equation**:
    $$ X_{std} = \frac{X - \mu}{\sigma} $$
    (Where $\mu$ is mean, $\sigma$ is standard deviation)
*   **Range**: No fixed range (typically -3 to 3).
*   **Note**: Less affected by outliers than Min-Max, but still assumes a Gaussian-like distribution.

#### 3) Robust Scaler
Uses statistics that are robust to outliers (Median and IQR).

*   **Equation**:
    $$ X_{robust} = \frac{X - Q_2}{Q_3 - Q_1} $$
    (Where $Q_2$ is Median, $Q_3-Q_1$ is Interquartile Range IQR)
*   **Range**: Varies.
*   **Note**: Best choice if your dataset has significant outliers.

---

## 4. Types of Gradient Descent

The difference lies in how much data we use to calculate the gradient in one step.

### 1) Batch Gradient Descent
Uses the **entire dataset** to calculate the gradient for one update step.

*   **Definition**: Compute gradient of cost function w.r.t parameters for the *whole training set*.
*   **Pros**: Stable convergence; optimal for convex error surfaces.
*   **Cons**: Very slow for large datasets; requires loading all data into memory.
*   **Graph**: Smooth curve directly to minimum.

### 2) Stochastic Gradient Descent (SGD)
Uses a **single random training example** to calculate the gradient for one update step.

*   **Definition**: Pick one instance randomly, calculate error, and update parameters. Repeat.
*   **Cost Function (per instance)**:
    $$ J(\theta) = \frac{1}{2} (\bar{y}^{(i)} - y^{(i)})^2 $$
*   **Update**: Frequent updates with high variance.
*   **Pros**: Much faster; helps escape local minima (due to noise); low memory usage.
*   **Cons**: Noisy convergence (dances around the minimum rather than settling).

> **Why Shuffle Data?**
> 1.  **Break Cycles**: Data often comes ordered (e.g., sorted by date or class). Shuffling ensures the model doesn't learn these spurious patterns.
> 2.  **I.I.D Assumption**: SGD assumes samples are Independent and Identically Distributed. Shuffling approximates this.
> 3.  **Escaping Saddle Points**: Random sampling introduces noise that can help "kick" the optimizer out of flat saddle points where the gradient is near zero.

### 3) Mini-Batch Gradient Descent
Uses a **small batch** of data (e.g., 32, 64, 128 samples) for each update.

*   **Definition**: A compromise between Batch and SGD.
*   **Pros**:
    *   More stable than SGD.
    *   Faster than Batch.
    *   Allows matrix vectorization (GPU acceleration).
*   **Graph**: Wiggles towards minimum but smoother than SGD.
*   **Note**: This is the most common algorithm in Deep Learning.
