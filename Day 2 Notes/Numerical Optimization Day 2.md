---
share_link: https://share.note.sx/hg0x3jr3#82VtiYWnh9/lysCmIpMVynwLv7ixtKgzwFe6gC6UWqw
share_updated: 2026-01-17T12:27:42+02:00
---
# Introduction

- [[Numerical Optimization Day 1#Steps of Gradient Descent Algorithm|Gradient Descent]] works well with convex functions.
- In Deep Learning, 99% of the time the loss function *is not convex*.
- We use other optimization techniques that work better in these cases.

## Problems of GD in Deep Learning

### 1) Vanishing Gradient

- In deep learning, there are many layers with many parameters.
- When you calculate [[Numerical Optimization Day 1#Calculating the Update (Step 4 & 5 logic)|Gradients]] using the Chain rule, the values become really small numbers.
- When the gradients are really small, it's like you are in a **saddle point**.
- To overcome the saddle point, you need a lot of iterations, which is inefficient.

> في التزحلق على الجليد، الرياضي يقوم بدفع نفسه بالقوة عندما يكون في السطح، بينما لا يفعل هذا اذا كان يوجد انحناء كبير في الجبل
> This *Rush* is called **Momentum**

# Momentum

- The original update equation:
  $$\theta_{t+1} = \theta_{t} - \alpha \frac{dJ}{d \theta} $$
- The change is to add Momentum:
  $$\theta_{t+1} = \theta_{t} - \alpha \frac{dJ}{d \theta}  + \text {Momentum}$$
  Let:
  $$\alpha \frac{dJ}{d \theta}  + \text {Momentum} = V_t$$
  Then:
  $$ \theta_{t+1} = \theta_{t} - V_t$$

- We need to adjust the momentum based on *the direction of the gradient*.
    - When the direction changes (meaning *you are going towards the maximum not the minimum*), you need to decrease the momentum.

$$ V_t = \gamma V_{t-1} + \alpha \nabla \theta_t $$

- When $\gamma$ is large (e.g., $\gamma \approx 1$) $\Rightarrow$ You take into account the entire history.
- When $\gamma$ is small (e.g., $\gamma \approx 0$) $\Rightarrow$ You *don't* take any history $\Rightarrow$ same as GD.
- When the direction changes, the history contradicts that change, so the step becomes smaller (slower).

## Example

### 1) Iteration 1

$$ V_0 = 0 $$
$$ V_1 = \gamma V_0 + \alpha \nabla \theta_1 $$
$$ V_1 = \alpha \nabla \theta_1 $$

### 2) Iteration 2

$$ V_2 = \gamma [\alpha \nabla \theta_1] + \alpha \nabla \theta_2 $$

### 3) Iteration 3

$$ V_3 = \gamma [\gamma [\alpha \nabla \theta_1] + \alpha \nabla \theta_2] + \alpha \nabla \theta_3 $$
$$ V_3 = \gamma^2 [\alpha \nabla \theta_1] + \gamma [\alpha \nabla \theta_2] + \alpha \nabla \theta_3 $$

- So the general equation is:
  $$ V_t = \sum_{i=0}^{t} \gamma^{t-i} [\alpha \nabla \theta_i] $$

> It's an **Exponentially Weighted Sum**.
> We weight the history exponentially.
> If $\gamma = 0.9$, you give a lot of weight to history.

![[Pasted image 20260110112848.png]]
[[Diagram Coding#Diagram 1|Code]]

## Notes

1. In a Convex function, when you get to the minimum, the previous momentum is already large, so you overshoot.
2. The overshooting is small.
3. Then the momentum becomes smaller and smaller, so you go towards the minimum again.
4. You overshoot again, but less than the previous time.
5. Loop to step 3.
6. Finally, you settle in the minimum.

![[Pasted image 20260110113334.png]]
[[Diagram Coding#Diagram 2|Code]]

> Momentum is bad in Convex functions because it increases the number of iterations (oscillations).
> Momentum in most cases $\gamma = 0.9$.
> $V_t$ is a vector of zeros $n \times 1$ where $n$ is the number of parameters.

# NAG (Nesterov Accelerated Gradient)

- We need to look a step ahead to see if, after the momentum update, the loss will increase or not.
- This look-ahead point is called $\theta _{temp}$.
- $$ \theta _{temp} = \theta_t - \gamma V_{t-1} $$
- We will update our theta based on the gradient of that future point:
  $$ \theta_{t+1} = \theta_{temp} - \alpha \nabla J(\theta _{temp}) $$
- The velocity vector becomes:
  $$ V_t = \gamma V_{t-1} + \alpha \nabla J(\theta _{temp}) $$

## Notes for interpretation

1. We have 2 terms in Momentum: $V_t = \gamma V_{t-1} + \alpha \nabla \theta_t$
    1. $V_{t-1}$ (History/Momentum)
    2. $\nabla \theta_t$ (Current Gradient)
2. In NAG, we first update by the momentum term $V_{t-1}$ to get $\theta_{temp}$.
3. Then we calculate the gradient at this new position: $\nabla \theta_{temp}$.
4. Finally, we combine $V_{t-1}$ and $\nabla \theta_{temp}$ to get the new update.
    1. This is equivalent to $\theta_{t+1} = \theta_{temp} - \alpha \nabla J(\theta _{temp})$.

### Why do we do this?

1. In standard Momentum, if we add $\gamma V_{t-1} + \alpha \nabla \theta_t$, the update might be too large and overshoot the minimum.
2. In NAG, we have a "future outlook".
    1. We adjust that future outlook by *the gradient* at that future point, correcting the trajectory earlier.

# Adaptive Gradient (AdaGrad)

## Problem

- If the range of features is different, the gradient of each feature varies—some are small and some are large. We have to scale the features to minimize the number of iterations.
- This works for numerical features, not categorical ones.
- Categorical features are encoded using one-hot encoding.
- The generated features are **Sparse features**.
- We can't use [[Day 1#Feature Scaling|Feature Scaling]] on Sparse features!

## Solution

- We have to make the gradients of all features comparable in range.
- So we have to **adapt** the gradients:
    - For small gradients, increase the effective learning rate.
    - For large gradients, decrease the effective learning rate.

## How?

- We will divide the large gradient by a large number.
- We will divide the small gradient by a small number.
- That number is the sum of squared gradients:
  $$G_t = G_{t-1} + g_t \odot g_t$$
- **Why Squared?**
    - Because we care about the magnitude (quantity), not the direction.
- **Why not absolute value?**
    - Because the square root of the sum of squares works better for normalization (L2 norm vs L1 norm). [[Day 1#L1 norm vs L2 norm]]

Then we update:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t $$

> This is Okay for Sparse Features.
> This is a large problem for Dense Features.

## Problem of AdaGrad

- In Dense Features, the sum of squared gradients ($G_t$) becomes larger and larger.
- The effective learning rate becomes smaller and smaller, leading to a **KILLING LEARNING RATE** (vanishing learning rate).
- For sparse features, $\theta$ updates are okay (since $g_t$ is often 0), but for dense features, updates stop because the learning rate becomes really small.

> Maybe we can take a chunk of history instead of the entire history?

# RMSProp

1. We just change $G_t$ to take a weighted average of $G_{t-1}$:
   $$G_t = \beta G_{t-1} +(1- \beta) g_t ^2$$
    - $\beta$ ranges from 0 to 1 $\Rightarrow$ most probably 0.9.
2. Then we update:
   $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t $$
3. The equation $G_t = \beta G_{t-1} +(1- \beta) g_t ^2$ is an **Exponentially Weighted Moving Average (EWMA)**.

---

# Adam

## Problem

1. Momentum & NAG solve the Vanishing Gradient problem.
2. AdaGrad & RMSProp solve the Sparse Features problem.

> We need an algorithm to merge both.

## How?

### Notations

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| $\theta_t$ | parameters at step $t$ | - |
| $g_t = \nabla_\theta J(\theta_t)$ | gradient at step $t$ | - |
| $\eta$ | learning rate | - |
| $\beta_1$ | momentum decay | 0.9 |
| $\beta_2$ | RMS decay | 0.999 |
| $\epsilon$ | small constant | $10^{-8}$ |
| $m_t$ | 1st moment estimate (**momentum / EWMA of gradients**) | - |
| $v_t$ | 2nd moment estimate (**EWMA of squared gradients**) | - |

### Equation of Momentum (EWMA of gradients)

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

### Equation of RMSProp (EWMA of squared gradients)

$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$

### Final equation of updating parameters

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## Problem of EWMA (Bias)

$m_t$ and $v_t$ start from zero.
Why is this a problem?

### Example:

If $\beta_1 = 0.98$:

1. **Iteration 1**
   $$m_1 = (1-0.98) g_1 = 0.02 g_1$$

2. **Iteration 2**
   $$m_2 = 0.98 [0.02 g_1] + 0.02 g_2 = 0.0196 g_1 + 0.02 g_2$$

3. **Iteration 3**
   $$m_3 = 0.98 [0.0196 g_1 + 0.02 g_2] + 0.02 g_3 = 0.019208 g_1 + 0.0196 g_2 + 0.02 g_3$$

> Notice that the term $(1 - \beta_1)$ makes $m_t$ really small in the first iterations!
> There is a bias in the first iterations because we start from zero.

## Bias Correction

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

- In the first iteration, $m_t = (1-\beta_1) g_t$.
- So if we divide by $(1 - \beta_1^1)$, we get $g_t$ (unbiased).
- In later iterations, $\beta_1^t$ becomes really small, so the denominator becomes approximately 1 $\rightarrow$ no effect on later iterations.

## Final Equations

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$
