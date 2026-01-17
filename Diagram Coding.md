---
share_link: https://share.note.sx/j4cp3pii#rVSEO83ZWBIsMAVHzpuA7mW89zeN4l8KUovpBQun+b4
share_updated: 2026-01-17T12:27:36+02:00
---
# Day 2
## Diagram 1

```python
import matplotlib.pyplot as plt

import numpy as np

  

def plot_momentum_weights(t=10):

    # Setup gammas to compare

    gammas = [0.5, 0.7, 0.9]

    iterations = np.arange(1, t + 1)

    plt.figure(figsize=(10, 6), facecolor='#1e1e1e')

    ax = plt.gca()

    ax.set_facecolor('#1e1e1e')

  

    # Colors to match your hand-drawn style or clear distinction

    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    for gamma, color in zip(gammas, colors):

        # Weight formula: gamma^(t-i)

        weights = [gamma**(t - i) for i in iterations]

        # Plotting the curves

        plt.plot(iterations, weights, label=f'Gamma = {gamma}',

                 marker='o', linewidth=2, color=color)

  

    # Formatting to match the provided image

    plt.xlabel('Iterations (i)', color='white', fontsize=12)

    plt.ylabel('Weight Contribution to $V_t$', color='white', fontsize=12)

    plt.title(f'Exponentially Weighted Sum (at Iteration $t={t}$)', color='white', fontsize=14)

    # Grid and axis styling

    plt.xticks(iterations, color='white')

    plt.yticks(color='white')

    ax.spines['bottom'].set_color('red')

    ax.spines['left'].set_color('red')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.legend(facecolor='#333333', labelcolor='white')

    plt.show()

  

plot_momentum_weights(t=10)
```

## Diagram 2

```python
import numpy as np

import matplotlib.pyplot as plt

  

# --- 1. Setup Simulation ---

def loss_function(x):

    return x**2

  

def gradient(x):

    return 2 * x

  

# Parameters causing overshoot

x_start = -4.0  

velocity = 0.0    

gamma = 0.9        

learning_rate = 0.1

iterations = 50

  

# Data containers

path_x = [x_start]

path_y = [loss_function(x_start)]

  

# Run Simulation

x_current = x_start

for _ in range(iterations):

    velocity = gamma * velocity + learning_rate * gradient(x_current)

    x_current = x_current - velocity

    path_x.append(x_current)

    path_y.append(loss_function(x_current))

  
  

# --- 2. Visualization ---

plt.figure(figsize=(10, 6), facecolor='#1e1e1e')

ax = plt.gca()

ax.set_facecolor('#1e1e1e')

  

# Plot the Red Loss Landscape

x_grid = np.linspace(-5, 5, 100)

plt.plot(x_grid, loss_function(x_grid), color='red', linewidth=3, label='Loss Surface')

  

# --- Modified Path Plotting (Segmented Colors) ---

# We iterate through the points and plot segments individually.

# If the next point (x_next) is greater than 0 (opposite side from start),

# we color it yellow to indicate overshoot.

  

overshoot_color = '#FFD700' # Gold/Yellow

normal_color = 'white'

  

for i in range(len(path_x) - 1):

    x_curr, x_next = path_x[i], path_x[i+1]

    y_curr, y_next = path_y[i], path_y[i+1]

    # Determine color based on position.

    # Since we start at x=-4, crossing to x>0 is the overshoot.

    if x_next > 0.1: # Using 0.1 threshold to avoid coloring exactly at 0

        seg_color = overshoot_color

        lw = 3 # Make overshoot slightly thicker

    else:

        seg_color = normal_color

        lw = 2

    plt.plot([x_curr, x_next], [y_curr, y_next],

             color=seg_color, marker='o', markersize=5,

             linestyle='-', linewidth=lw, alpha=0.9)

  

# Add dummy plots for legend (since segmented plots confuse auto-legend)

plt.plot([], [], color=normal_color, marker='o', label='Descent / Settling')

plt.plot([], [], color=overshoot_color, linewidth=3, marker='o', label='Overshoot Phase')

  

# --- Annotations & Formatting ---

plt.annotate('Start', xy=(path_x[0], path_y[0]), xytext=(-4.5, 18),

             color=normal_color, arrowprops=dict(facecolor=normal_color, arrowstyle='->'))

  

# Find the peak of the first overshoot for annotation

overshoot_peak_idx = np.argmax(path_x[0:15])

plt.annotate('Overshoot Peak\n(High Velocity)',

             xy=(path_x[overshoot_peak_idx], path_y[overshoot_peak_idx]),

             xytext=(2, 10), color=overshoot_color, fontweight='bold',

             arrowprops=dict(facecolor=overshoot_color, arrowstyle='->'))

  
  

plt.title(f'Momentum Overshoot Visualized ($\gamma={gamma}$)', color='white', fontsize=14)

plt.xlabel('Parameter $\\theta$', color='white')

plt.ylabel('Loss', color='white')

  

ax.spines['bottom'].set_color('red')

ax.spines['left'].set_color('red')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.tick_params(colors='white')

  

plt.legend(facecolor='#333333', labelcolor='white')

plt.grid(True, linestyle='--', alpha=0.2)

plt.show()
```