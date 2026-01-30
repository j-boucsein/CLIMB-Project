import numpy as np
import matplotlib.pyplot as plt

def latin_hypercube_sampling(n, d=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    samples = np.zeros((n, d))

    for dim in range(d):
        bins = np.linspace(0, 1, n + 1)
        points = rng.uniform(bins[:-1], bins[1:])
        rng.shuffle(points)
        samples[:, dim] = points

    return samples

# -------------------------
# Parameters
# -------------------------
n = 9
seed = 14  # 4 extreme clustering for random, 14 a bit less extreme but still nicely visible clustering for random
rng = np.random.default_rng(seed)

grid_size = int(np.ceil(np.sqrt(n)))

# -------------------------
# Sampling methods
# -------------------------
random_samples = rng.random((n, 2))

# Uniform grid (truncate to n points) — deterministic, no RNG needed
x = np.linspace(0.1, 0.9, grid_size)
y = np.linspace(0.1, 0.9, grid_size)
X, Y = np.meshgrid(x, y)
grid_samples = np.column_stack([X.ravel(), Y.ravel()])[:n]

# Latin Hypercube
lhs_samples = latin_hypercube_sampling(n, d=2, rng=rng)

# -------------------------
# Plotting
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)

titles = [
    "Random Sampling",
    "Uniform Grid Sampling",
    "Latin Hypercube Sampling"
]

for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Remove tick labels (keep ticks for grid lines)
    ax.tick_params(labelbottom=False, labelleft=False)

# --- Random (no grid) ---
axes[0].scatter(random_samples[:, 0], random_samples[:, 1], s=50)

# --- Uniform grid (coarse grid) ---
axes[1].scatter(grid_samples[:, 0], grid_samples[:, 1], s=50)

ticks_grid = np.linspace(0, 1, grid_size)
axes[1].set_xticks(ticks_grid)
axes[1].set_yticks(ticks_grid)
axes[1].grid(True, linestyle="--", alpha=0.6)

# --- LHS (fine grid: n x n) ---
axes[2].scatter(lhs_samples[:, 0], lhs_samples[:, 1], s=50)

ticks_lhs = np.linspace(0, 1, n + 1)
axes[2].set_xticks(ticks_lhs)
axes[2].set_yticks(ticks_lhs)
axes[2].grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/comparison_plot.pdf", format="PDF")
