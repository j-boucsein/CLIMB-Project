import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix


# Generate fully random samples in [a_min, a_max] x [b_min, b_max], filter for valid ones
def random_simplex_samples(n, a_bounds, b_bounds):
    a_min, a_max = a_bounds
    b_min, b_max = b_bounds

    a_samples = []
    b_samples = []
    c_samples = []

    while len(a_samples) < n:
        a_rand = np.random.uniform(a_min, a_max, size=n)
        b_rand = np.random.uniform(b_min, b_max, size=n)
        c_rand = 1 - a_rand - b_rand

        # Keep only valid ones (where c >= 0)
        mask = c_rand >= 0

        a_samples = [*a_samples, *a_rand[mask]]
        b_samples = [*b_samples, *b_rand[mask]]
        c_samples = [*c_samples, *c_rand[mask]]

    return a_samples[:n], b_samples[:n], c_samples[:n]


def random_points_sample(n, a_bounds): # , b_bounds, c_bounds):
    a_min, a_max = a_bounds
    # b_min, b_max = b_bounds
    # c_min, c_max = c_bounds
    a_rand = np.random.uniform(a_min, a_max, size=n)
    # b_rand = np.random.uniform(b_min, b_max, size=n)
    # c_rand = np.random.uniform(c_min, c_max, size=n)

    return a_rand # , b_rand, c_rand


# Metrik:
def sum_of_nearest_neighbor_distances(points, scale=True):
    if scale:
        # scale all columns to [0, 1] to remove bias due to bigger intervals in bounds
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        scaled_points = (points - min_vals) / (max_vals - min_vals)
    else:
        scaled_points = points

    # Compute full pairwise distance matrix
    D = distance_matrix(scaled_points, scaled_points)

    # Set diagonal to infinity to ignore self-distance (0)
    np.fill_diagonal(D, np.inf)

    # Get the minimum (nearest neighbor) for each point
    min_distances = np.min(D, axis=1)

    # Sum of all nearest-neighbor distances
    return np.sum(min_distances)


n_iterations = 10000

Oc_bounds = (0.1, 0.4)
Ob_bounds = (0.001, 0.1)
OL_bounds = (0.5, 0.9)
h_bounds = (0.55, 0.85)
# SBHM_bounds = (1e-6, 1e-2)
# WE_bounds = (0.1, 10.0)

current_best_sum = 0
current_best_points = []
sum_plot = []

for i in range(n_iterations):
    # Get random samples
    a_samp, b_samp, c_samp = random_simplex_samples(50, Oc_bounds, Ob_bounds)
    a_samp = [sum(i) for i in [*zip(a_samp, b_samp)]]  # calculate Omega_m from Omega_c and Omega_b

    # d_samp, e_samp, f_samp = random_points_sample(50, h_bounds, SBHM_bounds, WE_bounds)
    d_samp = random_points_sample(50, h_bounds)
    # sample = np.stack([a_samp, b_samp, c_samp, d_samp, e_samp, f_samp], axis=1)
    sample = np.stack([a_samp, b_samp, c_samp, d_samp], axis=1)

    sum_of_distances = sum_of_nearest_neighbor_distances(sample)

    if sum_of_distances > current_best_sum:
        current_best_points = sample
        current_best_sum = sum_of_distances
        sum_plot.append(sum_of_distances)

print(current_best_sum)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

####################### Plot Surface #######################
# Einschränkungen
a_min, a_max = 0.1, 0.4                 # A = Omega_cold-darkmatter
b_min, b_max = 0.001, 0.1               # B = Omega_baryon
c_min, c_max = 0.5, 0.9                 # C = Omega_Lambda

# Auflösung
n_grid = 100

# Gitter erzeugen
a_vals = np.linspace(a_min, a_max, n_grid)
b_vals = np.linspace(b_min, b_max, n_grid)
A, B = np.meshgrid(a_vals, b_vals)
C = 1 - A - B

A = A + B

# Ungültige Bereiche maskieren (wo c < 0 → np.nan)
C[(C > c_max) | (C < c_min)] = np.nan

# Glatte Fläche ohne Gitterstruktur
ax.plot_surface(A, B, C, alpha=0.4, color='gray', rstride=1, cstride=1, edgecolor='none')

####################### Plot fully random sample #######################

ax.scatter(a_samp, b_samp, c_samp, color='red', alpha=0.6, label='Random Samples')

####################### Set Options for Plot #######################

# Achsen
ax.set_xlabel(r'$\Omega_m$')
ax.set_ylabel(r'$\Omega_b$')
ax.set_zlabel(r'$\Omega_\Lambda$')
ax.set_title(r'Fully random Sampling with constraint'
             r' of $\Omega_m$+$\Omega_\Lambda$=1 $\wedge$ $\Omega_m \geq \Omega_b$' + '\n'
             r'and $\Omega_m$ $\in$ [0.1, 0.5], $\Omega_b$ $\in$ [0.001, 0.2], $\Omega_\Lambda$ $\in$ [0.5, 0.9]')

# Achsengrenzen
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.15)
ax.set_zlim(0.5, 1)

plt.tight_layout()

# Setze Kameraperspektive
ax.view_init(elev=34, azim=17)

# Speichern
plt.savefig("plots/Fully_random_sampling.pdf", format="PDF", dpi=300)
plt.show()
