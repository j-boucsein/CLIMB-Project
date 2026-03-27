import numpy as np
from scipy.stats import qmc
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


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


def lhs_simplex_with_bounds(n_samples, a_bounds, b_bounds, c_bounds):
    samples = []
    n_loops = []

    while len(samples) < n_samples:
        # LHS in 2D → für Stick-Breaking
        sampler = qmc.LatinHypercube(d=2)
        u = sampler.random(n=round(n_samples*1.0))
        u_sorted = np.sort(u, axis=1)

        a = u_sorted[:, 0]
        b = u_sorted[:, 1] - u_sorted[:, 0]
        c = 1 - u_sorted[:, 1]

        # Alle Samples gleichzeitig filtern (vektorisiert)
        mask = (
            (a_bounds[0] <= a) & (a <= a_bounds[1]) &
            (b_bounds[0] <= b) & (b <= b_bounds[1]) &
            (c_bounds[0] <= c) & (c <= c_bounds[1])
        )

        filtered = np.stack([a[mask], b[mask], c[mask]], axis=1)
        # print(np.stack([a[mask], b[mask], c[mask]], axis=1).shape)
        samples.extend(filtered.tolist())
        n_loops.append(len(filtered.tolist()))

    return np.array(samples[:n_samples]), n_loops


def sample_other_params_lhs(n_samples, bounds):
    d = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    return scaled


n_samples = 50
indexes = ["Omega_m", "Omega_b", "Omega_Lambda",
           "Hubble_parameter"] #, "SeedBlackHoleMass", "WindEnergyIn1e51erg"]

# Sample erstellen
Oc_bounds = (0.1, 0.4)
Ob_bounds = (0.001, 0.1)
OL_bounds = (0.5, 0.9)  # Attention: this bound is hard coded in the functions. Changing it here will not have the wanted effect
h_bounds = (0.55, 0.85)
# SBHM_bounds = (1e-6, 1e-2)
# WE_bounds = (0.1, 10.0)

current_best_sum = 0
current_best_points = []
sum_plot = []
n_loops_sol = 0

for i in range(10000):
    abc, n_loops = lhs_simplex_with_bounds(n_samples, Oc_bounds, Ob_bounds, OL_bounds)
    abc[:, 0] = abc[:, 0] + abc[:, 1]  # calculate Omega_m from Omega_c and Omega_b

    others = sample_other_params_lhs(n_samples, np.array([h_bounds])) #, SBHM_bounds, WE_bounds]))

    final_samples = np.hstack([abc, others])

    sum_of_distances = sum_of_nearest_neighbor_distances(final_samples)

    if sum_of_distances > current_best_sum:
        current_best_points = final_samples
        current_best_sum = sum_of_distances
        sum_plot.append(sum_of_distances)
        n_loops_sol = n_loops


df = pd.DataFrame(current_best_points, columns=indexes)

print(df.head())

print(current_best_points.shape)
print(current_best_sum)
print(n_loops_sol)

df.to_csv("grid_lhs_constrained.csv", sep=",", index=False)

sum_plot = [i/sum_plot[0]*100 for i in sum_plot]

plt.plot([i for i in range(len(sum_plot))], sum_plot, marker="o", linestyle="None")
plt.xlabel("Number of improved grids")
plt.ylabel("Sum of Eucledian distances [%]")
plt.savefig("plots/grid_sample_improvemnt.pdf", format="PDF", dpi=300)
# plt.show()
