import numpy as np
from scipy.stats import qmc
from scipy.spatial import distance_matrix
import time


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


def random_points_sample(n, a_bounds, b_bounds, c_bounds):
    a_min, a_max = a_bounds
    b_min, b_max = b_bounds
    c_min, c_max = c_bounds
    a_rand = np.random.uniform(a_min, a_max, size=n)
    b_rand = np.random.uniform(b_min, b_max, size=n)
    c_rand = np.random.uniform(c_min, c_max, size=n)

    return a_rand, b_rand, c_rand


n_samples = 50
n_iterations = 100000

# parameters for the sampling
indexes = ["Omega_m", "Omega_b", "Omega_Lambda",
           "Hubble_parameter", "SeedBlackHoleMass",
           "WindEnergyIn1e51erg"]
# Bounds for the sampling
Oc_bounds = (0.1, 0.8)
Ob_bounds = (0.01, 0.5)
OL_bounds = (0.0, 1.0)
h_bounds = (0.4, 1.6)
SBHM_bounds = (1e-6, 1e-2)
WE_bounds = (0.1, 10.0)

################################# LHS sampling #################################

mean_sum_of_distances_lhs = 0
current_best_sum_lhs = 0

start_lhs_sampling = time.time()
for i in range(n_iterations):
    abc, _ = lhs_simplex_with_bounds(n_samples, Oc_bounds, Ob_bounds, OL_bounds)
    abc[:, 0] = abc[:, 0] + abc[:, 1]  # calculate Omega_m from Omega_c and Omega_b

    others = sample_other_params_lhs(n_samples, np.array([h_bounds, SBHM_bounds, WE_bounds]))

    final_samples = np.hstack([abc, others])

    sum_of_distances = sum_of_nearest_neighbor_distances(final_samples)

    mean_sum_of_distances_lhs += sum_of_distances

    if sum_of_distances > current_best_sum_lhs:
        current_best_sum_lhs = sum_of_distances

end_lhs_sampling = time.time()
total_time_lhs = end_lhs_sampling - start_lhs_sampling

mean_sum_of_distances_lhs /= n_iterations

################################# Fully random sampling #################################

mean_sum_of_distances_random = 0
current_best_sum_random = 0

start_random_sampling = time.time()
for i in range(n_iterations):
    a_samp, b_samp, c_samp = random_simplex_samples(50, Oc_bounds, Ob_bounds)
    a_samp = [sum(i) for i in [*zip(a_samp, b_samp)]]  # calculate Omega_m from Omega_c and Omega_b

    d_samp, e_samp, f_samp = random_points_sample(50, h_bounds, SBHM_bounds, WE_bounds)
    sample = np.stack([a_samp, b_samp, c_samp, d_samp, e_samp, f_samp], axis=1)
    sum_of_distances = sum_of_nearest_neighbor_distances(sample)
    mean_sum_of_distances_random += sum_of_distances

    if sum_of_distances > current_best_sum_random:
        current_best_sum_random = sum_of_distances

end_random_sampling = time.time()
total_time_random = end_random_sampling - start_random_sampling

mean_sum_of_distances_random /= n_iterations

#################################                       #################################
print(f"\nRunning both sampling methods for {n_iterations} iterations \n")

print(f"--------- Latin Hypercube Sampling ---------  [took {total_time_lhs:.4f} sec]")
print(f"Mean of sum of distances (closest neighbour): {mean_sum_of_distances_lhs:.4f}")
print(f"Highest sum of distances (closest neighbour): {current_best_sum_lhs:.4f}")
print("")
print(f"------------- Random Sampling --------------  [took {total_time_random:.4f} sec]")
print(f"Mean of sum of distances (closest neighbour): {mean_sum_of_distances_random:.4f}")
print(f"Highest sum of distances (closest neighbour): {current_best_sum_random:.4f}")
