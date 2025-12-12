import numpy as np

# This is not super pretty, but I think this is the best way to import stuff from ..util?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.sim_data_helpers import get_data_from_snap_folder, get_data_from_header


def custom_hist_z(outpath, snapN, n_bins=100, axis=2):
    """ Creates 1D gas mass histogram from simulated box
    
    Args:
        outpath (string): path to the gridpoint base folder of a simulation
        snapN (integer): number of the snapshot to use
        n_bins (integer): number of bins
        axis (integer): axis to use (0: x-axis, 1: y-axis, 2: z-axis)

    Returns:
        bin_masses (np.array): the summed masses of every gas cell for each bin
        bin_edges (np.array): the edges of the bins (this will be n+1 edges if n bins)
    """

    coordinates = get_data_from_snap_folder(outpath, snapN, "PartType0", "Coordinates")
    masses = get_data_from_snap_folder(outpath, snapN, "PartType0", "Masses")
    box_size = get_data_from_header(outpath, snapN)

    min_coordinate = coordinates[:,axis].min()
    max_coordinate = coordinates[:,axis].max()

    bin_width = (max_coordinate - min_coordinate)/n_bins
    bin_edges = [min_coordinate+bin_width*i for i in range(n_bins+1)]

    bin_masses = []
    # get slices
    for i in range(n_bins):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i+1]

        slice_map = (coordinates[:,axis] > lower_edge) & (coordinates[:,axis] < upper_edge)
        slice_masses = masses[slice_map]

        slice_total_mass = np.sum(slice_masses)

        bin_masses.append(slice_total_mass)

    bin_masses = np.array(bin_masses)
    bin_masses = bin_masses/(box_size*box_size*bin_width) *1e10 # in (M_sun/h) / (cMpc/h)^3

    bin_edges = np.array(bin_edges)

    return bin_masses, bin_edges