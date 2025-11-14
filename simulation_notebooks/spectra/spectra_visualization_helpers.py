import h5py
import glob
import os

from scale_factor_helpers import *


def load_spectra_from_boxes(gp_paths, spectrumNums):
    """ Loads all spectra specified in spectrumNums for all boxes specified in gp_paths

    Args:
        gp_paths (list): List of Gridpoint folder paths to load spectra from.
        spectrumNums (list): List of indices for the spectra.

    Returns:
        wavelengths (list): one np.array containing the wavelengths of the spectra for each box
        fluxes (list): len(spectrumNums) number of np.arrays for each box (each is one spectrum)
        params (list): [Omega0, OmegaBaryon, OmegaLambda, HubbleParam] for each box
    """
    
    wavelengths = []
    fluxes = []
    params = []
    
    for gp_path in gp_paths:
        gp_name = gp_path.split("/")[-2]
        path = gp_path + f"data.files/spectra/spectra_{gp_name}_z2.0_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"

        with h5py.File(path, "r") as f:     
            # Zugriff auf einzelne Datasets
            wavelengths.append(f["wave"][:])  # Das [:] liest das gesamte Dataset in ein NumPy-Array ein
            fluxes.append(f["flux"][:][spectrumNums])

        outdir_path = gp_path + "output/"
        Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(outdir_path)

        params.append([Omega0, OmegaBaryon, OmegaLambda, HubbleParam])

    return wavelengths, fluxes, params


def get_cosmo_parameters(basepath):
    """ Returns the cosmological parameters for one simulation

    Args:
        basepath (string): path to the output folder of the simulation

    Returns:
        Omega0 (float): Energydensity of matter in the simulation
        OmegaBaryon (float): Energydensity of Baryons in the simulation
        OmegaLambda (float): Energydensity of Dark Energy in the simulation
        HubbleParam (float): h in the simulation
    """
    path = basepath+"txt-files/parameters-usedvalues"
    Omega0 = None
    OmegaLambda = None
    HubbleParam = None
    OmegaBaryon = None
    
    with open(path, "r") as f:
        for line in f:
            if "Omega0" in line:
                Omega0 = float(line.split()[-1])
            if "OmegaBaryon" in line:
                OmegaBaryon = float(line.split()[-1])
            if "OmegaLambda" in line:
                OmegaLambda = float(line.split()[-1])
            if "HubbleParam" in line:
                HubbleParam = float(line.split()[-1])
    
    return Omega0, OmegaBaryon, OmegaLambda, HubbleParam


def redshift_wavelength_forward(z, wavelength):
    """ Redshifts a wavelength forward (wavelength gets bigger with more redshift)
    
    Args:
        z (float): redshift to apply to wavelength
        wavelength (float or np.array): wavelength to be shifted

    Returns:
        (float or np.array): redshifted wavelength
    """
    return (z+1)*wavelength


def redshift_wavelength_backward(z, wavelength):
    """ Redshifts a wavelength backward (wavelength gets smaller with more redshift)
    
    Args:
        z (float): redshift to apply to wavelength
        wavelength (float or np.array): wavelength to be shifted

    Returns:
        (float or np.array): redshifted wavelength
    """
    return wavelength/(z+1)


def get_data_from_fof_folder(output_dir, snapN, param1, param2):
    snapdir = glob.glob(output_dir+f"output/groups_*{snapN}")[0]
    snap_files = os.listdir(snapdir)
    
    all_values = None

    for file_name in snap_files:
        file_path = snapdir+f"/{file_name}"

        with h5py.File(file_path, "r") as f:
            values_this_file = f[f'{param1}/{param2}'][:]
            
            if all_values is None:
                all_values = values_this_file
            else:
                all_values = np.concatenate((all_values, values_this_file))

    return all_values


def get_data_from_snap_folder(output_dir, snapN, param1, param2):
    snapdir = glob.glob(output_dir+f"snapdir_*{snapN}")[0]
    snap_files = os.listdir(snapdir)
    
    all_values = None

    for file_name in snap_files:
        file_path = snapdir+f"/{file_name}"

        with h5py.File(file_path, "r") as f:
            values_this_file = f[f'{param1}/{param2}'][:]
            
            if all_values is None:
                all_values = values_this_file
            else:
                all_values = np.concatenate((all_values, values_this_file))
    
    return all_values


def custom_hist_z(outpath, snapN, n_bins=100, axis=2):
    """ Creates 1D gas mass histogram from simulated box
    
    Args:
        outpath (string): path to the output folder of a simulation
        snapN (integer): number of the snapshot to use
        n_bins (integer): number of bins
        axis (integer): axis to use (0: x-axis, 1: y-axis, 2: z-axis)

    Returns:
        bin_masses (np.array): the summed masses of every gas cell for each bin
        bin_edges (np.array): the edges of the bins (this will be n+1 edges if n bins)
    """

    coordinates = get_data_from_snap_folder(outpath, snapN, "PartType0", "Coordinates")
    masses = get_data_from_snap_folder(outpath, snapN, "PartType0", "Masses")

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
    bin_edges = np.array(bin_edges)

    return bin_masses, bin_edges