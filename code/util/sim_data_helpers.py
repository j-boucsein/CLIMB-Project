import h5py
import os
import glob
import numpy as np


def get_data_from_header(output_dir, snapN, param='BoxSize'):
    """ get data from the header of a specified simulation and snapshot
    
    Args:
        output_dir (string): base directory of the simulation
        snapN (int): number of the snapshot
        param (string): parameter of the hdf5 file (the parameter to read)

    Returns:
        np.array : Array containing the specific data
    """

    snapdir = glob.glob(output_dir+f"output/snapdir_*{snapN}")[0]
    snap_files = os.listdir(snapdir)
    file_name = snap_files[0]
    file_path = snapdir+f"/{file_name}"

    with h5py.File(file_path, "r") as f:
        header = f['Header']
        return_param = header.attrs[param]
        
    return return_param


def get_cosmo_parameters(basepath):
    """ Returns the cosmological parameters for one simulation

    Args:
        basepath (string): base directory of the simulation

    Returns:
        Omega0 (float): Energydensity of matter in the simulation
        OmegaBaryon (float): Energydensity of Baryons in the simulation
        OmegaLambda (float): Energydensity of Dark Energy in the simulation
        HubbleParam (float): h in the simulation
    """
    path = basepath+"output/txt-files/parameters-usedvalues"
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


def get_data_from_fof_folder(output_dir, snapN, param1, param2):
    """ function to get data from a specific group folder

    Args:
        output_dir (string): base directory of the simulation
        snapN (int): number of the snapshot
        param1 (string): first parameter in the hdf5 file (i.e. "Halos")
        param2 (string): second paramter in the hdf5 file (i.e. "GroupMass")

    Returns:
        np.array: Array containing the specified data
    """
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
    """ function to get data from a specific snapshot folder

    Args:
        output_dir (string): base directory of the simulation
        snapN (int): number of the snapshot
        param1 (string): first parameter in the hdf5 file (i.e. "PartType0")
        param2 (string): second paramter in the hdf5 file (i.e. "Coordinates")

    Returns:
        np.array: Array containing the specified data
    """
    snapdir = glob.glob(output_dir+f"output/snapdir_*{snapN}")[0]
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