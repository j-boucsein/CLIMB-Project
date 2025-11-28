import glob
import numpy as np
import os
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    # print(snap_files)
    
    all_values = None

    for file_name in snap_files:
        # print("-------------- File begin ----------------")
        file_path = snapdir+f"/{file_name}"
        # print(file_path)

        with h5py.File(file_path, "r") as f:
            # header = f['Header']
            # for key, val in header.attrs.items():
                # print(f"{key}: {val}")
            values_this_file = f[f'{param1}/{param2}'][:]
            
            if all_values is None:
                all_values = values_this_file
            else:
                all_values = np.concatenate((all_values, values_this_file))
            
            # print(values_this_file.shape)
            # print(all_values.shape)
        
        # print("--------------- File end ------------------")
    
    return all_values


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
        basepath (string): path to the output folder of the simulation

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


def find_centeral_bh(path, snapN, zoom_in_box_middle, zoom_in_box_size):
    """ find the ceneral black hole for a given box within a simulation

    Args:
        path (string): base directory of the simulation
        snapN (int): number of the snapshot
        zoom_in_box_middle (np.array): coordinates of center of the box in which to search for the BH
        zoom_in_box_size (float): half the length of the box to seach for the box (will search from -zoom_in_box_size to zoom_in_box_size)

    Returns:
        np.array: coordinates of the most massive black hole within the searched box
    """

    particle_type = "PartType5"
    coordinates = get_data_from_snap_folder(path, snapN, particle_type, "Coordinates")
    boxsize = get_data_from_header(path, snapN, 'BoxSize') # in kpc
    
    # transform into periodic box with center at zoom_in_box_middle
    center_coordinates = (coordinates - zoom_in_box_middle + boxsize / 2) % boxsize - boxsize / 2
    
    # only look at objects within a box centered around 0 with zoom_in_box_size as length, heigt and width 
    center_coordinates = center_coordinates[center_coordinates[:, 0] < zoom_in_box_size]
    center_coordinates = center_coordinates[center_coordinates[:, 0] > -zoom_in_box_size]
    center_coordinates = center_coordinates[center_coordinates[:, 1] < zoom_in_box_size]
    center_coordinates = center_coordinates[center_coordinates[:, 1] > -zoom_in_box_size]
    center_coordinates = center_coordinates[center_coordinates[:, 2] < zoom_in_box_size]
    center_coordinates = center_coordinates[center_coordinates[:, 2] > -zoom_in_box_size]
    
    # center_coordinates is ordered by mass, so center_coordinates[0] is the most massive BH in the box
    # then transform back to original coordinates
    centeral_bh_coords = (center_coordinates[0] + zoom_in_box_middle + boxsize / 2) % boxsize - boxsize / 2
    
    return centeral_bh_coords


def find_index_of_closest(cms_list, reference_cms):
    """ finds the index of the coordinates in cms_list which are closest to the reference_cms coordinates

    Args:
        cms_list (np.array): list of coordinates
        reference_cms (np.array): coordinates of reference point

    Returns:
        int: index of coordinates in cms_list which are closest to the reference_cms coordinates
    """

    min_distance = np.inf
    min_dist_index = None
    for i in range(len(cms_list)):
        x, y, z = cms_list[i]
        x_ref, y_ref, z_ref = reference_cms
        distance = np.sqrt((x-x_ref)**2 + (y-y_ref)**2 + (z-z_ref)**2)
        
        if distance < min_distance:
            min_distance = distance
            min_dist_index = i

    return min_dist_index


def subhalo_projection(path, snapN, particle_type="PartType4", zomm_in_box_size=500, look_for_centeral_bh_radius=500, reference_subhalo_cms=None):
    """ finds coordinates and masses of particles within a box around the centeral particle of a halo clostest to reference_subhalo_cms with length zomm_in_box_size
        Also finds the Radius with 500 times the critical density of the universe.

        Note: the variable names in this function can be very missleading, sorry :)

    Args:
        path (string): base directory of the simulation
        snapN (int): number of the snapshot
        particle_type (str, optional): the particle type can either be "PartType4" for stars, "PartType0" for gas or "PartType1" for dm. Defaults to "PartType4".
        zomm_in_box_size (int, optional): half the width of the box. Defaults to 500.
        look_for_centeral_bh_radius (int, optional): radius in which to look for the centeral black hole. This parameter is not used anymore. Defaults to 500.
        reference_subhalo_cms (np.array, optional): coordinates which to find the closest halo to. Defaults to None.

    Returns:
        cms_center_coordinates (np.array): list of coordinates of particles within the box
        masses (np.array): list of masses for each particle within the box
        biggest_subhalo_halfmassrad (float): Radius with 500 times the critical density of the universe for that halo
    """
    
    subhalo_mass = get_data_from_fof_folder(path, snapN, "Group", "GroupMass") * 1e10 # in M_sun

    index_biggest_halo = max(range(len(subhalo_mass)), key=lambda i: subhalo_mass[i])
    bigges_subhalo = subhalo_mass[index_biggest_halo]

    subhalo_cms = get_data_from_fof_folder(path, snapN, "Group", "GroupCM")
    if reference_subhalo_cms is not None:
        index_target_halo = find_index_of_closest(subhalo_cms, reference_subhalo_cms)
        target_subhalo_cms = subhalo_cms[index_target_halo]
    else:
        index_target_halo = index_biggest_halo
        target_subhalo_cms = subhalo_cms[index_target_halo]

    subhalo_halfmassrad = get_data_from_fof_folder(path, snapN, "Group", "Group_R_Crit500")
    biggest_subhalo_halfmassrad = subhalo_halfmassrad[index_target_halo]
    
    #centeral_galaxy_bh = find_centeral_bh(path, snapN, target_subhalo_cms, look_for_centeral_bh_radius)
    centeral_galaxy_bh = get_data_from_fof_folder(path, snapN, "Group", "GroupPos")
    centeral_galaxy_bh = centeral_galaxy_bh[index_target_halo]

    boxsize = get_data_from_header(path, snapN, 'BoxSize') # in kpc
    
    # transform coordinates to center biggest_halo_cms
    coordinates = get_data_from_snap_folder(path, snapN, particle_type, "Coordinates")
    if particle_type == "PartType0" or particle_type == "PartType4":
        masses = get_data_from_snap_folder(path, snapN, particle_type, "Masses")*1e10 # in M_sun
    elif particle_type == "PartType1":
        mass_temp = get_data_from_header(path, snapN, "MassTable")[1]*1e10 # in M_sun
        masses = np.zeros(len(coordinates)) + mass_temp
    else:
        assert False, "Error: particle_type must be one of [PartType0, PartType1, PartType4]"

    # transform to periodic coordinates centered around centeral_galaxy_bh
    cms_center_coordinates = coordinates - centeral_galaxy_bh
    cms_center_coordinates = (cms_center_coordinates + boxsize / 2) % boxsize - boxsize / 2
    
    masses = masses[cms_center_coordinates[:, 0] < zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 0] < zomm_in_box_size]
    masses = masses[cms_center_coordinates[:, 0] > -zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 0] > -zomm_in_box_size]
    masses = masses[cms_center_coordinates[:, 1] < zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 1] < zomm_in_box_size]
    masses = masses[cms_center_coordinates[:, 1] > -zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 1] > -zomm_in_box_size]
    masses = masses[cms_center_coordinates[:, 2] < zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 2] < zomm_in_box_size]
    masses = masses[cms_center_coordinates[:, 2] > -zomm_in_box_size]
    cms_center_coordinates = cms_center_coordinates[cms_center_coordinates[:, 2] > -zomm_in_box_size]
    
    
    return cms_center_coordinates, masses, biggest_subhalo_halfmassrad


def create_circle(radius, res=1000):
    """create res number of points on a circle with radius around 0

    Args:
        radius (float): radius of the circle
        res (int, optional): number of points to generate. Defaults to 1000.

    Returns:
        x (np.array): x coordinates of the points
        y (np.array): y coordinates of the points
    """

    par = np.linspace(0, 2*np.pi, res)

    x = radius * np.cos(par)
    y = radius * np.sin(par)

    return x, y
    


if __name__ == "__main__":
    x, y = create_circle(2)

    #plt.plot(x, y)
    #plt.savefig("plots/test_plot.pdf", format="PDF")
    testpath = "/vera/u/jerbo/my_ptmp/L25n256_suite/gridpoint0/"
    test_snapN = 5
    subhalo_projection(testpath, test_snapN, reference_subhalo_cms=[20000, 20000, 20000])