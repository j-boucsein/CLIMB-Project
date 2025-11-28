import glob
import numpy as np
import os
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


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

    snapdir = glob.glob(output_dir+f"output/snapdir_*{snapN}")[0]
    snap_files = os.listdir(snapdir)
    file_name = snap_files[0]
    file_path = snapdir+f"/{file_name}"

    with h5py.File(file_path, "r") as f:
        header = f['Header']
        return_param = header.attrs[param]
        
    return return_param


def get_cosmo_parameters(basepath):
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