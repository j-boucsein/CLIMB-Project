import h5py
import pandas as pd
import random


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


def load_dataset(gp_basepaths, load_n_spect_per_file=1000, total_spectra_per_file=10000):
    
    df = pd.DataFrame()
    first_file = True
    for gp_path in gp_basepaths:
        gp_name = gp_path.split("/")[-2]
        out_path = gp_path + "output/"
        path = gp_path + f"data.files/spectra/spectra_{gp_name}_z2.0_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"

        spectra_Ns = random.sample(range(0, total_spectra_per_file), load_n_spect_per_file)

        data = []
        with h5py.File(path, "r") as f:
            if first_file:
                data.append(f["wave"][:])

            for i in f["flux"][:][spectra_Ns]:
                data.append(i)

        column_names = [f"lambda {i}" for i in range(len(data[0]))]
        df_this_file = pd.DataFrame(data, columns=column_names)

        Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(out_path)

        df_this_file["Omega m"] = Omega0
        df_this_file["Omega b"] = OmegaBaryon
        df_this_file["Omega Lambda"] = OmegaLambda
        df_this_file["h"] = HubbleParam

        if first_file:
            df_this_file.at[0, "Omega m"] = None
            df_this_file.at[0, "Omega b"] = None
            df_this_file.at[0, "Omega Lambda"] = None
            df_this_file.at[0, "h"] = None

            first_file = False
        
        df = pd.concat([df, df_this_file], axis=0, ignore_index=True)

    return df


if __name__ == "__main__":
    n_spectra_each_box = 10000
    gps_list = [i for i in range(50)][:5]
    base_path = "/vera/u/jerbo/my_ptmp/L25n256_suite"
    gp_paths = [base_path+f"/gridpoint{i}/" for i in gps_list]
    df = load_dataset(gp_paths)
    print(df)