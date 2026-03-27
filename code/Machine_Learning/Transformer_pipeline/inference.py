import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dataset_functions import get_sdss_spectra, get_shuffled_and_split_datasets, get_spectra_reference_point
from transformer_model import Transformer
from plotting_functions import make_corner_plot


def get_sdss_spectra_for_inference(cat_path, resid_file, snr_filter):
    """ Collect the SDSS spectra above a certain snr for inference

    Args:
        cat_path (string): path to the custom catalogue file
        resid_file (string): path to the resid file
        snr_filter (float): signal to noise ratio that is used as lower filter

    Returns:
        torch.Tensor: Tensor containing the SDSS spectra
    """
    cat = np.load(cat_path)
    snrs_cat = cat["SNR"]
    pmfs_cat = cat["PMF"]

    pmf_list = []
    for i in range(len(snrs_cat)):
        if snrs_cat[i] > snr_filter:
            pmf_list.append(pmfs_cat[i])

    print(len(pmf_list))

    # Note: this function might take a couple of minutes to run as it has to open thousands of data files
    _, _, fluxes_boss_specs = get_sdss_spectra(resid_file, pmf_list)

    np_specs = []
    for i, spec in enumerate(fluxes_boss_specs):
        if len(spec) == 402:
            np_specs.append(np.append(spec, spec[-1]))

    specs = torch.Tensor(np.array(np_specs))

    return specs


def initialize_trafo_from_saved_state(config_path, len_in, len_out, state_path, device):
    """ Initializes Transformer from a saved state

    Args:
        config_path (string): path to the config file
        len_in (int): input dimension of transformer
        len_out (int): ouptut dimension of transformer
        state_path (string): path to the state file that should be loaded
        device (torch.device): cpu or cuda

    Returns:
        model (Transformer): Transformer model object
        criterion (torch.nn.MSELoss): Loss function object
        optimizer (torch.optim.AdamW): optimizer object
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']

    d_model = params['model']['d_model']
    num_heads = params['model']['num_heads']
    num_layers = params['model']['num_layers']
    d_ff = params['model']['d_ff']
    dropout = params['model']['dropout']

    model = Transformer(len_in, len_out, d_model, num_heads, num_layers, d_ff, dropout).to(device)

    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer


def eval_model(model, X, device):
    """ Does model forward pass with given data X

    Args:
        model (Transformer): Transformer model object
        X (torch.Tensor): The input data
        device (torch.device): cpu or cuda

    Returns:
        torch.Tensor: The predicted output of the model
    """
    model.eval()

    with torch.no_grad():
        X = X.to(device)

        y_pred = model(X)

    return y_pred


def main():
    model_name = "realistic_noise_model_snr2"
    suite_of_spectra = "L25n256_realistic_noise_v2_snr2"
    snr_filter = 2

    cat_path = "SDSS_support_files/Custom_cat.npz"
    resid_file = "SDSS_support_files/residcorr_v5_4_45.dat"
    config_path = f"log_files/{model_name}_config.yaml"
    state_path = f"model_states/{model_name}_weights.pt"
    sdss_corner_path = f"plots/{model_name}_SDSS_cornerplot.pdf"
    ref_corner_path = f"plots/{model_name}_refbox_cornerplot.pdf"

    sdss_specs = get_sdss_spectra_for_inference(cat_path, resid_file, snr_filter)
    n_specs = sdss_specs.shape[0]
    if n_specs > 10000:
        n_specs = 10000
    ref_box_specs, _ = get_spectra_reference_point(suite_of_spectra, n_spectra=n_specs)
    ref_box_specs = torch.Tensor(ref_box_specs)

    input_len = sdss_specs.shape[1]
    output_len = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = initialize_trafo_from_saved_state(config_path, input_len, output_len, state_path, device)

    y_pred_sdss = eval_model(model, sdss_specs, device)
    y_pred_ref = eval_model(model, ref_box_specs, device)

    # TODO: This is an embarrassingly inefficient way to get the y_mean and y_std. Should probably change this in the future
    _, _, _, y_mean, y_std = get_shuffled_and_split_datasets(suite_of_spectra, True)

    y_pred_sdss, y_pred_ref = y_pred_sdss*y_std + y_mean, y_pred_ref*y_std + y_mean
    y_pred_sdss, y_pred_ref = y_pred_sdss.numpy(), y_pred_ref.numpy()

    make_corner_plot(y_pred_sdss, sdss_corner_path, True)
    make_corner_plot(y_pred_ref, ref_corner_path, True)


if __name__ == "__main__":
    main()