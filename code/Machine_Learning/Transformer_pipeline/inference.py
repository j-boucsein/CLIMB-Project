import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dataset_functions import get_shuffled_and_split_datasets, get_spectra_reference_point
from transformer_model import Transformer
from plotting_functions import make_corner_plot


def get_sdss_spectra_for_inference(cache_path):

    with np.load(cache_path) as npz:
        data_sdss = {
            "wavelength": npz["wavelength"],
            "flux": npz["flux"],
            "mask": npz["mask"],
            "snr": npz["snr"]
            }

    specs = torch.Tensor(data_sdss["flux"])
    mask = torch.from_numpy(data_sdss["mask"]).to(torch.bool) # need to retain the datatype, so cant use torch.Tensor

    assert torch.isfinite(specs[mask]).all(), "X contains NaN or Inf values in the valid pixels"

    return specs, mask


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


def eval_model(model, X, X_mask, device):
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
        X_mask = X_mask.to(device)

        y_pred = model(X, X_mask)

    return y_pred.cpu()


def main():
    model_name = "test_model"
    suite_of_spectra = "ML_training_data_test"

    cache_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/spectra_filtered_cache.npz"
    config_path = f"log_files/{model_name}_config.yaml"
    state_path = f"model_states/{model_name}_weights.pt"
    sdss_corner_path = f"plots/{model_name}_SDSS_cornerplot.pdf"
    ref_corner_path = f"plots/{model_name}_refbox_cornerplot.pdf"

    sdss_specs, sdss_masks = get_sdss_spectra_for_inference(cache_path)
    ref_box_specs, ref_box_masks, _ = get_spectra_reference_point(suite_of_spectra)
    ref_box_specs = torch.Tensor(ref_box_specs)
    ref_box_masks = torch.from_numpy(ref_box_masks).to(torch.bool) # need to retain the datatype, so cant use torch.Tensor

    assert sdss_specs.shape[1] == ref_box_specs.shape[1], f"SDSS and reference box spectra have different lengths: {sdss_specs.shape[1]} vs {ref_box_specs.shape[1]}"

    input_len = sdss_specs.shape[1]
    output_len = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = initialize_trafo_from_saved_state(config_path, input_len, output_len, state_path, device)

    y_pred_sdss = eval_model(model, sdss_specs, sdss_masks, device)
    y_pred_ref = eval_model(model, ref_box_specs, ref_box_masks, device)

    with open(config_path) as f:
        params = yaml.safe_load(f)

    # TODO: This is an embarrassingly inefficient way to get the y_mean and y_std. Should probably change this in the future
    _, _, _, y_mean, y_std = get_shuffled_and_split_datasets(suite_of_spectra, params['reduced_dataset'])

    y_pred_sdss, y_pred_ref = y_pred_sdss*y_std + y_mean, y_pred_ref*y_std + y_mean
    y_pred_sdss, y_pred_ref = y_pred_sdss.numpy(), y_pred_ref.numpy()

    make_corner_plot(y_pred_sdss, sdss_corner_path, True)
    make_corner_plot(y_pred_ref, ref_corner_path, True)


if __name__ == "__main__":
    main()