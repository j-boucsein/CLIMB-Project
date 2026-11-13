import os
import numpy as np
import torch
import yaml
from dataset_functions import get_spectra_reference_point, transform_flux, read_hash_mapping
from transformer_model import Transformer
from plotting_functions import make_corner_plot

# folder of this script, so that the files written by train_model.py are found from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sdss_cache_path(dataset_dir):
    """ Get the path of the filtered SDSS spectra that the data pipeline used for the noise and the masks of a dataset

    Args:
        dataset_dir (string): folder of the dataset (the one that contains Configs_used.txt)

    Returns:
        string: path of the spectra_filtered_<hash>.npz file that belongs to the dataset
    """
    sdss_specs_hash, spectra_cache_file = None, None

    with open(os.path.join(dataset_dir, "Configs_used.txt")) as f:
        for line in f:
            if line.startswith("sdss_specs_hash:"):
                sdss_specs_hash = line.split(":", 1)[1].strip()
            if line.startswith("spectra_cache_file:"):
                spectra_cache_file = line.split(":", 1)[1].strip()

    # a dataset without SDSS noise and SDSS masks has no filtered SDSS spectra ("None" / "null" in the file)
    if sdss_specs_hash in (None, "None") or spectra_cache_file in (None, "None", "null"):
        raise ValueError(f"No filtered SDSS spectra belong to the dataset {dataset_dir} "
                         f"({sdss_specs_hash=}, {spectra_cache_file=} in Configs_used.txt)")

    # the data pipeline writes the filtered spectra into the folder of the unfiltered ones
    return os.path.join(os.path.dirname(spectra_cache_file), f"spectra_filtered_{sdss_specs_hash}.npz")


def get_sdss_spectra_for_inference(cache_path):

    with np.load(cache_path) as npz:
        data_sdss = {
            "flux": npz["flux"],
            "mask": npz["mask"],
            }

    specs = torch.Tensor(data_sdss["flux"])
    mask = torch.from_numpy(data_sdss["mask"]).to(torch.bool) # need to retain the datatype, so cant use torch.Tensor

    assert torch.isfinite(specs[mask]).all(), "X contains NaN or Inf values in the valid pixels"

    return specs, mask


def initialize_trafo_from_saved_state(checkpoint, device):
    """ Initializes Transformer from a saved state

    Args:
        checkpoint (dict): the dict that train_model.py saved for the model, loaded with torch.load
        device (torch.device): cpu or cuda

    Returns:
        model (Transformer): Transformer model object
    """
    params = checkpoint["config"]

    d_model = params['model']['d_model']
    num_heads = params['model']['num_heads']
    num_layers = params['model']['num_layers']
    d_ff = params['model']['d_ff']
    dropout = params['model']['dropout']

    model = Transformer(checkpoint["len_spectra"], checkpoint["len_output"], d_model, num_heads, num_layers, d_ff, dropout).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    return model


def eval_model(model, X, X_mask, device, batch_size=512):
    """ Does model forward pass with given data X

    Args:
        model (Transformer): Transformer model object
        X (torch.Tensor): The input data
        X_mask (torch.Tensor): boolean mask, True for valid pixels
        device (torch.device): cpu or cuda
        batch_size (int, optional): number of spectra per forward pass. The reference box alone has
                                    10000 spectra, which in one pass would not fit into GPU memory

    Returns:
        torch.Tensor: The predicted output of the model
    """
    model.eval()

    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size].to(device)
            X_mask_batch = X_mask[i:i + batch_size].to(device)

            y_pred.append(model(X_batch, X_mask_batch).cpu())

    return torch.cat(y_pred)


def main():
    # The model name is taken from the standard config. Everything else is read from the file that
    # train_model.py saved for this model, so it is what the model was really trained with
    with open("config_optimized.yaml") as f:
        model_name = yaml.safe_load(f)['model']['name']

    state_path = os.path.join(SCRIPT_DIR, "model_states", f"{model_name}_weights.pt")
    sdss_corner_path = os.path.join(SCRIPT_DIR, "plots", f"{model_name}_SDSS_estimates_cornerplot.pdf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loaded to the cpu, because y_mean and y_std are applied to the predictions there. The weights are
    # copied to the device when they are put into the model
    checkpoint = torch.load(state_path, map_location="cpu")
    params = checkpoint["config"]
    y_mean, y_std = checkpoint["y_mean"], checkpoint["y_std"]
    x_params = checkpoint["x_params"]

    # the dataset the model was trained on, needed for the SDSS spectra and the reference boxes
    dataset_dir = params['dataset_dir']

    # the SDSS spectra that gave the noise and the masks of this dataset, so they have the same pixels
    cache_path = get_sdss_cache_path(dataset_dir)
    sdss_specs, sdss_masks = get_sdss_spectra_for_inference(cache_path)

    assert sdss_specs.shape[1] == checkpoint["len_spectra"], f"SDSS spectra and the spectra the model was trained on have different lengths: {sdss_specs.shape[1]} vs {checkpoint['len_spectra']}"

    model = initialize_trafo_from_saved_state(checkpoint, device)

    # The model was trained on transformed flux, so feeding it raw SDSS or reference box spectra
    # would give silently wrong predictions
    sdss_specs = transform_flux(sdss_specs, sdss_masks, x_params)

    y_pred_sdss = eval_model(model, sdss_specs, sdss_masks, device)
    y_pred_sdss = y_pred_sdss*y_std + y_mean
    y_pred_sdss = y_pred_sdss.numpy()

    make_corner_plot(y_pred_sdss, sdss_corner_path, True)

    # predictions and true parameters of every reference box, for the comparison between the boxes below
    ref_predictions = {}

    # every reference box is evaluated on its own, with its own printout and corner plot
    for box_path, hash_name in read_hash_mapping(os.path.join(dataset_dir, "reference_boxes")):
        box_name = os.path.basename(box_path)
        ref_corner_path = os.path.join(SCRIPT_DIR, "plots", f"{model_name}_refbox_{box_name}_estimates_cornerplot.pdf")

        ref_box_specs, ref_box_masks, ref_box_y = get_spectra_reference_point(dataset_dir, hash_name)
        ref_box_specs = torch.Tensor(ref_box_specs)
        ref_box_masks = torch.from_numpy(ref_box_masks).to(torch.bool) # need to retain the datatype, so cant use torch.Tensor

        assert sdss_specs.shape[1] == ref_box_specs.shape[1], f"SDSS and reference box spectra have different lengths: {sdss_specs.shape[1]} vs {ref_box_specs.shape[1]}"

        ref_box_specs = transform_flux(ref_box_specs, ref_box_masks, x_params)

        y_pred_ref = eval_model(model, ref_box_specs, ref_box_masks, device)
        y_pred_ref = y_pred_ref*y_std + y_mean
        y_pred_ref = y_pred_ref.numpy()

        print(f"Reference box {box_name} ({len(y_pred_ref)} spectra): median [16th, 84th percentile] vs true value")
        for name, pred, true in zip(["Omega_m", "Omega_b", "Omega_L", "h"], y_pred_ref.T, ref_box_y[0]):
            lo, med, hi = np.percentile(pred, [16, 50, 84])
            print(f"    {name:8s} {med:.4f} [{lo:.4f}, {hi:.4f}]   true {true:.4f}")

        make_corner_plot(y_pred_ref, ref_corner_path, True)

        ref_predictions[box_name] = (y_pred_ref, ref_box_y[0])

    # The reference boxes have the same cosmology and differ only in their initial conditions, so the model
    # should predict the same for all of them. A difference of the medians that is not small compared to the
    # scatter between spectra means that the model uses more than the cosmology
    first_name = list(ref_predictions)[0]
    y_pred_first, y_true_first = ref_predictions[first_name]

    for box_name in list(ref_predictions)[1:]:
        y_pred_box, y_true_box = ref_predictions[box_name]

        assert np.allclose(y_true_box, y_true_first), f"{box_name} and {first_name} have different cosmologies"

        print(f"Difference of the medians, {box_name} - {first_name}, and scatter between spectra "
              f"(half of the 16th to 84th percentile range of {first_name})")
        for name, pred_first, pred_box in zip(["Omega_m", "Omega_b", "Omega_L", "h"], y_pred_first.T, y_pred_box.T):
            difference = np.median(pred_box) - np.median(pred_first)
            lo, hi = np.percentile(pred_first, [16, 84])
            scatter = (hi - lo) / 2
            print(f"    {name:8s} {difference:+.4f}   scatter {scatter:.4f}   ratio {difference/scatter:+.2f}")


if __name__ == "__main__":
    main()