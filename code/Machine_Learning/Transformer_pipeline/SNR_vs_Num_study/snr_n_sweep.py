import numpy as np
import torch
import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from dataset_functions import get_spectra_reference_point, get_shuffled_and_split_datasets
from inference import initialize_trafo_from_saved_state, eval_model
from plotting_functions import make_corner_plot


def main():
    snr_grid_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 100]
    n_spectra_grid_vals = [500] + [i*1000 for i in range(1, 11)]

    results = np.full(
        (len(n_spectra_grid_vals),
        len(snr_grid_vals),
        3,          # 0: y_pred, 1: y_pred_std, 2: y_refbox
        4),         # 0: Omega_m, 1: Omega_b, 2: Omega_Lambda, 3: H0
        np.nan
    )

    for j, snr in enumerate(snr_grid_vals):

        print(f"Loading Data and Model for SNR point {snr}")
        
        model_name = f"sweep_model_{snr}"
        suite_of_spectra = f"L25n256_snr_sweep_{snr}"

        config_path = f"../log_files/{model_name}_config.yaml"
        state_path = f"../model_states/{model_name}_weights.pt"

        # TODO: This is an embarrassingly inefficient way to get the y_mean and y_std. Should probably change this in the future
        _, _, _, y_mean, y_std = get_shuffled_and_split_datasets(suite_of_spectra, True)

        # load all ref box spectra once per suite
        ref_box_specs, y_ref_box = get_spectra_reference_point(suite_of_spectra)
        y_ref_box = y_ref_box[0, :]     # this has shape (n_spec, 4) but all values of axis 0 are the same

        input_len = ref_box_specs.shape[1]
        output_len = 4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the Transformer model once per suite
        model, _, _ = initialize_trafo_from_saved_state(config_path, input_len, output_len, state_path, device)

        # Do inference for the different number of spectra
        for i, n_spec in enumerate(n_spectra_grid_vals):

            print(f"    start inference for n_spec {n_spec}")

            corner_save_path = f"corner_plots/{model_name}_n{n_spec}.pdf"

            # randomly select n_spec number of spectra
            np.random.shuffle(ref_box_specs)
            ref_box_specs_tensor = torch.Tensor(ref_box_specs[:n_spec])

            y_pred_ref = eval_model(model, ref_box_specs_tensor, device)

            y_pred_ref = y_pred_ref*y_std + y_mean
            y_pred_ref = y_pred_ref.numpy()

            y_pred_vals = y_pred_ref.mean(axis=0)
            y_pred_std = y_pred_ref.std(axis=0)

            results[i, j, 0, :] = y_pred_vals
            results[i, j, 1, :] = y_pred_std
            results[i, j, 2, :] = y_ref_box

            make_corner_plot(y_pred_ref, corner_save_path, save_plot=True, show_plot=False)

    np.savez(
        "sweep_results.npz",
        results=results,
        snr=np.array(snr_grid_vals),
        n_spectra=np.array(n_spectra_grid_vals)
    )


if __name__ == "__main__":
    main()