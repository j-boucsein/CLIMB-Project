import numpy as np
import matplotlib.pyplot as plt
import torch
import time

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from inference import initialize_trafo_from_saved_state
from train_model import eval_model, collect_dataloaders


def get_bias_and_var(y_true, y_pred):
    params = ["Omega_m", "Omega_b", "Omega_L", "H0"]

    y_true_by_param = [y_true[:, i] for i in range(len(params))]
    y_pred_by_param = [y_pred[:, i] for i in range(len(params))]

    bias_by_par = []
    std_by_par = []
    true_y_by_par = []

    for index in range(4):

        y_true_par = y_true_by_param[index]
        y_pred_par = y_pred_by_param[index]

        y_pred_bias = []
        y_pred_std_par = []
        y_true_unique_par = []
        for true_value in set(y_true_par):
            y_pred_this_true_value = y_pred_par[y_true_par == true_value]
            bias_this_true_value = y_pred_this_true_value.mean()
            std_this_true_value = y_pred_this_true_value.std()

            y_pred_std_par.append(std_this_true_value)
            y_pred_bias.append(bias_this_true_value)
            y_true_unique_par.append(true_value)

        # sort for y_true vals
        combined = list(zip(y_pred_std_par, y_pred_bias, y_true_unique_par))
        combined_sorted = sorted(combined, key=lambda x: x[2])
        y_pred_std_par, y_pred_bias, y_true_unique_par = map(list, zip(*combined_sorted))

        bias_by_par.append(y_pred_bias)
        std_by_par.append(y_pred_std_par)
        true_y_by_par.append(y_true_unique_par)

    return bias_by_par, std_by_par, true_y_by_par



models_to_plot = ["sweep_model_2", "sweep_model_10", "sweep_model_100"]
bias_by_model = []
std_by_model = []
true_y_by_model = []

for model_name in models_to_plot:

    print(f"Modle {model_name}")

    weights_path = f"../model_states/{model_name}_weights.pt"
    config_path = f"../log_files/{model_name}_config.yaml"

    print("Collecting dataset")
    t1 = time.time()
    _, _, test_loader, y_mean, y_std = collect_dataloaders(config_path)
    t2 = time.time()
    print(f"finished. Took {t2-t1:.3f} s")
    len_in = test_loader.dataset.X.shape[1]
    len_out = test_loader.dataset.y.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initalizing model")
    model, criterion, optimizer = initialize_trafo_from_saved_state(config_path, len_in, len_out, weights_path, device)

    print("Evaluating model")
    t3 = time.time()
    _, y_test_true, y_test_pred = eval_model(model, test_loader, criterion, device, return_predictions=True)
    t4 = time.time()
    print(f"finished. Took {t4-t3:.3f} s")
    y_test_true, y_test_pred  = y_test_true*y_std + y_mean, y_test_pred*y_std + y_mean
    y_true, y_pred = y_test_true.numpy(), y_test_pred.numpy() 

    print("Calculating bias and std")
    t5 = time.time()
    bias_by_par, std_by_par, true_y_by_par = get_bias_and_var(y_true, y_pred)
    t6 = time.time()
    bias_by_model.append(bias_by_par)
    std_by_model.append(std_by_par)
    true_y_by_model.append(true_y_by_par)


fig, axs = plt.subplots(2, 2, figsize=(7, 7))

model_names_plot = ["snr2", "snr10", "snr100"]

for model_index in range(len(models_to_plot)):

    for parameter_index in range(4):

        ax = axs[parameter_index%2, parameter_index//2]

        y_bias = np.array(bias_by_model[model_index][parameter_index])
        y_std = np.array(std_by_model[model_index][parameter_index])
        y_true = np.array(true_y_by_model[model_index][parameter_index])

        bias = y_true - y_bias

        ax.plot(y_true, bias, label=f"{model_names_plot[model_index]}")
        ax.fill_between(y_true, bias + y_std, bias - y_std, alpha=0.3)

param_names = [r"$\Omega_m$", r"$\Omega_b$", r"$\Omega_\Lambda$", r"$H_0$"]

for parameter_index in range(4):
    ax = axs[parameter_index%2, parameter_index//2]

    ax.legend()
    ax.set_title(rf"{param_names[parameter_index]}")
    ax.set_ylabel("y_true - y_pred")
    ax.set_xlabel("y_true")

plt.tight_layout()
plt.savefig("plots/model_biases_const.pdf", format="PDF")