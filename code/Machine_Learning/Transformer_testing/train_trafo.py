import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import time
import yaml
import matplotlib.pyplot as plt

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.spectra_helpers import SpectraCustomHDF5
from util.sim_data_helpers import get_cosmo_parameters

from first_tranformer_test import *


def write_to_log_file(file_path, message, mode="a"):
    """Function to write content to the log file

    Args:
        file_path (string): path to the log file
        message (string): message to write to the log file
        mode (str, optional): Mode of opening the file. Defaults to "a".
    """
    with open(file_path, mode) as file:
        file.write(message)
        file.write("\n")


def build_dataset_for_gridpoints(gridpoints, suite_of_spectra, shuffle_and_split_type=None):
    """Collects the Flux data from the gridpoint spectra files and returns them as X and y data np.arrays

    Args:
        gridpoints (list): list with the grid point numbers to use
        suite_to_use (string): simulation suite to use

    Returns:
        np.array(X): Array of spectra (flux values)
        np.array(y): Array of cosmo parameters (for each spectrum)
    """
    X, y = [], []

    for i in gridpoints:
        if shuffle_and_split_type is None:
            path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/gp{i}_spectra.hdf5"
        else:
            path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/gp{i}_spectra_{shuffle_and_split_type}.hdf5"
        spec_file = SpectraCustomHDF5(path_to_file)
        _, flux = spec_file.get_all_spectra()  
        flux = flux[:1000]  # (1000, 468) TODO: delete this line for training with the full dataset

        metadata = spec_file.get_header()
        params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

        for spec in flux:
            X.append(spec)
            y.append(params)

    return np.array(X), np.array(y)


class SpectraCosmoDataset(Dataset):
    """Class that defines a custom dataset for the spectra data
    """
    def __init__(self, X, y, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def _normalize(dataset, X_mean, X_std, y_mean, y_std):
    """Normalize the given dataset using the other given parameters

    Args:
        dataset (torch.utils.data.Dataset): dataset to be normalized
        X_mean (float): mean of the training X data
        X_std (_type_): std of the training X data
        y_mean (_type_): mean of the training y data
        y_std (_type_): std of the training y data
    """
    # dataset.X = (dataset.X - X_mean) / X_std
    dataset.y = (dataset.y - y_mean) / y_std


def get_datasets(suite_of_spectra, log_file_path):
    """Split the gridpoints into training, test and eval and then collect the datasets and normalize them

    Args:
        suite_to_use (string): simulation suite to use
        log_file_path (string): path to the log file

    Returns:
        train_dataset (SpectraCosmoDataset): training dataset
        test_dataset (SpectraCosmoDataset): test dataset
        eval_dataset (SpectraCosmoDataset): evaluation dataset
    """

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Collecting datasets")

    ################# Make random list of gridpoints for train, eval and test sets #################
    index_list = np.array(list(range(50)))
    np.random.seed(123)
    np.random.shuffle(index_list)

    n_train = int(0.7 * len(index_list))
    n_eval  = int(0.15 * len(index_list))

    train_gps = index_list[:n_train]
    eval_gps  = index_list[n_train:n_train + n_eval]
    test_gps  = index_list[n_train + n_eval:]

    ################ Make the datasets #####################

    X_train, y_train = build_dataset_for_gridpoints(train_gps, suite_of_spectra)
    X_eval,  y_eval  = build_dataset_for_gridpoints(eval_gps, suite_of_spectra)
    X_test,  y_test  = build_dataset_for_gridpoints(test_gps, suite_of_spectra)

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)

    ############### Standardize the data ####################

    X_mean = train_dataset.X.mean(dim=0)
    X_std  = train_dataset.X.std(dim=0) + 1e-8

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(eval_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(test_dataset, X_mean, X_std, y_mean, y_std)

    ############## Test if there is no information leakege between datasets ##############

    train_cosmo = set(map(tuple, y_train))
    test_cosmo  = set(map(tuple, y_test))

    assert train_cosmo.isdisjoint(test_cosmo)

    write_to_log_file(log_file_path, f"Created training dataset with {train_dataset.X.size()=}, {train_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created evaluation dataset with {eval_dataset.X.size()=}, {eval_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created testing dataset with {test_dataset.X.size()=}, {test_dataset.y.size()=}")
    write_to_log_file(log_file_path, "")

    return train_dataset, test_dataset, eval_dataset, y_mean, y_std


def train_one_epoch(model, loader, optimizer, criterion, device, log_file_path):
    """train the model for one epoch

    Args:
        model (Transformer): the transformer model to train
        loader (torch.utils.data.DataLoader): the dataloader for the training data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Loss_function): the loss function
        device (torch.device): device to train on

    Returns:
        float: the avarage loss over this epoch
    """
    model.train()

    total_loss = 0.0
    for batch, (X, y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        if batch%100 == 0:
            write_to_log_file(log_file_path, f"    Loss: {loss.item():4f} [{(batch+1)*len(X)}/{len(loader.dataset)}]")

    avarage_loss = total_loss / len(loader.dataset)
    return avarage_loss


def eval_model(model, loader, criterion, device):
    """evaluate the model

    Args:
        model (Transformer): the transformer model to evaluate
        loader (torch.utils.data.DataLoader): the dataloader for the evaluation data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Loss_function): the loss function
        device (torch.device): device to do inference on

    Returns:
        float: the avarage loss over all data batches
    """
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            total_loss += loss.item() * X.size(0)
    
    avarage_loss = total_loss / len(loader.dataset)
    return avarage_loss


def write_loss_to_file(file_path, train_loss, eval_loss, mode="a"):
    """write training loss and evaluation loss to a file

    Args:
        file_path (string): path to the file
        train_loss (float): training loss
        eval_loss (float): evaluation loss
        mode (str, optional): Mode to open file in. Defaults to "a".
    """
    with open(file_path, mode) as file:
        writer = csv.writer(file, delimiter=',',)
        writer.writerow([train_loss, eval_loss])


def get_datasets_shuffle_over_gps(suite_of_spectra, log_file_path):
    """ This function is for debugging purposes only
    """
    gps_list = [i for i in range(50)]

    X_all, y_all = build_dataset_for_gridpoints(gps_list, suite_of_spectra)

    print(X_all.shape[0])

    index_list = np.array(list(range(X_all.shape[0])))
    np.random.seed(42)
    np.random.shuffle(index_list)

    X_all = X_all[index_list]
    y_all = y_all[index_list]

    n_train = int(0.7 * X_all.shape[0])
    n_eval  = int(0.15 * X_all.shape[0])

    print(n_train, n_eval)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_eval,  y_eval = X_all[n_train:n_eval+n_train], y_all[n_train:n_eval+n_train]
    X_test,  y_test = X_all[n_eval+n_train:], y_all[n_eval+n_train:]

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)


    X_mean = train_dataset.X.mean(dim=0)
    X_std  = train_dataset.X.std(dim=0) + 1e-8

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(eval_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(test_dataset, X_mean, X_std, y_mean, y_std)

    return train_dataset, eval_dataset, test_dataset, y_mean, y_std


def plot_loss_curve(train_loss_path, model_name):
    
    with open(train_loss_path, "r") as file:
        reader = csv.reader(file)
        loss_plot = [i for i in reader][1:]

    train_loss, eval_loss = [list(map(float, col)) for col in zip(*loss_plot)]

    plt.plot([i for i in range(len(train_loss))], train_loss, label="train loss")
    plt.plot([i for i in range(len(eval_loss))], eval_loss, label="validation loss")

    plt.ylabel("MSE Loss")
    plt.xlabel("Number of epochs")
    plt.legend()
    plt.title(f"Transformer '{model_name}' - Loss curve")

    plt.savefig(f"plots/{model_name}_losscurve.pdf", format="PDF")


def eval_model_plots(model, loader, criterion, device):
    all_preds = []
    all_targets = []
    model.eval()

    total_loss = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            total_loss.append(loss.item())

            all_preds.append(y_pred)
            all_targets.append(y)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
    
    avarage_loss = np.array(total_loss).mean()
    return avarage_loss, all_targets, all_preds


def inference_plot(y_true, y_pred, model_name):
    params = ["Omega_m", "Omega_b", "Omega_L", "H0"]

    y_true_by_param = [y_true[:, i] for i in range(len(params))]
    y_pred_by_param = [y_pred[:, i] for i in range(len(params))]

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for index in range(4):

        y_true_par = y_true_by_param[index]
        y_pred_par = y_pred_by_param[index]

        y_pred_mean_par = []
        y_pred_std_par = []
        y_true_unique_par = []
        y_pred_per_unique_value = []
        for true_value in set(y_true_par):
            y_pred_this_true_value = y_pred_par[y_true_par == true_value]
            mean_this_true_value = y_pred_this_true_value.mean()
            std_this_true_value = y_pred_this_true_value.std()

            y_pred_std_par.append(std_this_true_value)
            y_pred_mean_par.append(mean_this_true_value)
            y_true_unique_par.append(true_value)

            y_pred_per_unique_value.append(y_pred_this_true_value)

        ax = axs[index%2, index//2]

        ax.plot(sorted(y_true_par), sorted(y_true_par), linestyle="--", c="red", alpha=0.7, zorder=2)

        ax.scatter(y_true_par, y_pred_par, alpha=0.5, c="lightgrey", zorder=1, label="inference points", rasterized=True)
        ax.errorbar(y_true_unique_par, y_pred_mean_par, yerr=y_pred_std_par, linestyle="None", marker="x", color="black", zorder=3, label=r"mean with 1 $\sigma$ std")
        #plt.scatter(y_true_unique_Om, [y_pred_mean_Om[i]+y_pred_std_Om[i] for i in range(len(y_true_unique_Om))], linestyle="None", marker="o", color="red")
        #plt.scatter(y_true_unique_Om, [y_pred_mean_Om[i]-y_pred_std_Om[i] for i in range(len(y_true_unique_Om))], linestyle="None", marker="o", color="red")
        
        ax.set_title(f"{params[index]}")
        ax.legend(prop={'size': 8})
        #ax.set_xticks(sorted(y_true_unique_par))
        #ax.set_xticklabels([f"{x:.2f}" for x in y_true_unique_par])

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_inference.pdf", format="PDF")


def save_test_inference(y_true, y_pred, model_name):
    
    file_path = f"test_inference_files/{model_name}_testset_inference.csv"
    with open(file_path, "w") as file:
        writer = csv.writer(file, delimiter=',',)
        for i in range(len(y_true)):
            writer.writerow([y_true[i], y_pred[i]])


def get_mock_datasets(suite_of_spectra, log_file_path):

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Collecting datasets using mock function")

    ################# Make random list of gridpoints for train, eval and test sets #################
    index_list = np.array(list(range(100)))
    np.random.seed(123)
    np.random.shuffle(index_list)

    n_train = int(0.5 * len(index_list))

    train_gps = index_list[:n_train]
    eval_gps  = index_list[n_train:]
    test_gps  = index_list[n_train:]

    ################ Make the datasets #####################

    X_train, y_train = build_dataset_for_gridpoints(train_gps, suite_of_spectra)
    X_eval,  y_eval  = build_dataset_for_gridpoints(eval_gps, suite_of_spectra)
    X_test,  y_test  = build_dataset_for_gridpoints(test_gps, suite_of_spectra)

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)

    ############### Standardize the data ####################

    X_mean = train_dataset.X.mean(dim=0)
    X_std  = train_dataset.X.std(dim=0) + 1e-8

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(eval_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(test_dataset, X_mean, X_std, y_mean, y_std)

    ############## Test if there is no information leakege between datasets ##############

    write_to_log_file(log_file_path, f"Created training dataset with {train_dataset.X.size()=}, {train_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created evaluation dataset with {eval_dataset.X.size()=}, {eval_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created testing dataset with {test_dataset.X.size()=}, {test_dataset.y.size()=}")
    write_to_log_file(log_file_path, "")

    return train_dataset, test_dataset, eval_dataset, y_mean, y_std


def get_shuffled_and_split_datasets(suite_of_spectra, log_file_path):

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Collecting datasets using shuffled_and_split function")

    gps_list = [i for i in range(50)]

    X_train, y_train = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "train")
    X_eval, y_eval = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "eval")
    X_test, y_test = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "test")

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)

    X_mean = train_dataset.X.mean(dim=0)
    X_std  = train_dataset.X.std(dim=0) + 1e-8

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(eval_dataset, X_mean, X_std, y_mean, y_std)
    _normalize(test_dataset, X_mean, X_std, y_mean, y_std)

    write_to_log_file(log_file_path, f"Created training dataset with {train_dataset.X.size()=}, {train_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created evaluation dataset with {eval_dataset.X.size()=}, {eval_dataset.y.size()=}")
    write_to_log_file(log_file_path, f"Created testing dataset with {test_dataset.X.size()=}, {test_dataset.y.size()=}")
    write_to_log_file(log_file_path, "")

    return train_dataset, eval_dataset, test_dataset, y_mean, y_std



def main():
    # read out the config file
    with open("config.yaml") as f:
        params = yaml.safe_load(f)

    dataset_function_name = params['dataset_collection_fucntion']
    suite_of_spectra = params['suite_of_spectra']
    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    learning_rate = params['learning_rate']

    d_model = params['model']['d_model']
    num_heads = params['model']['num_heads']
    num_layers = params['model']['num_layers']
    d_ff = params['model']['d_ff']
    dropout = params['model']['dropout']

    model_name = params['model']['name']

    # set paths
    path_to_model_states = f"model_states/train_{model_name}/"
    state_name = "finished_model_weights.pt"
    checkpoint_name = "checkpoint.pt"
    train_eval_loss_path = f"loss_files/{model_name}_train_eval_loss.csv"
    log_file_path = f"log_files/{model_name}_log.txt"
    yaml_config_save_path = f"log_files/{model_name}_config.yaml"

    dataset_function = None
    # set which dataset collection function to use
    if dataset_function_name == "shuffled":
        dataset_function = get_datasets_shuffle_over_gps
    elif dataset_function_name == "split":
        dataset_function = get_datasets
    elif dataset_function_name == "mock":
        dataset_function = get_mock_datasets
    elif dataset_function_name == "shuffled_and_split":
        dataset_function = get_shuffled_and_split_datasets

    assert dataset_function is not None, "invalid dataset function name was given in config file"

    # save the config file as part of the logging
    with open(yaml_config_save_path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    write_to_log_file(log_file_path, "Starting script...", mode="w")
    write_loss_to_file(train_eval_loss_path, [], [], mode="w")

    # check if model_states path exists and create if not
    os.makedirs(path_to_model_states, exist_ok=True)

    # try to use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_log_file(log_file_path, f"Using device: {device}")

    # get the datasets
    train_dataset, eval_dataset, test_dataset, y_mean, y_std = dataset_function(suite_of_spectra, log_file_path)

    # make the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # get input and output size
    len_spectra = train_dataset.X.shape[1]
    len_output = train_dataset.y.shape[1]

    write_to_log_file(log_file_path, "Initialize transformer model with parameters:")
    write_to_log_file(log_file_path, f"{len_spectra=} \n{len_output=} \n{d_model=} \n{num_heads=}" \
                       f"\n{num_layers=} \n{d_ff=} \n{dropout=} \n{learning_rate=} \n{batch_size=}" \
                        f"\n{n_epochs=}")

    # initialize model
    model = Transformer(len_spectra, len_output, d_model, num_heads, num_layers, d_ff, dropout).to(device)

    # initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Beginning training ...")

    # save model performence before first training step
    eval_loss = eval_model(model, eval_loader, criterion, device)
    train_loss = eval_model(model, train_loader, criterion, device)
    write_loss_to_file(train_eval_loss_path, train_loss, eval_loss)

    # train model for the epochs
    for epoch in range(n_epochs):
        start_train = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, log_file_path)
        end_train = time.time()
        eval_loss = eval_model(model, eval_loader, criterion, device)
        end_eval = time.time()

        write_to_log_file(log_file_path, f"Training epoch {epoch+1} done. Training took {end_train - start_train:.0f}s" \
                          f", Evaluating took {end_eval - end_train:.0f}s")
        write_to_log_file(log_file_path, f"         Current Train loss: {train_loss}")
        write_to_log_file(log_file_path, f"         Current Eval  loss: {eval_loss}")
        write_to_log_file(log_file_path, "")

        write_loss_to_file(train_eval_loss_path, train_loss, eval_loss)

        # save a training checkpoint
        if (epoch+1)%10 == 0:
            training_checkpoint_path = path_to_model_states + f"epoch{epoch+1}_" + checkpoint_name
            write_to_log_file(log_file_path, f"Saving state of model as training checkpoint to {training_checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, training_checkpoint_path)

    # save the final model state
    write_to_log_file(log_file_path, f"Finished training, saving model state to {path_to_model_states+state_name}")
    torch.save(model.state_dict(), path_to_model_states+state_name)
    write_to_log_file(log_file_path, "Done!")

    write_to_log_file(log_file_path, "\n\nMaking Plots...")
    plot_loss_curve(train_eval_loss_path, model_name)
    write_to_log_file(log_file_path, "Saved loss curve plot")

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Starting evaluation on test set")
    test_loss, y_test_true, y_test_pred = eval_model_plots(model, test_loader, criterion, device)

    y_test_true, y_test_pred  = y_test_true.to(torch.device("cpu"))*y_std + y_mean, y_test_pred.to(torch.device("cpu"))*y_std + y_mean
    y_true, y_pred = y_test_true.numpy(), y_test_pred.numpy()

    save_test_inference(y_true, y_pred, model_name)

    inference_plot(y_true, y_pred, model_name)
    write_to_log_file(log_file_path, "Done with inference on test set")
    write_to_log_file(log_file_path, f"Final Test loss: {test_loss:.2e}")
    write_to_log_file(log_file_path, "Saved inference plot")

    write_to_log_file(log_file_path, "-------------- DONE --------------")


if __name__ == "__main__":
    main()