import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import time

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.spectra_helpers import SpectraCustomHDF5
from util.sim_data_helpers import get_cosmo_parameters

from first_tranformer_test import *


def write_to_log_file(file_path, message, mode="a"):
    with open(file_path, mode) as file:
        file.write(message)
        file.write("\n")


def build_dataset_for_gridpoints(gridpoints, suite_to_use):
    X, y = [], []

    for i in gridpoints:
        path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_to_use}/gp{i}_spectra.hdf5"
        spec_file = SpectraCustomHDF5(path_to_file)
        _, flux = spec_file.get_all_spectra()  
        flux = flux[:1000]  # (1000, 468) TODO: delete this line for training with the full dataset

        params = get_cosmo_parameters(
            f"/vera/ptmp/gc/jerbo/{suite_to_use}/gridpoint{i}/"
        )

        for spec in flux:
            X.append(spec)
            y.append(params)

    return np.array(X), np.array(y)


class SpectraCosmoDataset(Dataset):
    def __init__(self, X, y, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def _normalize(dataset, X_mean, X_std, y_mean, y_std):
        dataset.X = (dataset.X - X_mean) / X_std
        dataset.y = (dataset.y - y_mean) / y_std


def get_datasets(suite_to_use, log_file_path):
    
    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Collecting datasets")

    ################# Make random list of gridpoints for train, eval and test sets #################
    index_list = np.array(list(range(50)))
    np.random.seed(42)
    np.random.shuffle(index_list)

    n_train = int(0.7 * len(index_list))
    n_eval  = int(0.15 * len(index_list))

    train_gps = index_list[:n_train]
    eval_gps  = index_list[n_train:n_train + n_eval]
    test_gps  = index_list[n_train + n_eval:]

    ################ Make the datasets #####################

    X_train, y_train = build_dataset_for_gridpoints(train_gps, suite_to_use)
    X_eval,  y_eval  = build_dataset_for_gridpoints(eval_gps,  suite_to_use)
    X_test,  y_test  = build_dataset_for_gridpoints(test_gps,  suite_to_use)

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

    return train_dataset, test_dataset, eval_dataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    avarage_loss = total_loss / len(loader.dataset)
    return avarage_loss


def eval_model(model, loader, criterion, device):
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
    with open(file_path, mode) as file:
        writer = csv.writer(file, delimiter=',',)
        writer.writerow([train_loss, eval_loss])


def main():

    suite_to_use = "L25n256_suite"
    batch_size = 256
    n_epochs = 5
    state_save_path = "transformer_weights.pt"
    train_eval_loss_path = "train_eval_loss.csv"
    log_file_path = "transformer_log.txt"

    write_to_log_file(log_file_path, "Starting script...", mode="w")
    write_loss_to_file(train_eval_loss_path, [], [], mode="w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_log_file(log_file_path, f"Using device: {device}")

    train_dataset, eval_dataset, test_dataset = get_datasets(suite_to_use, log_file_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    len_spectra = train_dataset.X.shape[1]
    len_output = train_dataset.y.shape[1]
    d_model = 128
    num_heads = 4
    num_layers = 6
    d_ff = 128*2
    dropout = 0.1
    learning_rate = 0.0001

    write_to_log_file(log_file_path, "Initialize transformer model with parameters:")
    write_to_log_file(log_file_path, f"{len_spectra=} \n{len_output=} \n{d_model=} \n{num_heads=}" \
                       f"\n{num_layers=} \n{d_ff=} \n{dropout=} \n{learning_rate=}")

    model = Transformer(len_spectra, len_output, d_model, num_heads, num_layers, d_ff, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Beginning training ...")

    for epoch in range(n_epochs):
        start_train = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        end_train = time.time()
        eval_loss = eval_model(model, eval_loader, criterion, device)
        end_eval = time.time()

        write_to_log_file(log_file_path, f"Training epoch {epoch} done. Training took {end_train - start_train:.0f}s" \
                          f", Evaluating took {end_eval - end_train:.0f}s")
        write_to_log_file(log_file_path, f"         Current Train loss: {train_loss}")
        write_to_log_file(log_file_path, f"         Current Eval  loss: {eval_loss}")
        
        write_loss_to_file(train_eval_loss_path, train_loss, eval_loss)

    write_to_log_file(log_file_path, f"Finished training, saving model state to {state_save_path}")
    torch.save(model.state_dict(), state_save_path)


main()