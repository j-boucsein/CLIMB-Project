import yaml
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset_functions import get_shuffled_and_split_datasets
from transformer_model import Transformer
from plotting_functions import plot_test_inference_errorbars, plot_test_inference_colors, plot_all_PDFs, plot_curvature_param


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


def train_one_epoch(model, loader, optimizer, criterion, device):
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


def eval_model(model, loader, criterion, device, return_predictions=False):
    """evaluate the model

    Args:
        model (Transformer): the transformer model to evaluate
        loader (torch.utils.data.DataLoader): the dataloader for the evaluation data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Loss_function): the loss function
        device (torch.device): device to do inference on
        return_predictions (bool, optional): if set to True also return y_true and y_pred

    Returns:
        float: the avarage loss over all data batches
    """
    all_preds = []
    all_targets = []
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            total_loss += loss.item() * X.size(0)

            if return_predictions:
                all_preds.append(y_pred)
                all_targets.append(y)

    avarage_loss = total_loss / len(loader.dataset)

    if return_predictions:
        all_preds = torch.cat(all_preds, dim=0).to(torch.device("cpu"))
        all_targets = torch.cat(all_targets, dim=0).to(torch.device("cpu"))
        return avarage_loss, all_targets, all_preds
    
    else:
        return avarage_loss
    

def plot_loss_curve(train_loss, eval_loss, save_path):
    """ Plot the loss curve for the training loss and evaluation loss

    Args:
        train_loss (list): list containing training loss
        eval_loss (list): list containing eval loss
        save_path (string): path of where to save the plot to
    """

    plt.plot([i for i in range(len(train_loss))], train_loss, label="train loss")
    plt.plot([i for i in range(len(eval_loss))], eval_loss, label="validation loss")

    plt.ylabel("MSE Loss")
    plt.xlabel("Number of epochs")
    plt.legend()

    plt.savefig(save_path, format="PDF")


def collect_dataloaders(config_path):
    """ Collects the datasets and creates the DataLoader objects for the given config file

    Args:
        config_path (string): path to the config file

    Returns:
        train_loader (torch.utils.data.DataLoader): Dataloader of the training data
        eval_loader (torch.utils.data.DataLoader): Dataloader of the evaluation data
        test_loader (torch.utils.data.DataLoader): Dataloader of the test data
        y_mean (torch.Tensor): mean of each cosmo param in training dataset
        y_std (torch.Tensor): std of each cosmo param in training dataset
    """

    with open(config_path) as f:
        params = yaml.safe_load(f)

    suite_of_spectra = params['suite_of_spectra']
    batch_size = params['batch_size']
    reduce_dataset = params['reduced_dataset']

    train_dataset, eval_dataset, test_dataset, y_mean, y_std = get_shuffled_and_split_datasets(suite_of_spectra, reduce_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, eval_loader, test_loader, y_mean, y_std


def initiate_transformer(config_path, len_in, len_out):
    """ Make the transformer object and loss function and optimizer for a given config file

    Args:
        config_path (string): path to the config file
        len_in (int): input dimension of the transformer
        len_out (int): output dimension of the transformer

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

    model = Transformer(len_in, len_out, d_model, num_heads, num_layers, d_ff, dropout)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer


def main(config_path):

    with open(config_path) as f:
        params = yaml.safe_load(f)
    
    model_name = params['model']['name']
    n_epochs = params['n_epochs']

    yaml_config_save_path = f"log_files/{model_name}_config.yaml"
    path_to_model_states = f"model_states/{model_name}_weights.pt"
    log_file_path = f"log_files/{model_name}_log.txt"

    write_to_log_file(log_file_path, "Starting script...", mode="w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_log_file(log_file_path, f"Using device: {device}")

    train_loader, eval_loader, test_loader, y_mean, y_std = collect_dataloaders(config_path)

    len_spectra = train_loader.dataset[0][0].shape[0]
    len_output = train_loader.dataset[0][1].shape[0]

    model, criterion, optimizer = initiate_transformer(config_path, len_spectra, len_output)
    model.to(device)

    write_to_log_file(log_file_path, "Initialize transformer model:")
    write_to_log_file(log_file_path, f"{model}")

    # save the config file as part of the logging
    with open(yaml_config_save_path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Beginning training ...")

    loss_plot_train = []
    loss_plot_eval = []

    loss_plot_eval.append(eval_model(model, eval_loader, criterion, device))
    loss_plot_train.append(eval_model(model, train_loader, criterion, device))

    # train model for the epochs
    for epoch in range(n_epochs):
        t1 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        t2 = time.time()
        eval_loss = eval_model(model, eval_loader, criterion, device)
        t3 = time.time()

        write_to_log_file(log_file_path, f"Training epoch {epoch+1} done. Training took {t2 - t1:.0f}s" \
                          f", Evaluating took {t3 - t2:.0f}s")
        write_to_log_file(log_file_path, f"         Current Train loss: {train_loss}")
        write_to_log_file(log_file_path, f"         Current Eval  loss: {eval_loss}")
        write_to_log_file(log_file_path, "")

        loss_plot_train.append(train_loss)
        loss_plot_eval.append(eval_loss)

    # save model weights
    write_to_log_file(log_file_path, f"Finished training, saving model state to {path_to_model_states}")
    torch.save(model.state_dict(), path_to_model_states)

    write_to_log_file(log_file_path, "Starting Evaluation on testset...")
    _, y_test_true, y_test_pred = eval_model(model, test_loader, criterion, device, return_predictions=True)

    y_test_true, y_test_pred  = y_test_true*y_std + y_mean, y_test_pred*y_std + y_mean
    y_true, y_pred = y_test_true.numpy(), y_test_pred.numpy()  

    write_to_log_file(log_file_path, "\n\nMaking Plots...")

    plot_loss_curve(loss_plot_train, loss_plot_eval, f"plots/{model_name}_losscurve.pdf")
    plot_test_inference_errorbars(y_true, y_pred,f"plots/{model_name}_errorbars.pdf", save_plot=True)
    plot_test_inference_colors(y_true, y_pred, f"plots/{model_name}_colors.pdf", save_plot=True)
    plot_all_PDFs(y_true, y_pred, f"plots/{model_name}_PDF.pdf", save_plot=True)
    plot_curvature_param(y_true, y_pred, f"plots/{model_name}_curvature.pdf", save_plot=True)

    write_to_log_file(log_file_path, "-------------- DONE --------------")


if __name__ == "__main__":
    config_path = "config_optimized.yaml"
    main(config_path)