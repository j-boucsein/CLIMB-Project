import yaml
import os
import math
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset_functions import get_shuffled_and_split_datasets
from transformer_model import Transformer
from plotting_functions import plot_test_inference_errorbars, plot_test_inference_colors, plot_all_PDFs, plot_curvature_param

# folder of this script, so that the outputs are written to the same place from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def make_scheduler(optimizer, total_steps, warmup_frac, min_lr_factor=0.05):
    """Linear warmup followed by cosine decay

    The encoder is post-LN (``norm(x + sublayer(x))``), and post-LN transformers train badly from a
    cold start: the first few large updates push the residual stream somewhere the later layers
    cannot recover from. A short warmup is the standard remedy. warmup_frac = 0 disables it and
    leaves a plain cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to schedule
        total_steps (int): total number of optimiser steps over the whole run
        warmup_frac (float): fraction of the steps spent warming up
        min_lr_factor (float, optional): floor of the decay, as a fraction of the peak lr

    Returns:
        torch.optim.lr_scheduler.LambdaLR: the scheduler
    """
    warmup_steps = max(1, int(round(warmup_frac * total_steps))) if warmup_frac > 0 else 0

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, grad_clip=0.0):
    """train the model for one epoch

    Args:
        model (Transformer): the transformer model to train
        loader (torch.utils.data.DataLoader): the dataloader for the training data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Loss_function): the loss function
        device (torch.device): device to train on
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): stepped once per batch, not
                                                                   once per epoch, because the
                                                                   warmup is measured in updates
        grad_clip (float, optional): clip the gradient norm to this value, 0 disables clipping

    Returns:
        float: the avarage loss over this epoch
    """
    model.train()

    total_loss = 0.0
    for X, X_mask, y in loader:
        X = X.to(device, non_blocking=True)
        X_mask = X_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        y_pred = model(X, X_mask)
        loss = criterion(y_pred, y)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * X.size(0)

    avarage_loss = total_loss / len(loader.dataset)
    return avarage_loss


def eval_model(model, loader, criterion, device, return_predictions=False):
    """evaluate the model

    Args:
        model (Transformer): the transformer model to evaluate
        loader (torch.utils.data.DataLoader): the dataloader for the evaluation data
        criterion (torch.nn.Loss_function): the loss function
        device (torch.device): device to do inference on
        return_predictions (bool, optional): if set to True also return y_true and y_pred

    Returns:
        float: the avarage loss over all data batches
        torch.Tensor: the avarage loss of every parameter on its own. The targets are standardised with the
                      training set, so 1 means "as good as always predicting the training mean"
    """
    all_preds = []
    all_targets = []
    model.eval()

    total_loss = 0.0
    total_loss_per_param = 0.0
    with torch.no_grad():
        for X, X_mask, y in loader:
            X = X.to(device, non_blocking=True)
            X_mask = X_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(X, X_mask)
            loss = criterion(y_pred, y)

            total_loss += loss.item() * X.size(0)
            # squared errors summed over the spectra of the batch, separately for every parameter
            total_loss_per_param += ((y_pred - y) ** 2).sum(dim=0).cpu()

            if return_predictions:
                all_preds.append(y_pred)
                all_targets.append(y)

    avarage_loss = total_loss / len(loader.dataset)
    avarage_loss_per_param = total_loss_per_param / len(loader.dataset)

    if return_predictions:
        all_preds = torch.cat(all_preds, dim=0).to(torch.device("cpu"))
        all_targets = torch.cat(all_targets, dim=0).to(torch.device("cpu"))
        return avarage_loss, avarage_loss_per_param, all_targets, all_preds

    else:
        return avarage_loss, avarage_loss_per_param
    

def plot_loss_curve(train_loss, eval_loss, save_path):
    """ Plot the loss curve for the training loss and evaluation loss

    Args:
        train_loss (list): list containing training loss
        eval_loss (list): list containing eval loss
        save_path (string): path of where to save the plot to
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    plt.plot([i for i in range(len(train_loss))], train_loss, label="train loss")
    plt.plot([i for i in range(len(eval_loss))], eval_loss, label="validation loss")

    plt.ylabel("MSE Loss")
    plt.xlabel("Number of epochs")
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path, format="PDF")
    plt.close(fig)


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
        x_params (dict): the flux transform fitted on the training dataset (output of fit_x_transform)
    """

    with open(config_path) as f:
        params = yaml.safe_load(f)

    dataset_dir = params['dataset_dir']
    batch_size = params['batch_size']
    reduce_dataset = params['reduced_dataset']

    # Optional block, so an older config without it keeps working unchanged
    training = params.get('training', {})
    x_transform = training.get('x_transform', 'none')

    train_dataset, eval_dataset, test_dataset, y_mean, y_std, x_params = get_shuffled_and_split_datasets(
        dataset_dir, reduce_dataset, x_transform)

    # Evaluation needs no gradients, so much larger batches than in the training fit into memory. This
    # makes it faster and does not change the result
    eval_batch_size = 512

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)

    return train_loader, eval_loader, test_loader, y_mean, y_std, x_params


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
       grad_clip (float): gradient norm clipping value, 0 means no clipping
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']

    training = params.get('training', {})
    grad_clip = float(training.get('grad_clip', 0.0))

    d_model = params['model']['d_model']
    num_heads = params['model']['num_heads']
    num_layers = params['model']['num_layers']
    d_ff = params['model']['d_ff']
    dropout = params['model']['dropout']

    model = Transformer(len_in, len_out, d_model, num_heads, num_layers, d_ff, dropout)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer, grad_clip


def main(config_path):

    with open(config_path) as f:
        params = yaml.safe_load(f)
    
    model_name = params['model']['name']
    n_epochs = params['n_epochs']
    seed = params['seed']
    final_test = params['final_test']

    training = params.get('training', {})
    warmup_frac = float(training.get('warmup_frac', 0.0))
    x_transform = training.get('x_transform', 'none')

    log_dir = os.path.join(SCRIPT_DIR, "log_files")
    model_dir = os.path.join(SCRIPT_DIR, "model_states")
    plot_dir = os.path.join(SCRIPT_DIR, "plots")
    for folder in (log_dir, model_dir, plot_dir):
        os.makedirs(folder, exist_ok=True)

    yaml_config_save_path = os.path.join(log_dir, f"{model_name}_config.yaml")
    path_to_model_states = os.path.join(model_dir, f"{model_name}_weights.pt")
    log_file_path = os.path.join(log_dir, f"{model_name}_log.txt")

    write_to_log_file(log_file_path, "Starting script...", mode="w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_log_file(log_file_path, f"Using device: {device}")

    # fixes the initial weights of the model, the order of the training batches and the dropout
    torch.manual_seed(seed)
    write_to_log_file(log_file_path, f"Random seed: {seed}")

    train_loader, eval_loader, test_loader, y_mean, y_std, x_params = collect_dataloaders(config_path)

    len_spectra = train_loader.dataset[0][0].shape[0]
    len_output = train_loader.dataset[0][2].shape[0]

    model, criterion, optimizer, grad_clip = initiate_transformer(config_path, len_spectra, len_output)
    model.to(device)

    # The warmup is measured in optimiser steps, so the schedule has to know how many there are in
    # the whole run, not just in one epoch
    total_steps = len(train_loader) * n_epochs
    scheduler = make_scheduler(optimizer, total_steps, warmup_frac)

    write_to_log_file(log_file_path, "Initialize transformer model:")
    write_to_log_file(log_file_path, f"{model}")
    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, f"Training recipe: x_transform={x_transform}, "
                                     f"warmup_frac={warmup_frac} "
                                     f"({max(1, int(round(warmup_frac * total_steps))) if warmup_frac > 0 else 0}"
                                     f" of {total_steps} steps), grad_clip={grad_clip}")

    # save the config file as part of the logging
    with open(yaml_config_save_path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    write_to_log_file(log_file_path, "")
    write_to_log_file(log_file_path, "Beginning training ...")

    loss_plot_train = []
    loss_plot_eval = []

    # [0]: only the loss over all parameters goes into the loss curve
    loss_plot_eval.append(eval_model(model, eval_loader, criterion, device)[0])
    loss_plot_train.append(eval_model(model, train_loader, criterion, device)[0])

    # The model state is saved every time the eval loss improves. The file therefore always holds the best
    # epoch so far, and a job that hits the time limit still leaves a usable model
    best_eval_loss = float("inf")
    best_epoch = None

    # train model for the epochs
    for epoch in range(n_epochs):
        t1 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     scheduler=scheduler, grad_clip=grad_clip)
        t2 = time.time()
        eval_loss, eval_loss_per_param = eval_model(model, eval_loader, criterion, device)
        t3 = time.time()

        write_to_log_file(log_file_path, f"Training epoch {epoch+1} done. Training took {t2 - t1:.0f}s" \
                          f", Evaluating took {t3 - t2:.0f}s")
        write_to_log_file(log_file_path, f"         Current Train loss: {train_loss}")
        write_to_log_file(log_file_path, f"         Current Eval  loss: {eval_loss}")
        write_to_log_file(log_file_path, f"         Eval loss per parameter (Omega_m, Omega_b, Omega_L, h): "
                                         + " ".join(f"{value:.4f}" for value in eval_loss_per_param.tolist()))
        write_to_log_file(log_file_path, f"         Current lr        : {scheduler.get_last_lr()[0]:.3e}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch + 1

            # save model weights, together with everything inference.py needs to use them: the normalisation of
            # y and of the flux (both fitted on the training set), the config and the input and output length
            checkpoint = {
                "state_dict": model.state_dict(),
                "y_mean": y_mean,
                "y_std": y_std,
                "x_params": x_params,
                "config": params,
                "len_spectra": len_spectra,
                "len_output": len_output,
                "epoch": best_epoch,
                "eval_loss": eval_loss,
            }
            # written to a temporary file and then renamed, so a job that is killed while writing cannot
            # leave a broken file behind
            torch.save(checkpoint, path_to_model_states + ".tmp")
            os.replace(path_to_model_states + ".tmp", path_to_model_states)
            write_to_log_file(log_file_path, "         Best eval loss so far, model state saved")

        write_to_log_file(log_file_path, "")

        loss_plot_train.append(train_loss)
        loss_plot_eval.append(eval_loss)

    write_to_log_file(log_file_path, f"Finished training. Best eval loss: {best_eval_loss} in epoch {best_epoch}, "
                                     f"that model state is saved in {path_to_model_states}")

    # The plots below are made with the saved model state (the best epoch), which is the one inference.py
    # uses. Without the assert, a file left over from an earlier run with the same model name would be loaded
    assert best_epoch is not None, "The eval loss was never finite, so no model state was saved"
    checkpoint = torch.load(path_to_model_states, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    # While models are still compared, the plots are made from the evaluation set. The test set is only used
    # for the final model (final_test: true in the config), so it takes no part in choosing the model
    if final_test:
        plot_loader, plot_set = test_loader, "test"
    else:
        plot_loader, plot_set = eval_loader, "eval"

    write_to_log_file(log_file_path, f"Starting Evaluation on {plot_set} set...")
    _, _, y_plot_true, y_plot_pred = eval_model(model, plot_loader, criterion, device, return_predictions=True)

    y_plot_true, y_plot_pred  = y_plot_true*y_std + y_mean, y_plot_pred*y_std + y_mean
    y_true, y_pred = y_plot_true.numpy(), y_plot_pred.numpy()

    write_to_log_file(log_file_path, "\n\nMaking Plots...")

    plot_loss_curve(loss_plot_train, loss_plot_eval, os.path.join(plot_dir, f"{model_name}_losscurve.pdf"))
    plot_test_inference_errorbars(y_true, y_pred, os.path.join(plot_dir, f"{model_name}_{plot_set}_errorbars.pdf"), save_plot=True)
    plot_test_inference_colors(y_true, y_pred, os.path.join(plot_dir, f"{model_name}_{plot_set}_colors.pdf"), save_plot=True)
    plot_all_PDFs(y_true, y_pred, os.path.join(plot_dir, f"{model_name}_{plot_set}_PDF.pdf"), save_plot=True)
    plot_curvature_param(y_true, y_pred, os.path.join(plot_dir, f"{model_name}_{plot_set}_curvature.pdf"), save_plot=True)

    write_to_log_file(log_file_path, "-------------- DONE --------------")


if __name__ == "__main__":
    config_path = "config_optimized.yaml"
    main(config_path)