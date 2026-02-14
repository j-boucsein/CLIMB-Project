import os
import optuna
import optuna.visualization as vis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_model import Transformer
from data_handling import get_shuffled_and_split_datasets


# --------------------------------------------------
# Training utilities
# --------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# --------------------------------------------------
# Optuna objective
# --------------------------------------------------

def objective(trial):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Hyperparameter search space
    # ------------------------------
    d_model = trial.suggest_int("d_model", 64, 256)

    num_heads = trial.suggest_categorical("num_heads", [2, 4, 6, 8])

    if d_model % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    num_layers = trial.suggest_int("num_layers", 2, 6)

    d_ff_mult = trial.suggest_int("d_ff_mult", 1, 8)

    d_ff = d_model * d_ff_mult

    dropout = trial.suggest_float("dropout", 0.0, 0.1)

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])


    # ------------------------------
    # Data
    # ------------------------------
    train_ds, val_ds, _, _, _ = get_shuffled_and_split_datasets(suite_of_spectra=suite_to_use)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # ------------------------------
    # Model
    # ------------------------------
    model = Transformer(
        len_spectra=train_ds.X.shape[1],
        len_output=train_ds.y.shape[1],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # ------------------------------
    # Training loop
    # ------------------------------
    max_epochs = 15
    best_val = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(max_epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = evaluate(model, val_loader, criterion, device)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val


def visualize_optuna_study(study: optuna.Study, out_dir: str = "optuna_plots", prefix: str = "optuna_broad", pdf_scale: int = 2):
    """
    Generate and save a comprehensive set of Optuna visualizations.

    Saves both HTML (interactive) and PDF (static) versions.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    out_dir : str
        Directory to save plots
    prefix : str
        Filename prefix
    pdf_scale : int
        Resolution scaling for PDF export (2–3 recommended)
    """

    os.makedirs(out_dir, exist_ok=True)

    def _save(fig, name):
        html_path = os.path.join(out_dir, f"{prefix}_{name}.html")
        fig.write_html(html_path)

    # --------------------------------------------------
    # Core plots
    # --------------------------------------------------
    _save(vis.plot_optimization_history(study), "history")
    _save(vis.plot_param_importances(study), "param_importance")
    _save(vis.plot_parallel_coordinate(study), "parallel_coords")
    _save(vis.plot_slice(study), "slice")

    # --------------------------------------------------
    # Contour plots (pairwise interactions)
    # --------------------------------------------------
    contour_pairs = [
        ("lr", "batch_size"),
        ("lr", "weight_decay"),
        ("d_model", "num_layers"),
        ("d_model", "dropout"),
    ]

    available_params = set(study.best_trial.params.keys())

    for p1, p2 in contour_pairs:
        if p1 in available_params and p2 in available_params:
            fig = vis.plot_contour(study, params=[p1, p2])
            _save(fig, f"contour_{p1}_vs_{p2}")


# --------------------------------------------------
# Run study
# --------------------------------------------------

if __name__ == "__main__":

    study_name = "final_sweep"
    suite_to_use = "L25n256snr100_6095_sas"

    study = optuna.create_study(
        study_name=f"transformer_{study_name}",
        direction="minimize",
        storage=f"sqlite:///studies/optuna_{study_name}.db",
        load_if_exists=True,
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2
        )
    )

    study.optimize(objective, n_trials=500)
