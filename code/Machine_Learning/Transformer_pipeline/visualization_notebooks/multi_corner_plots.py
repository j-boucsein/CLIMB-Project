import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import DataLoader, TensorDataset

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from inference import initialize_trafo_from_saved_state, get_sdss_spectra_for_inference, get_spectra_reference_point
from dataset_functions import get_shuffled_and_split_datasets, SpectraCosmoDataset
from train_model import eval_model


import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D

def make_corner_plot_multi(y_preds, save_path, save_plot=True, show_plot=True):
    """
    y_preds: list of arrays, each shape (N, 4)
    """

    labels = [
        r"$\Omega_{\mathrm{m}}$",
        r"$\Omega_{\mathrm{b}}$",
        r"$\Omega_{\Lambda}$",
        r"$H_0$"
    ]

    # Planck values
    planck = np.array([0.3089, 0.0486, 0.6911, 0.6774])
    planck_err = np.array([0.012, 2.2e-4, 0.009, 0.012])

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2
    })

    fig = None

    fill_colors = [
        (31/255, 119/255, 180/255, 0.7),   # blue with alpha
        (214/255, 39/255, 40/255, 0.7),    # red with alpha
        (44/255, 160/255, 44/255, 0.7),    # green with alpha
    ]

    edge_colors = [
        (31/255, 119/255, 180/255),
        (214/255, 39/255, 40/255, 0.7),
        (44/255, 160/255, 44/255, 0.7),
    ]

    import matplotlib.colors as mcolors

    for i, samples in enumerate(y_preds):
        rgba = mcolors.to_rgba(fill_colors[i], alpha=0.5)

        fig = corner.corner(
            samples,
            fig=fig,
            labels=labels if i == 0 else None,
            bins=50,
            color=rgba,
            smooth=1.0,
            levels=(0.68,),
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            # color=fill_colors[i],  # normal color, no alpha here

            hist_kwargs={"density": True, "linewidth": 1.2, "color": edge_colors[i]}
        )

    axes = np.array(fig.axes).reshape((4, 4))

    # Overlay Planck constraints
    for i in range(4):
        ax = axes[i, i]
        ax.axvline(planck[i], color="black", lw=2)
        ax.axvspan(
            planck[i] - planck_err[i],
            planck[i] + planck_err[i],
            color="black",
            alpha=0.25
        )

        for j in range(i):
            ax = axes[i, j]
            ax.axvline(planck[j], color="black", lw=1)
            ax.axhline(planck[i], color="black", lw=1)

    # Limits (same as your version)
    omega_m_lim = (0.1, 0.5)
    omega_b_lim = (0.0, 0.08)
    omega_L_lim = (0.56, 0.9)

    axes[0, 0].set_xlim(omega_m_lim)
    axes[1, 1].set_xlim(omega_b_lim)
    axes[2, 2].set_xlim(omega_L_lim)

    for i in range(1, 4):
        axes[i, 0].set_xlim(omega_m_lim)
        axes[0, i].set_ylim(omega_m_lim)

    for i in range(4):
        if i != 2:
            axes[2, i].set_ylim(omega_L_lim)
            axes[i, 2].set_xlim(omega_L_lim)

    for i in range(4):
        if i != 1:
            axes[1, i].set_ylim(omega_b_lim)
            axes[i, 1].set_xlim(omega_b_lim)

    plt.tight_layout()

    if save_plot:
        plt.savefig(save_path, format="PDF")
    if show_plot:
        plt.show()

    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


def _kde_2d(x, y, gridsize=200, bw_scale=0.5, bounds=None):
    values = np.vstack([x, y])
    kde = gaussian_kde(values)
    kde.covariance_factor = lambda: kde.scotts_factor() * bw_scale
    kde._compute_covariance()

    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _credible_threshold(density, level=0.68):
    z = density.ravel()
    idx = np.argsort(z)[::-1]
    z_sorted = z[idx]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    return z_sorted[np.searchsorted(cdf, level)]


def make_corner_plot_custom(
    y_preds,
    labels,
    model_names=None,
    axis_limits=None,
    planck=None,
    planck_err=None,
    planck_color="#333333",
    bw_scale=0.4,
    fill_alpha=0.5,
    gridsize=180,
    bins=50,
    figsize=(8, 8),
    save_path=None,
    show_plot=True
):
    n_dim = y_preds[0].shape[1]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_dim, n_dim, figure=fig, wspace=0.0, hspace=0.0)

    axes = np.empty((n_dim, n_dim), dtype=object)

    # ----- axis creation -----
    for i in range(n_dim):
        for j in range(n_dim):

            if i < j:
                axes[i, j] = None
                continue

            sharex = axes[n_dim - 1, j] if i < n_dim - 1 else None
            sharey = axes[i, 0] if (i != j and j > 0) else None

            axes[i, j] = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)

    fill_colors = ["#63a5d4", "#de7171", "#6fd66f", "#cfa2f8", "#ffa556"]
    edge_colors = ["#176395", "#a90c0c", "#199b3c", "#4b0082", "#8c3f00"]

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if ax is None:
                continue

            for k, samples in enumerate(y_preds):
                color = fill_colors[k % len(fill_colors)]
                edge = edge_colors[k % len(edge_colors)]

                # ---- diagonal hist ----
                if i == j:
                    data = samples[:, i]

                    if axis_limits and axis_limits[i] is not None:
                        xmin, xmax = axis_limits[i]
                        data = data[(data >= xmin) & (data <= xmax)]
                        hist_range = (xmin, xmax)
                    else:
                        hist_range = None

                    ax.hist(
                        data,
                        bins=bins,
                        range=hist_range,
                        density= True,
                        histtype="step",
                        linewidth=1.4,
                        color=color
                    )

                    if planck is not None:
                        ax.axvline(planck[i], color=planck_color, lw=1.3, linestyle="--", zorder=5)
                        ax.axvspan(
                            planck[i] - planck_err[i],
                            planck[i] + planck_err[i],
                            color=planck_color,
                            alpha=0.07,
                            zorder=5
                        )

                    ax.tick_params(labelleft=False)

                # ---- 2D contours ----
                else:
                    x = samples[:, j]
                    y = samples[:, i]

                    bounds = None
                    if axis_limits and axis_limits[i] is not None:
                        xmin, xmax = axis_limits[j]
                        ymin, ymax = axis_limits[i]
                        bounds = (xmin, xmax, ymin, ymax)

                    xx, yy, zz = _kde_2d(x, y, gridsize, bw_scale, bounds)
                    thr = _credible_threshold(zz, 0.68)

                    ax.contourf(
                        xx, yy, zz,
                        levels=[thr, zz.max()],
                        colors=[color],
                        alpha=fill_alpha
                    )
                    
                    ax.contour(
                        xx, yy, zz,
                        levels=[thr],
                        colors=[edge],
                        linewidths=1.5,
                        # linestyles="dashed"
                    )

                    if planck is not None:
                        ax.axvline(planck[j], color=planck_color, lw=1.0, linestyle="--")
                        ax.axhline(planck[i], color=planck_color, lw=1.0, linestyle="--")

            # ---- limits ----
            if axis_limits:
                if i == j:
                    ax.set_xlim(axis_limits[i])
                else:
                    ax.set_xlim(axis_limits[j])
                    ax.set_ylim(axis_limits[i])

            # ---- ticks ----
            ax.tick_params(direction="in", top=True, right=True)

            if j == 0 and i != j:
                ax.tick_params(labelleft=True)
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.tick_params(labelleft=False)

            if i == n_dim - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.tick_params(labelbottom=False)

    # ----- legend placed in empty top-right panel -----
    if model_names is not None:
        legend_elements = [
            Patch(
                facecolor=fill_colors[i],
                edgecolor=edge_colors[i],
                linestyle="dashed",
                label=model_names[i],
                alpha=fill_alpha
            )
            for i in range(len(model_names))
        ]

        legend_elements.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="Planck 2015"
            )
        )

        # center of missing upper-right panel
        anchor_x = (n_dim - 0.5) / n_dim
        anchor_y = (n_dim - 0.5) / n_dim

        fig.legend(
            handles=legend_elements,
            loc="center",
            bbox_to_anchor=(anchor_x, anchor_y),
            frameon=False
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()


def predict_loader(model, X, device, batch_size=16):
    loader = DataLoader(TensorDataset(X), batch_size=batch_size)
    preds = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())

    return torch.cat(preds)


model_type = "realistic_model" # "sweep_model"

if model_type == "sweep_model":
    models_to_plot = ["sweep_model_2", "sweep_model_10", "sweep_model_100"]
    datasets_used = ["L25n256_snr_sweep_2", "L25n256_snr_sweep_10", "L25n256_snr_sweep_100"]
    snrs_used = [2, 10, 100]
    sdss_corner_path = f"plots/sweep_multi_cornerplot.pdf"
elif model_type == "realistic_model":
    models_to_plot = ["realistic_noise_model_snr10", "realistic_noise_model_snr5", "realistic_noise_model_snr2"]
    datasets_used = ["L25n256_realistic_noise_v2_snr10", "L25n256_realistic_noise_v2_snr5", "L25n256_realistic_noise_v2_snr2"]
    snrs_used = [10, 5, 2]
    sdss_corner_path = f"plots/SDSS_multi_cornerplot.pdf"
else:
    assert False, "Model not found!"


y_pred_sdss_list = []

for index in range(len(models_to_plot)):

    model_name = models_to_plot[index]
    suite_of_spectra = datasets_used[index]
    snr_filter = snrs_used[index]

    cat_path = "../SDSS_support_files/Custom_cat.npz"
    resid_file = "../SDSS_support_files/residcorr_v5_4_45.dat"
    config_path = f"../log_files/{model_name}_config.yaml"
    state_path = f"../model_states/{model_name}_weights.pt"

    if model_type == "sweep_model":
        sdss_specs, _ = get_spectra_reference_point(suite_of_spectra, n_spectra=10000)
        sdss_specs = torch.Tensor(np.array(sdss_specs))
    elif model_type == "realistic_model":
        sdss_specs = get_sdss_spectra_for_inference(cat_path, resid_file, snr_filter)

    input_len = sdss_specs.shape[1]
    output_len = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer = initialize_trafo_from_saved_state(config_path, input_len, output_len, state_path, device)

    y_pred_sdss = predict_loader(model, sdss_specs, device)

    # TODO: This is an embarrassingly inefficient way to get the y_mean and y_std. Should probably change this in the future
    _, _, _, y_mean, y_std = get_shuffled_and_split_datasets(suite_of_spectra, True)

    y_pred_sdss = y_pred_sdss*y_std + y_mean
    y_pred_sdss = y_pred_sdss.numpy()

    y_pred_sdss_list.append(y_pred_sdss)

if model_type == "sweep_model":
    model_names_plot = [
        "Constant SNR 2 model",
        "Constant SNR 10 model",
        "Constant SNR 100 model"
    ]
    axis_limits = [
        (0.25, 0.37),
        (0.03, 0.075),
        (0.64, 0.739),
        (0.63, 0.74)
    ]
elif model_type == "realistic_model":
    model_names_plot = [
        "Realistic SNR 10 model",
        "Realistic SNR 5 model",
        "Realistic SNR 2 model"
    ]
    axis_limits = [
        (0.1, 0.5),
        (0, 0.08),
        (0.56, 0.87),
        None
    ]


make_corner_plot_custom(
    y_pred_sdss_list,

    labels=[
        r"$\Omega_{\mathrm{m}}$",
        r"$\Omega_{\mathrm{b}}$",
        r"$\Omega_{\Lambda}$",
        r"$H_0$"
    ],

    model_names=model_names_plot,

    axis_limits=axis_limits,

    planck=np.array([0.3089, 0.0486, 0.6911, 0.6774]),
    planck_err=np.array([0.012, 2.2e-4, 0.009, 0.012]),

    bw_scale=1,   # lower = sharper contours
    fill_alpha=0.7,
    gridsize=250,
    bins=50,
    figsize=(8,8),
    planck_color="#000000",

    save_path=sdss_corner_path
)
# make_corner_plot_multi(y_pred_sdss_list, sdss_corner_path)