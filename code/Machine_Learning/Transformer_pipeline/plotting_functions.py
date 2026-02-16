import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import corner
import numpy as np


def make_corner_plot(y_pred, save_path, save_plot=True, show_plot=True):
    # Your inferred samples
    samples = y_pred  # shape (N, 4)

    labels = [
        r"$\Omega_{\mathrm{m}}$",
        r"$\Omega_{\mathrm{b}}$",
        r"$\Omega_{\Lambda}$",
        r"$H_0$"
    ]

    # Planck values
    planck = np.array([0.315, 0.0493, 0.685, 0.674])
    planck_err = np.array([0.007, 2.2e-4, 0.007, 0.005])

    # Global matplotlib style tweaks (very Planck-y)
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2
    })

    fig = corner.corner(
        samples,
        labels=labels,
        bins=50,
        smooth=1.2,
        color="#1f77b4",
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.68, 0.95),
        contour_kwargs={"linewidths": [1, 1]},
        hist_kwargs={"density": True, "linewidth": 1.6},
    )

    axes = np.array(fig.axes).reshape((4, 4))

    # Overlay Planck constraints
    for i in range(4):
        # Diagonal: 1D constraints
        ax = axes[i, i]
        ax.axvline(planck[i], color="black", lw=2)
        ax.axvspan(
            planck[i] - planck_err[i],
            planck[i] + planck_err[i],
            color="black",
            alpha=0.25
        )

        # Off-diagonal: Planck crosshairs
        for j in range(i):
            ax = axes[i, j]
            ax.axvline(planck[j], color="black", lw=1)
            ax.axhline(planck[i], color="black", lw=1)

    axes = np.array(fig.axes).reshape((4, 4))

    # Define tighter limits (tune as needed)
    omega_m_lim = (0.1, 0.5)
    omega_b_lim = (0.0, 0.08)
    omega_L_lim = (0.56, 0.9)

    # Diagonal panels
    axes[0, 0].set_xlim(omega_m_lim)
    axes[1, 1].set_xlim(omega_b_lim)
    axes[2, 2].set_xlim(omega_L_lim)

    # Off-diagonal panels involving Ωm
    for i in range(1, 4):
        axes[i, 0].set_xlim(omega_m_lim)
        axes[0, i].set_ylim(omega_m_lim)

    # Off-diagonal panels involving ΩΛ
    for i in range(4):
        if i != 2:
            axes[2, i].set_ylim(omega_L_lim)
            axes[i, 2].set_xlim(omega_L_lim)

    # Off-diagonal panels involving Ωb
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


def plot_selected_spectra(spectra, wavelengths, save_path, save_plot=True):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    gs = gridspec.GridSpec(len(spectra), 1, hspace=0)
    axes = []

    for i in range(len(spectra)):
        ax = fig.add_subplot(gs[i, 0])

        ax.plot(wavelengths[i], spectra[i])
        ax.set_ylabel("Relative Flux")
        ax.set_xlim([3600, 3950])
        ax.tick_params(axis='x', direction='in')

        if i == len(spectra)-1:
            ax.set_xlabel(r"Wavelength [$\AA$]")
        if i < len(spectra)-1:
            ax.tick_params(labelbottom=False)

        axes.append(ax)

    if save_plot:
        plt.savefig(save_path, format="PDF")
    plt.show()


def plot_test_inference_errorbars(y_true, y_pred, save_path, save_plot=False):
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

        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_title(f"{params[index]}")
        ax.legend(prop={'size': 8})

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_path, format="PDF")
    plt.show()


def plot_test_inference_colors(y_true, y_pred, save_path, save_plot=False):
    params = ["Omega_m", "Omega_b", "Omega_L", "H0"]

    y_true_by_param = [y_true[:, i] for i in range(len(params))]
    y_pred_by_param = [y_pred[:, i] for i in range(len(params))]

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for index in range(4):

        y_true_par = y_true_by_param[index]
        y_pred_par = y_pred_by_param[index]

        # ---------- mean and std per true value ----------
        y_pred_mean_par = []
        y_pred_std_par = []
        y_true_unique_par = []

        for true_value in np.unique(y_true_par):
            y_pred_this_true_value = y_pred_par[y_true_par == true_value]
            y_pred_mean_par.append(y_pred_this_true_value.mean())
            y_pred_std_par.append(y_pred_this_true_value.std())
            y_true_unique_par.append(true_value)

        # ---------- equally spaced bins ----------
        n_bins =  35 # len(np.unique(y_true_par))

        x_min, x_max = y_true_par.min(), y_true_par.max()
        y_min, y_max = y_pred_par.min(), y_pred_par.max()

        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

        # ---------- 2D histogram ----------
        H, _, _ = np.histogram2d(
            y_true_par,
            y_pred_par,
            bins=[x_edges, y_edges]
        )

        # sum over y for each x-bin
        col_sums = H.sum(axis=1, keepdims=True)

        # avoid division by zero for empty columns
        H_norm = np.divide(
            H,
            col_sums,
            out=np.zeros_like(H),
            where=col_sums > 0
        )

        H_norm[H_norm == 0] = np.nan

        ax = axs[index % 2, index // 2]

        # ---------- density image ----------
        im = ax.imshow(
            H_norm.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="magma",
            norm=LogNorm(vmin=np.nanmin(H_norm), vmax=1.0),
            aspect="auto",
            zorder=1
        )

        # ---------- y = x line ----------
        line_min = min(x_min, y_min)
        line_max = max(x_max, y_max)

        ax.plot(
            [line_min, line_max],
            [line_min, line_max],
            linestyle="--",
            c="black",
            alpha=0.5,
            zorder=3
        )

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title(params[index])

    # ---------- colorbar ----------
    cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
    cbar.set_label("P(y_pred | x_true)")
    if save_plot:
        plt.savefig(save_path, format="PDF")
    plt.show()


def plot_all_PDFs(y_true, y_pred, save_path, save_plot=True):

    params = ["Omega_m", "Omega_b", "Omega_L", "H0"]
    params_plot_name = [r"$\Omega_m$", r"$\Omega_b$", r"$\Omega_\Lambda$", r"$H_0$"]

    for index in range(4):

        y_true_par = y_true[:, index]
        y_pred_par = y_pred[:, index]

        # ---------- unique true values ----------
        y_true_unique = np.sort(np.unique(y_true_par))

        x_edges = []
        x_edges.append(y_true_unique[0]/2)  # manually add first edge
        for i in range(len(y_true_unique)-1):
            x_edges.append(y_true_unique[i]+(y_true_unique[i+1]-y_true_unique[i])/2)
        x_edges.append(y_true_unique[-1]+(y_true_unique[-1]-y_true_unique[-2])/2)  # manually add last edge
        x_edges = np.array(x_edges)

        # make y bins equally spaced
        n_bins = len(y_true_unique)
        y_min, y_max = y_pred_par.min(), y_pred_par.max()
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

        # ---------- 2D histogram ----------
        H, _, _ = np.histogram2d(
            y_true_par,
            y_pred_par,
            bins=[x_edges, y_edges]
        )

        # ---------- column-wise normalization ----------
        col_sums = H.sum(axis=1, keepdims=True)
        H_norm = np.divide(
            H,
            col_sums,
            out=np.zeros_like(H),
            where=col_sums > 0
        )

        # ---------- remove zeros ----------
        H_norm[H_norm == 0] = np.nan

        # ---------- y-bin centers ----------
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])


        n_x = len(y_true_unique)

        # count non-empty columns
        non_empty_indices = [i for i in range(n_x) if not np.all(np.isnan(H_norm[i, :]))]

        n_non_empty = len(non_empty_indices)
        n_cols = 5  # or any number you like
        n_rows = int(np.ceil(n_non_empty / n_cols))

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(2*n_cols, int(n_rows/2)),
            sharex=True,
            sharey=True
        )

        fig.suptitle(rf"Infered conditional PDF slices for {params_plot_name[index]}", fontsize=14, y=0.995)

        # normalize x-values to [0,1] for colormap
        norm_x = (y_true_unique[non_empty_indices] - y_true_par.min()) / (y_true_par.max() - y_true_par.min())
        colors = plt.cm.winter(norm_x)

        # flatten axs to make indexing easier
        axs = axs.T.flatten()

        for plot_idx, i in enumerate(non_empty_indices):
            x_val = y_true_unique[i]
            hist_1d = H_norm[i, :]
            
            ax = axs[plot_idx]
            ax.plot(y_centers, hist_1d, lw=2.0, color=colors[plot_idx])
            ax.axvline(x=x_val, color="black", linestyle="-")


        for j in range(len(non_empty_indices), len(axs)):
            axs[j].axis("off")

        plt.subplots_adjust(
            left=0.05,
            right=0.98,
            top=0.94,
            bottom=0.05,
            wspace=0.0,
            hspace=0.0
        )    


        for ax in axs[:n_non_empty]:
            # set log y-axis
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-4)

            # manually enable minor ticks for the y-axis
            from matplotlib.ticker import LogLocator
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0, 3)*0.1, numticks=4))

            # ticks inside, no labels
            ax.tick_params(
                axis='both',
                which='both',       # major AND minor
                labelbottom=False,
                labelleft=False,
                direction='in',
                length=3
            )
            
        if save_plot:
            save_path_mod = save_path.split(".")[0] + f"_{params[index]}." + save_path.split(".")[-1]
            plt.savefig(save_path_mod, format="PDF")
        plt.show()


def plot_curvature_param(y_true, y_pred, save_path, save_plot=True):
    params = ["Omega_m", "Omega_b", "Omega_L", "H0"]

    y_true_by_param = [y_true[:, i] for i in range(len(params))]
    y_pred_by_param = [y_pred[:, i] for i in range(len(params))]

    fig, axs = plt.subplots(1, 1, figsize=(7, 4))

    y_true_Om = y_true_by_param[0]
    y_true_OL = y_true_by_param[2]

    y_pred_Om = y_pred_by_param[0]
    y_pred_OL = y_pred_by_param[2]

    y_true_par = y_true_Om

    # ---------- mean and std per true value ----------
    y_true_unique_par = []
    y_sum_Om_OL = []
    y_sum_Om_OL_mean = []
    y_sum_Om_OL_std = []

    for true_value in np.unique(y_true_par):
        y_pred_this_true_Om = y_pred_Om[y_true_par == true_value]
        y_pred_this_true_OL = y_pred_OL[y_true_par == true_value]
        Om_OL_sum = 1 - (y_pred_this_true_Om + y_pred_this_true_OL)
        y_sum_Om_OL.append(Om_OL_sum)
        y_sum_Om_OL_mean.append(Om_OL_sum.mean())
        y_sum_Om_OL_std.append(Om_OL_sum.std())
        y_true_unique_par.append(true_value)

    axs.plot(y_true_unique_par, [0 for _ in range(len(y_true_unique_par))], linestyle="--", c="red")
    axs.errorbar(y_true_unique_par, y_sum_Om_OL_mean, y_sum_Om_OL_std, y_sum_Om_OL_std, linestyle="None", marker="o", c="black", label=r"1-($\Omega_m$+$\Omega_\Lambda$) mean")
   
    axs.set_ylim([-4e-3, 4e-3])
    axs.set_xlabel(r"true $\Omega_m$")
    axs.set_title("Infered curvature parameter")
    axs.legend()

    if save_plot:
        plt.savefig(save_path, format="PDF")
    plt.show()