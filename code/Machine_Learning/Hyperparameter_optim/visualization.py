import optuna
import os
import optuna.visualization as vis


def visualize_optuna_study(study: optuna.Study, out_dir: str = "optuna_plots", prefix: str = "optuna_broad_2", pdf_scale: int = 2):
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
        pdf_path = os.path.join(out_dir, f"{prefix}_{name}.pdf")

        fig.write_html(html_path)

        try:
            fig.write_image(pdf_path, scale=pdf_scale)
        except Exception as e:
            pass

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

    print(f"[Optuna] Saved visualizations to '{out_dir}/'")




study = optuna.load_study(
    study_name="transformer_borad_search_2",
    storage="sqlite:///studies/optuna_borad_search_2_study.db"
)

visualize_optuna_study(study)

trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params:")
for k, v in trial.params.items():
    print( f"    {k}: {v}")

