import yaml
import os
import time
import shutil

from spectra_run import generate_short_spectra
from spectra_stitching import create_forest_spectra, box_rng
from noise_and_masking import apply_noise_and_mask

# folder of this script, so that the config and the logs are found from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_config(config_path):
    with open(config_path) as f:
        params = yaml.safe_load(f)

    return params


def initialize_log(log_path):
    with open(log_path, "w") as log_file:
        log_file.write("                      Spectra Generation Log                      \n")
        log_file.write("==================================================================\n\n")


def write_log(log_path, message):
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


def validate_config(config_args):
    """Check the settings that would otherwise only fail late (in the noise step) with an unclear error.

    Args:
        config_args (dict): Parsed spectra_creation_config.yaml.

    Raises:
        ValueError: Listing every invalid setting at once.
    """
    c = config_args
    errors = []
    uses_sdss = (c["noise"] and c["noise_type"] == "realistic") or c["masking"]

    if c["noise"]:
        if c["noise_type"] not in ("realistic", "constant"):
            errors.append(f"noise_type must be 'realistic' or 'constant', got {c['noise_type']!r}")
        if not isinstance(c["snr"], (int, float)) or c["snr"] <= 0:
            errors.append(f"snr must be a positive number when noise is on, got {c['snr']!r}")
        if c["noise_type"] == "realistic" and not c["masking"]:
            errors.append("realistic noise needs masking: SDSS spectra without any masked pixel do not exist in a typical window, so no noise template would pass")
    if uses_sdss and not (c["spectra_cache_file"] and os.path.isfile(c["spectra_cache_file"])):
        errors.append(f"spectra_cache_file is needed for realistic noise or masking, but {c['spectra_cache_file']!r} is not a file")
    if c["masking"] and (not isinstance(c["min_valid_pixels"], int) or c["min_valid_pixels"] <= 0):
        errors.append(f"min_valid_pixels must be a positive integer when masking is on, got {c['min_valid_pixels']!r}")

    long_min, long_max = c["wavelength_range_long_specs"]
    train_min, train_max = c["wavelength_range_train_specs"]
    if not (long_min <= train_min < train_max <= long_max):
        errors.append(f"wavelength_range_train_specs {[train_min, train_max]} must lie inside wavelength_range_long_specs {[long_min, long_max]}")

    if errors:
        raise ValueError("Invalid config:\n  " + "\n  ".join(errors))


def get_box_paths(config_args):

    # extract training boxes paths from config
    training_suite = config_args["training_boxes"]["suite_base"]
    if training_suite is None:
        raise ValueError("Training suite base path is not specified in the configuration file.")
    training_boxes_paths = [f"{training_suite}/gridpoint{i}" for i in range(50)]

    additional_training_boxes = config_args["training_boxes"]["extra_training_boxes"]
    if additional_training_boxes:
        training_boxes_paths.extend(additional_training_boxes)

    # extract reference box path from config
    reference_boxes_paths = []
    reference_suite = config_args["reference_boxes"]["suite_base"]
    if reference_suite:
        reference_boxes_paths.append(f"{reference_suite}/reference")
    additional_reference_boxes = config_args["reference_boxes"]["extra_reference_boxes"]
    if additional_reference_boxes:
        reference_boxes_paths.extend(additional_reference_boxes)

    # normalise once (e.g. strip trailing slashes), so that every step derives the same box name and hash
    training_boxes_paths = [os.path.normpath(p) for p in training_boxes_paths]
    reference_boxes_paths = [os.path.normpath(p) for p in reference_boxes_paths]

    return training_boxes_paths, reference_boxes_paths


def main():
    # unique log name per run: start time, plus the slurm job id if there is one (matches job_messages/job.*.<id>)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    log_name = f"spectra_generation_{timestamp}" + (f"_job{job_id}" if job_id else "") + ".log"
    log_file_path = os.path.join(SCRIPT_DIR, "logs", log_name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    initialize_log(log_file_path)

    config_path = os.path.join(SCRIPT_DIR, "spectra_creation_config.yaml")
    # read config file
    config_args = read_config(config_path)
    validate_config(config_args)

    # get random seed from config_args
    random_seed = config_args["random_seed"]

    # get all box paths from config_args
    training_boxes_paths, reference_boxes_paths = get_box_paths(config_args)

    # Combine training and reference boxes paths
    all_boxes_paths = training_boxes_paths + reference_boxes_paths
    ""
    log_message = f"                Starting short spectra generation                 "
    write_log(log_file_path, log_message)
    log_message = f"=================================================================="
    write_log(log_file_path, log_message)

    log_message = f"Read {len(all_boxes_paths)} target boxes from config file."
    write_log(log_file_path, log_message)
    
    # get redshifts from config_args
    redshifts = config_args["snapshot_redshifts"]
    
    log_message = f"Read redshifts: {redshifts} from config file."
    write_log(log_file_path, log_message)

    log_message = f"Starting spectra generation for {len(all_boxes_paths)} boxes and {len(redshifts)} redshifts:"
    write_log(log_file_path, log_message)

    # generate short spectra for each box and redshift
    for box_path in all_boxes_paths:
        write_log(log_file_path, f"Processing box: {box_path}")
        generate_short_spectra(box_path, redshifts, log_file_path)

    log_message = f"\n\n              Starting long spectra stitching                     "
    write_log(log_file_path, log_message)
    log_message = f"=================================================================="
    write_log(log_file_path, log_message)

    # get the wavelength range from config_args
    wavelength_range_long_specs = tuple(config_args["wavelength_range_long_specs"])

    # get data splits from config_args
    data_splits = {
        name: (tuple(split["region"]), split["num_spectra"])
        for name, split in config_args["data_splits"].items()
    }

    log_message = f"Read wavelength range: {wavelength_range_long_specs} from config file."
    write_log(log_file_path, log_message)
    log_message = f"Read data splits: {data_splits} from config file."
    write_log(log_file_path, log_message)

    # generate long spectra for each box
    for box_path in all_boxes_paths:
        write_log(log_file_path, f"Stitching spectra for box: {box_path}")
        _, long_spectra_hash = create_forest_spectra(box_path, wavelength_range_long_specs, redshifts, data_splits, save_spectra=True, seed=random_seed, log_path=log_file_path)

    log_message = f"\n\n              Starting noise and masking                      "
    write_log(log_file_path, log_message)
    log_message = f"=================================================================="
    write_log(log_file_path, log_message)

    # Get noise and masking parameters from config_args
    wavelength_range_train_min = config_args["wavelength_range_train_specs"][0]
    wavelength_range_train_max = config_args["wavelength_range_train_specs"][1]
    noise = config_args["noise"]
    noise_type = config_args["noise_type"]
    snr = config_args["snr"]
    masking = config_args["masking"]
    min_valid_pixels = config_args["min_valid_pixels"]
    spectra_cache_file = config_args["spectra_cache_file"]

    log_message =f"Reading parameters from config file:\nWavelengts: ({wavelength_range_train_min}, {wavelength_range_train_max}\nNoise: {noise}  -  Type: {noise_type}  -  SNR: {snr}\nMasking: {masking}  -  Min valid pixels: {min_valid_pixels}\nSpectra Cache File: {spectra_cache_file})"
    write_log(log_file_path, log_message)

    out_folder_base = config_args["out_path_base"]
    folder_name = f"dataset_wi{wavelength_range_train_min}{wavelength_range_train_max}_n{noise}{snr}_m{masking}"

    # find new folder name
    out_folder_path = os.path.join(out_folder_base, folder_name)
    n = 1
    while os.path.exists(out_folder_path):
        out_folder_path = os.path.join(out_folder_base, f"{folder_name}v{n}")
        n += 1
    
    # make the new output folder
    os.makedirs(out_folder_path)

    log_message = f"Created new Folder for Dataset: {out_folder_path}"
    write_log(log_file_path, log_message)

    # TODO: add logging to the apply_noise_and_mask code
    # training boxes:
    train_data_out_path = os.path.join(out_folder_path, "training_boxes")
    os.makedirs(train_data_out_path)

    for box_path in training_boxes_paths:
        sdss_specs_hash = apply_noise_and_mask(box_path, train_data_out_path, data_splits, wavelength_range_train_min, wavelength_range_train_max, long_spectra_hash, noise, masking, noise_type, snr, spectra_cache_file, min_valid_pixels, rng=box_rng(random_seed, box_path, stage=1))

    # reference boxes:
    reference_data_out_path = os.path.join(out_folder_path, "reference_boxes")
    os.makedirs(reference_data_out_path)
    
    for box_path in reference_boxes_paths:
        sdss_specs_hash = apply_noise_and_mask(box_path, reference_data_out_path, data_splits, wavelength_range_train_min, wavelength_range_train_max, long_spectra_hash, noise, masking, noise_type, snr, spectra_cache_file, min_valid_pixels, rng=box_rng(random_seed, box_path, stage=1))

    # Make a dataset description file
    configs_used_file_name = "Configs_used.txt"

    content = (
        "                     Dataset Configuration                        \n"
        "==================================================================\n\n"
        f"Config file: {config_path}\n"
        f"Output folder: {out_folder_path}\n\n"
        "Hashes\n"
        "------------------------------------------------------------------\n"
        f"long_spectra_hash: {long_spectra_hash}\n"
        f"sdss_specs_hash: {sdss_specs_hash}\n\n"
        "Config arguments\n"
        "------------------------------------------------------------------\n"
        + yaml.safe_dump(config_args, sort_keys=False, default_flow_style=False)
    )

    with open(os.path.join(out_folder_path, configs_used_file_name), "w") as conf_f:
        conf_f.write(content)

    # keep a copy of the log with the dataset
    write_log(log_file_path, f"\nDone! Copying this log to {out_folder_path}")
    shutil.copy(log_file_path, os.path.join(out_folder_path, log_name))


if __name__ == "__main__":
    main()