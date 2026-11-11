import yaml
import os

from spectra_run import generate_short_spectra
from spectra_stitching import create_forest_spectra
from noise_and_masking import apply_noise_and_mask


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

    return training_boxes_paths, reference_boxes_paths


def main():
    log_file_path = "logs/spectra_generation.log" # TODO: include timestamp to make this name unique
    initialize_log(log_file_path)

    config_path = "spectra_creation_config.yaml"
    # read config file
    config_args = read_config(config_path)

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
        sdss_specs_hash = apply_noise_and_mask(box_path, train_data_out_path, data_splits, wavelength_range_train_min, wavelength_range_train_max, long_spectra_hash, noise, masking, noise_type, snr, spectra_cache_file, min_valid_pixels, rng=random_seed)

    # reference boxes:
    reference_data_out_path = os.path.join(out_folder_path, "reference_boxes")
    os.makedirs(reference_data_out_path)
    
    for box_path in reference_boxes_paths:
        sdss_specs_hash = apply_noise_and_mask(box_path, reference_data_out_path, data_splits, wavelength_range_train_min, wavelength_range_train_max, long_spectra_hash, noise, masking, noise_type, snr, spectra_cache_file, min_valid_pixels, rng=random_seed)

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
    

if __name__ == "__main__":
    main()