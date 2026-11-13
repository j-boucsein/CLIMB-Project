import numpy as np


def find_closest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def write_log(log_path, message):
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
