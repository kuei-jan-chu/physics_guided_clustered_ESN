import numpy as np


def standardize(input_sequence):
    np.random.seed(0)  # For reproducibility
    data_std = np.std(input_sequence,0)
    data_mean = np.mean(input_sequence,0)
    safe_std = np.where(data_std == 0, 1, data_std)
    input_sequence = np.array((input_sequence-data_mean)/safe_std)
    return input_sequence