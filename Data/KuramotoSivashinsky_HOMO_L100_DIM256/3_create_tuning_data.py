#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np
import os # for saving
import pickle

base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Data"
os.makedirs(data_dir, exist_ok=True)

# max dynamical length for wash out and prediction lenght for test
dl_max = 20000
pl_max = 20000
N_data_train = 100000

file_name = base_path + f"/Data/training_data_N{N_data_train}.pickle"

with open(file_name, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            dt = data["dt"]
            mle = data["mle"]
            del data

time_series_length = np.shape(train_input_sequence)[0]
trained_time_series_length = int(time_series_length*0.6)
validate_time_series_length = time_series_length - trained_time_series_length

[train_input_sequence_for_hyp_tuning, validate_input_sequence_for_hyp_tuning, _] = \
    np.split(train_input_sequence, [trained_time_series_length, time_series_length], axis=0)

print("Traing data shape: ")
print(train_input_sequence_for_hyp_tuning.shape)

print("Test data shape: ")
print(validate_input_sequence_for_hyp_tuning.shape)

hyp_tuning_data_path = base_path + f"/Data/hypTuning"  
os.makedirs(hyp_tuning_data_path, exist_ok=True)

# generate the initial index for test data
max_idx = np.shape(train_input_sequence_for_hyp_tuning)[0] - pl_max
min_idx = dl_max
idx = np.arange(min_idx, max_idx)
np.random.shuffle(idx)

attractor_std = np.std(train_input_sequence_for_hyp_tuning, axis=0)
attractor_std = np.array(attractor_std).flatten()

data = {"train_input_sequence":train_input_sequence_for_hyp_tuning,
        "testing_ic_indexes":idx,
        "attractor_std":attractor_std, 
        "dt":dt,
        "mle":mle
    }

train_data_path = hyp_tuning_data_path+ f"/training_data_N{trained_time_series_length}.pickle"   
with open(train_data_path, "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data

# generate the initial index for test data
max_idx = np.shape(validate_input_sequence_for_hyp_tuning)[0] - pl_max
min_idx = dl_max
idx = np.arange(min_idx, max_idx)
np.random.shuffle(idx)

attractor_std = np.std(validate_input_sequence_for_hyp_tuning, axis=0)
attractor_std = np.array(attractor_std).flatten()
data = {"test_input_sequence":validate_input_sequence_for_hyp_tuning,
        "testing_ic_indexes":idx,
        "attractor_std":attractor_std, 
        "dt":dt,
        "mle":mle
    }
test_data_path = hyp_tuning_data_path+ f"/testing_data_N{validate_time_series_length}.pickle"
with open(test_data_path, "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data