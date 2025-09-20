import numpy as np
import os # for saving
import pickle

base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Data"
os.makedirs(data_dir, exist_ok=True)

F=12

# max dynamical length for wash out and prediction lenght for test
dl_max = 2000
pl_max = 10000
N_data_train = 100000
N_data_test = 100000


file_name = base_path + "/Simulation_Data/F"+str(F)+"_data.pickle"

with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["X"]
    dt = data["dt"]
    mle = data["MLE"]
    del data

N = np.shape(u)[0]

print("Total data shape: ")
print(u.shape)


[u_train, u_test, _] = np.split(u, [N_data_train, N_data_train+N_data_test], axis=0)
print("Traing data shape: ")
print(u_train.shape)

print("Test data shape: ")
print(u_test.shape)

train_input_sequence = u_train
test_input_sequence = u_test

max_idx = np.shape(test_input_sequence)[0] - pl_max
min_idx = dl_max
testing_ic_indexes = np.arange(min_idx, max_idx)
np.random.shuffle(testing_ic_indexes)

attractor_std = np.std(train_input_sequence, axis=0)
attractor_std = np.array(attractor_std).flatten()
print("attractor_std shape: ")
print(np.shape(attractor_std))

print("train_input_sequence shape: ")
print(train_input_sequence.shape)

data = {"train_input_sequence":train_input_sequence,
        "testing_ic_indexes":testing_ic_indexes,
        "attractor_std":attractor_std, 
        "dt":dt,
        "mle":mle
    }

with open(base_path + "/Data/training_data_N{:d}.pickle".format(N_data_train), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data

attractor_std = np.std(test_input_sequence, axis=0)
attractor_std = np.array(attractor_std).flatten()

data = {"test_input_sequence":test_input_sequence,
        "testing_ic_indexes":testing_ic_indexes,
        "attractor_std":attractor_std, 
        "dt":dt,
        "mle":mle
    }

print("test_input_sequence shape: ")
print(test_input_sequence.shape)

# np.savetxt(base_path + "/Data/testing_data_N{:d}.txt".format(N_data_test), test_input_sequence, fmt="%.6f", delimiter=",")
with open(base_path + "/Data/testing_data_N{:d}.pickle".format(N_data_test), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data







