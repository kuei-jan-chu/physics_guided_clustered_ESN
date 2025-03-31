import os
import numpy as np
from numpy import pi
import pickle
import sys
sys.path.insert(0, "../Utils")
from kuramoto_sivashinsky import KS

base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Data"
os.makedirs(data_dir, exist_ok=True)

L    = 100/(2*pi)
N    = 256
dt   = 0.25
t_transient = 2500
ninittransients = int(t_transient/dt)
tend = 50000 + t_transient  #50000
# dns  = KS(L=L, N=N, inhomo=0, mu_inhomo=None, lambda_inhomo=None, dt=dt, tend=tend, t_transient=t_transient)
dns  = KS(L=L, N=N, dt=dt, tend=tend, t_transient=t_transient)
# dns  = KS(L=L, N=N, dt=dt, tend=tend)

N_data_train = 100000
N_data_test = 100000
dl_max = 20000
pl_max = 20000


file_name = base_path + "/Simulation_Data/simulation_data.pickle"

with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["U"]
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

with open(base_path + "/Data/testing_data_N{:d}.pickle".format(N_data_test), "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data






