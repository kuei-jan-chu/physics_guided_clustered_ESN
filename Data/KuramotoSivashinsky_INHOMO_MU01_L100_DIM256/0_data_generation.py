import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.insert(0, "../Utils")
from kuramoto_sivashinsky import KS


base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Simulation_Data/"
os.makedirs(data_dir, exist_ok=True)

# fix random seed for reproducibility
np.random.seed(100)

#------------------------------------------------------------------------------
# define data and initialize simulation
# L    = 100/(2*pi)
# N    = 512
L    = 100/(2*pi)
N    = 256
mu_inhomo = 0.1
lambda_inhomo = 2*pi*L/4
dt   = 0.25
t_transient = 10000
ninittransients = int(t_transient/dt)
tend = 50000 + t_transient  #60000
dns  = KS(L=L, N=N, inhomo=1, mu_inhomo=mu_inhomo, lambda_inhomo=lambda_inhomo, dt=dt, tend=tend, t_transient=t_transient)


N_data_train = 100000
N_data_test = 100000
dl_max = 20000
pl_max = 20000


#------------------------------------------------------------------------------
# simulate initial transient
dns.simulate()
# convert to physical space
dns.fou2real()

u = dns.uu[ninittransients+1:]      # neglect the initial value
print("Simulated data shape: ", u.shape)

data = {"U":u,
        "L":L,
        "N":N, 
        "dt":dt}

with open(base_path + "/Simulation_Data/simulation_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    
del data





