import os
import numpy as np
from numpy import pi
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
dt   = 0.25
t_transient = 10000
ninittransients = int(t_transient/dt)
tend = 50000 + t_transient  #60000
# dns  = KS(L=L, N=N, inhomo=0, mu_inhomo=None, lambda_inhomo=None, dt=dt, tend=tend, t_transient=t_transient)
dns  = KS(L=L, N=N, dt=dt, tend=tend, t_transient=t_transient)
# dns  = KS(L=L, N=N, dt=dt, tend=tend)

N_data_train = 100000
N_data_test = 100000
dl_max = 20000
pl_max = 20000


#------------------------------------------------------------------------------
# simulate initial transient
dns.simulate()
# convert to physical space
dns.fou2real()

u = dns.uu[ninittransients+1:]
print("Simulated data shape: ", u.shape)


data = {"U":u,
        "L":L,
        "N":N, 
        "dt":dt}

with open(base_path + "/Simulation_Data/simulation_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    
del data





