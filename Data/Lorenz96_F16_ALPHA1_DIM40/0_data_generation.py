#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
import sys
sys.path.insert(0, "../Utils")
from lorenz96 import *
import os # for saving


base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Simulation_Data/"
os.makedirs(data_dir, exist_ok=True)

# fix random seed for reproducibility
np.random.seed(100)

# Forcing term
F=16
DIM=40
print("# F="+str(F))
base_path.format(F)

T_transients = 1000
T = 2000
dt = 0.01

N_transients = int(np.floor(T_transients/dt))
N = int(np.floor(T/dt))
print("Script to generate data for Lorenz96 for F={:d}".format(F))
print("Generating time-series with {:d} data-points".format(N))

# Data generation
X0 = F*np.ones((1,DIM))
X0 = X0 + 0.01 * np.random.randn(1,DIM)

# Initialization
X = np.zeros((N,DIM))


print("Get past initial transients\n")
# Get past initial transients
for i in range(N_transients):
    X0 = RK4(Lorenz96,X0,0,dt, F, 1);
    print("{:d}/{:d}".format((i+1), N_transients), end="\r")

print("\nGenerate time series\n")
# Generate time series
for i in range(N):
    X0 = RK4(Lorenz96,X0,0,dt, F, 1);
    X[i,:] = X0
    print("{:d}/{:d}".format((i+1), N),  end="\r")
    sys.stdout.write("\033[F")


data = {
    "F":F,
    "DIM":DIM,
    "T":T,
    "T_transients":T_transients,
    "dt":dt,
    "N":N,
    "N_transients":N_transients,
    "X":X,
}

with open(base_path + "/Simulation_Data/" + "F"+str(F)+"_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data




