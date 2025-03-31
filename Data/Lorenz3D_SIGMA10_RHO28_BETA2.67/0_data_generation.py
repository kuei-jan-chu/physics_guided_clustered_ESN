import os
import numpy as np
import sys
sys.path.insert(0, "../Utils")
from lorenz import *
import pickle
from scipy.integrate import ode
import sys

base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Simulation_Data/"
os.makedirs(data_dir, exist_ok=True)

# fix random seed for reproducibility
np.random.seed(100)

sigma = 10
rho = 28
beta = 8./3
DIM = 3

T_transients = 1000
T = 2000
dt = 0.01

N_transients = int(np.floor(T_transients/dt))
N = int(np.floor(T/dt))

# Data generation
X0 = np.ones((1,DIM))
X0 = X0 + 0.01 * np.random.randn(1,DIM)
# Initialization
X = np.zeros((N,DIM))

# Get past initial transients
print("Get past initial transients")
for i in range(N_transients):
    X0 = RK4(Lorenz, X0, 0, dt, sigma, rho, beta)
    print("{:d}/{:d}".format((i+1), N_transients), end="\r")
print("\n")
    
# Generate time series
print("\nGenerate time series")
for i in range(N):
    X0 = RK4(Lorenz, X0, 0, dt, sigma, rho, beta)
    X[i,:] = X0
    print("{:d}/{:d}".format((i+1), N),  end="\r")

data = {
    "sigma":sigma,
    "rho":rho,
    "beta":beta,
    "DIM":DIM,
    "T":T,
    "T_transients":T_transients,
    "dt":dt,
    "N":N,
    "N_transients":N_transients,
    "X":X,
}


with open("./Simulation_Data/lorenz3D_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)