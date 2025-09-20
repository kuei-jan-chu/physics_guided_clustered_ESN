import os
import numpy as np
import pickle
import sys
sys.path.insert(0, "../Utils")
from lorenz96 import *

# Plotting parameters
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

# fix random seed for reproducibility
np.random.seed(100)

# F=8
RDIM=40
T=20
dt=0.01
epsilon=1e-8    # the perturbation
F=8
perturbation = 1e-8 

base_path = "."

file_name = base_path + "/Simulation_Data/F"+str(F)+"_data.pickle"
with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["X"]
    dt = data["dt"]

# fix random seed for reproducibility
np.random.seed(100)

# time steps number of input sequence
n_steps = np.shape(u)[0]

# Initialize the divergence array, just n_steps - 1 # of entries !
divergence = np.zeros(n_steps - 1)

# Perturb the state
# p = np.random.normal(0, epsilon, size=RDIM)
p = np.random.rand(RDIM)
p /= np.linalg.norm(p)
perturbed_state = np.reshape(u[0], (1,-1)) + perturbation * p

# Iterate over each time step
for t in range(n_steps - 1):
    # Evolve the perturbed state using the updator
    perturbed_state_evolved = RK4(Lorenz96,perturbed_state,0,dt, F, 10)

    # Compute the Euclidean distance between the perturbed and evolved states
    p = (perturbed_state_evolved - u[t + 1])
    distance = np.linalg.norm(p)
        
    p /= distance
    perturbed_state = np.reshape(u[t+1], (1,-1)) + perturbation * p
    
    # Update the divergence array
    divergence[t] = np.log(distance / perturbation)/dt

# Calculate the average Lyapunov exponent
lyapunov_exponent = np.mean(divergence)
print("\nMAXIMUM LYAPUNOV EXPONENT = {:}".format(lyapunov_exponent))

data["MLE"] = lyapunov_exponent
with open(base_path + "/Simulation_Data/" + "F"+str(F)+"_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data