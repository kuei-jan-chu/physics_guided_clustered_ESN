import os
import sys
import numpy as np
import pickle
sys.path.insert(0, "../Utils")
from kuramoto_sivashinsky import KS

# Plotting parameters
import matplotlib.pyplot as plt
from numpy import pi
from scipy.fftpack import ifft

plt.rcParams["text.usetex"] = False
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

T=500
dt=0.25
perturbation=1e-8    # the perturbation
L    = 100/(2*pi)
N    = 256
mu_inhomo = 0.1
lambda_inhomo = 2*pi*L/4
dt   = 0.25
dns_1  = KS(L=L, N=N, inhomo=1, mu_inhomo=mu_inhomo, lambda_inhomo=lambda_inhomo, dt=dt)
dns_2  = KS(L=L, N=N, inhomo=1, mu_inhomo=mu_inhomo, lambda_inhomo=lambda_inhomo, dt=dt)


base_path = "."
file_name = base_path + "/Simulation_Data/simulation_data.pickle"
with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["U"]
    dt = data["dt"]


# fix random seed for reproducibility
np.random.seed(100)

# time steps number of input sequence
n_steps = np.shape(u)[0]

# Initialize the divergence array, just n_steps - 1 # of entries !
divergence = np.zeros(n_steps - 1)

# Perturb the state
# p = np.random.normal(0, epsilon, size=RDIM)
p = np.random.rand(N)
p /= np.linalg.norm(p)
perturbed_state = np.reshape(u[0], (1,-1)) + perturbation * p

# Iterate over each time step
for t in range(n_steps - 1):
    # Evolve the perturbed state using the updator
    dns_1.IC(np.reshape(u[t], (1,-1)).flatten())
    state_evoled = np.real(ifft(dns_1.step()))
    dns_2.IC(u0=perturbed_state.flatten())
    perturbed_state_evolved = np.real(ifft(dns_2.step()))

    # Compute the Euclidean distance between the perturbed and evolved states
    p = (perturbed_state_evolved - state_evoled)
    distance = np.linalg.norm(p)
        
    p /= distance
    perturbed_state = np.reshape(u[t+1], (1,-1)) + perturbation * p
    
    # Update the divergence array
    divergence[t] = np.log(distance / perturbation)/dt

# Calculate the average Lyapunov exponent
lyapunov_exponent = np.mean(divergence)
print("\nMAXIMUM LYAPUNOV EXPONENT = {:}".format(lyapunov_exponent))

data["MLE"] = lyapunov_exponent
with open(base_path + "/Simulation_Data/simulation_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data



