import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams["text.usetex"] = False
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

import pickle

base_path = "."

# Ensure the directory exists
data_dir = base_path + "/Figures/"
os.makedirs(data_dir, exist_ok=True)

F=44

with open(base_path + "/Simulation_Data/F"+str(F)+"_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    X = data["X"]
    DIM = data["DIM"]
    dt = data["dt"]
    del data


for N_plot in [1000, 10000, 100000]:
    # Data Preparation:
    x_plot = X[:N_plot,:]
    N_plot = np.shape(x_plot)[0]
    # Figure Setup:
    fig = plt.subplots()
    # Grid Setup for Contour Plot:
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(DIM))+1)     # n for time steps, s for state dimension, + 1 to make it more interperetable
    # Contour Plot
    # display 15 distinct color regions, representing ranges of values within the data.
    # seismic color map diverge colormap with a center around a neutral color, often white or gray, representing a middle value (like zero or a mean). 
    # Shades of blue and red extend from this neutral center—typically, one color represents values below the center, and the other color represents values above it. 
    plt.contourf(s, n, np.transpose(x_plot), 15, cmap=plt.get_cmap("seismic"))  
    # Color Bar and Labels
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    # Saving the Plot
    plt.savefig(base_path + "/Figures/Plot_U_first_N{:d}.pdf".format(N_plot), bbox_inches="tight", dpi = 300)
    plt.close()

for N_plot in [1000, 10000, 100000]:
    x_plot = X[-N_plot:,:]
    N_plot = np.shape(x_plot)[0]
    # Plotting the contour plot
    fig = plt.subplots()
    n, s = np.meshgrid(np.arange(N_plot), np.array(range(DIM))+1)
    plt.contourf(s, n, np.transpose(x_plot), 15, cmap=plt.get_cmap("seismic"))
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    plt.savefig(base_path + "/Figures/Plot_U_last_N{:d}.pdf".format(N_plot), bbox_inches="tight", dpi = 300)
    plt.close()

