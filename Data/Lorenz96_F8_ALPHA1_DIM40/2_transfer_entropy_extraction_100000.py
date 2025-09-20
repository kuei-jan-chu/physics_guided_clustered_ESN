import os
import numpy as np
import pickle
import sys
sys.path.insert(0, "../Utils")
from lorenz96 import *
from add_noise import addNoise
from transfer_entropy import transfer_entropy_matrix
from standardize import standardize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RDIM=40
T=20
dt=0.01
F=8

base_path = "."

file_name = base_path + "/Simulation_Data/F"+str(F)+"_data.pickle"
with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["X"]

length_for_transfer_entropy = 100000

selected_sequence = u[:length_for_transfer_entropy, :].copy()  
selected_sequence = addNoise(selected_sequence, 5)
selected_sequence = standardize(selected_sequence)
TE_matrix = transfer_entropy_matrix(selected_sequence, 5, 1)
print("Pairwise TE matrix:\n", TE_matrix)

data["transfer_entropy_matrix_from_length_{:}".format(length_for_transfer_entropy)] = TE_matrix
with open(base_path + "/Simulation_Data/F"+str(F)+"_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data

np.savetxt(base_path + "/Simulation_Data/transfer_entropy_matrix_from_length_{:}.txt".format(length_for_transfer_entropy), TE_matrix, fmt="%.6f", delimiter=",")

 
true_offsets = np.array([-2, -1, 1])
dim = TE_matrix.shape[0]
correct_count = 0
coupling_matrix = np.zeros((dim, dim))

for target in range(dim):
    # TE values for current target (incoming influences)
    te_values = TE_matrix[target, :]

    # Exclude self-connection if needed:
    # te_values[target] = -np.inf
    # no need as self-connection is set as zero in the transfer matrix

    # Get indices of top-3 largest TE values
    top_sources = np.argsort(te_values)[-3:]  # ascending → last 3 are largest

    # Set self-coupling to 2
    coupling_matrix[target, target] = 2

    # Set inferred coupled dimensions for the target (# of coupled dimensions is truely 3)
    coupling_matrix[target, top_sources] = 1

    # Compute expected true sources for this target based on true_offsets
    expected_sources = (target + true_offsets) % dim

    # Compare detected sources with expected sources
    correct_count += sum([s in expected_sources for s in top_sources])


accuracy = correct_count / (3*dim) * 100
print(f"Correct coupling recovery accuracy: {accuracy:.2f}%")


# Plotting the TE matrix
title = "From observed sequence of length {:}".format(length_for_transfer_entropy)
save_path = base_path + "/Figures/transfer_entropy_matrix_plot.pdf"
plt.figure(figsize=(6, 5))
im = plt.imshow(TE_matrix, cmap='turbo', origin='lower', interpolation='nearest', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label("Transfer Entropy")

# Add labels and ticks
num_dims = TE_matrix.shape[0]
plt.title(title, fontsize=14)
plt.xlabel("Sender dimension", fontsize=14)
plt.ylabel("Receiver dimension", fontsize=14)
tick_positions = np.linspace(0, num_dims - 1, 4, dtype=int)
plt.xticks(tick_positions, tick_positions + 1)
plt.yticks(tick_positions, tick_positions + 1)

plt.tight_layout()

plt.savefig(save_path, dpi=400)


# Plotting the coupling matrix
title_2 = "From observed sequence of length {:}".format(length_for_transfer_entropy)
save_path = base_path + "/Figures/coupling_matrix_plot_from_length_{:}.pdf".format(length_for_transfer_entropy)
plt.figure(figsize=(6, 5))

# Define custom color map: 0 = white, 1 = blue, 2 = green
# Define light colors
light_blue = "#5294CF"
light_green = "#93CB73"
# Custom colormap: 0 = white, 1 = light blue (coupled), 2 = light green (self)
cmap = ListedColormap(["white", light_blue, light_green])
im = plt.imshow(coupling_matrix, cmap=cmap, interpolation='nearest', origin='lower', aspect='equal')

# Add labels and ticks
num_dims = coupling_matrix.shape[0]
# plt.title(title_2, fontsize=14)
plt.xlabel("Sender dimension", fontsize=14)
plt.ylabel("Receiver dimension", fontsize=14)
tick_positions = np.linspace(0, num_dims - 1, 4, dtype=int)
plt.xticks(tick_positions, tick_positions + 1)
plt.yticks(tick_positions, tick_positions + 1)

plt.tight_layout()

plt.savefig(save_path, dpi=400)